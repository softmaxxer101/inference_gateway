# gateway4.py

"""Phase 4 gateway: reliable inference proxy.



    uvicorn gateway4:app --port 9000



Reliability features:

  * bounded FIFO queue with a queue-wait deadline

  * background health probe; fails fast with 503 while vLLM is unreachable

  * client disconnects cancel queued or in-flight work (so no wasted GPU work)

  * controlled retries only when vLLM provably did no work (connect error / 503),

    with exponential backoff and a token-bucket retry budget

  * every request gets an X-Request-Id, forwarded upstream, logged and returned

"""

import asyncio

import json

import os

import random

import time

import uuid

from collections import deque

from contextlib import asynccontextmanager



import httpx

from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse, Response



VLLM_URL         = os.getenv("VLLM_URL", "http://localhost:8000")

MAX_ACTIVE       = int(os.getenv("MAX_ACTIVE", "1"))

MAX_QUEUE        = int(os.getenv("MAX_QUEUE", "8"))

QUEUE_DEADLINE   = float(os.getenv("QUEUE_DEADLINE", "30"))

UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "60"))

CONNECT_TIMEOUT  = float(os.getenv("CONNECT_TIMEOUT", "2"))

HEALTH_INTERVAL  = float(os.getenv("HEALTH_INTERVAL", "1"))

FAIL_THRESHOLD   = int(os.getenv("FAIL_THRESHOLD", "2"))

MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "2"))

RETRY_BACKOFF    = float(os.getenv("RETRY_BACKOFF", "0.1"))

RETRY_RATIO      = float(os.getenv("RETRY_RATIO", "0.05"))

LOG_PATH         = os.getenv("LOG_PATH", "results/phase4_requests.jsonl")





class Scheduler:

    def __init__(self):

        self.q = deque()

        self.active = 0

        self.waiting = 0



    def _dispatch(self):

        while self.active < MAX_ACTIVE and self.q:

            fut = self.q.popleft()

            if fut.done():

                continue                       # canceller already decremented waiting

            self.active += 1

            self.waiting -= 1

            fut.set_result(None)



    async def acquire(self, wait_limit):

        if self.waiting >= MAX_QUEUE:

            return "full"

        fut = asyncio.get_running_loop().create_future()

        self.q.append(fut)

        self.waiting += 1

        self._dispatch()

        try:

            async with asyncio.timeout(wait_limit):

                await fut

            return "ok"

        except TimeoutError:

            if fut.done() and not fut.cancelled():

                return "ok"                    # granted at the last instant

            fut.cancel()

            try: self.q.remove(fut)

            except ValueError: pass

            self.waiting -= 1

            return "deadline"

        except asyncio.CancelledError:

            if fut.done() and not fut.cancelled():

                self.release()

            else:

                fut.cancel()

                try: self.q.remove(fut)

                except ValueError: pass

                self.waiting -= 1

            raise



    def release(self):

        self.active -= 1

        self._dispatch()





class Health:

    def __init__(self):

        self.up, self.fails, self.down_since, self.outages = True, 0, None, []



    def ok(self):

        self.fails = 0

        if not self.up:

            self.up = True

            self.outages.append(round(time.time() - self.down_since, 2))

            self.down_since = None



    def fail(self):

        self.fails += 1

        if self.up and self.fails >= FAIL_THRESHOLD:

            self.up, self.down_since = False, time.time()





sched, health = Scheduler(), Health()

retry_tokens = 2.0

stats = {k: 0 for k in ["received", "completed", "rejected_queue_full",

                        "rejected_deadline", "rejected_unhealthy", "upstream_timeout",

                        "upstream_error", "client_disconnected", "retries", "retries_denied"]}

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)





def spend_retry():

    global retry_tokens

    if retry_tokens >= 1:

        retry_tokens -= 1

        stats["retries"] += 1

        return True

    stats["retries_denied"] += 1

    return False





def log_event(rid, outcome, status, t0, queue_wait=0.0, service=0.0, attempts=0):

    with open(LOG_PATH, "a") as f:

        f.write(json.dumps({"ts": time.time(), "id": rid, "outcome": outcome, "status": status,

                            "attempts": attempts, "queue_wait_s": round(queue_wait, 4),

                            "service_s": round(service, 4),

                            "total_s": round(time.perf_counter() - t0, 4)}) + "\n")





def fail(outcome, status, rid, t0, queue_wait=0.0, service=0.0, attempts=0, retry_after=None):

    stats[outcome] += 1

    log_event(rid, outcome, status, t0, queue_wait, service, attempts)

    headers = {"X-Request-Id": rid}

    if retry_after:

        headers["Retry-After"] = str(retry_after)

    return JSONResponse({"error": outcome, "request_id": rid},

                        status_code=status, headers=headers)





async def forward(client, body, rid):

    """Retry ONLY when vLLM provably did no work (connect error / 503)."""

    attempts = 0

    while True:

        attempts += 1

        can_retry = attempts <= MAX_RETRIES

        try:

            r = await client.post("/v1/chat/completions", json=body,

                                  headers={"X-Request-Id": rid})

        except (httpx.ConnectError, httpx.ConnectTimeout):

            health.fail()

            if can_retry and health.up and spend_retry():

                await asyncio.sleep(RETRY_BACKOFF * 2 ** (attempts - 1) * (0.5 + random.random()))

                continue

            return "unreachable", None, attempts

        except httpx.TimeoutException:

            return "timeout", None, attempts

        except httpx.TransportError:

            health.fail()

            return "broken", None, attempts

        if r.status_code == 503 and can_retry and spend_retry():

            await asyncio.sleep(RETRY_BACKOFF * 2 ** (attempts - 1) * (0.5 + random.random()))

            continue

        if r.status_code < 500:

            health.ok()

        return "ok", r, attempts





async def process(client, body, rid, t0):

    status = await sched.acquire(QUEUE_DEADLINE)

    if status == "full":

        return fail("rejected_queue_full", 429, rid, t0, retry_after=1)

    if status == "deadline":

        return fail("rejected_deadline", 429, rid, t0, retry_after=1)

    queue_wait = time.perf_counter() - t0

    t1 = time.perf_counter()

    try:

        if not health.up:                          # vLLM died while we queued

            return fail("rejected_unhealthy", 503, rid, t0, queue_wait, retry_after=2)

        kind, r, attempts = await forward(client, body, rid)

        service = time.perf_counter() - t1

        if kind == "timeout":

            return fail("upstream_timeout", 504, rid, t0, queue_wait, service, attempts)

        if kind != "ok":

            return fail("upstream_error", 502, rid, t0, queue_wait, service, attempts)

        if r.status_code == 200:

            stats["completed"] += 1

        log_event(rid, "ok" if r.status_code == 200 else "upstream_http_error",

                  r.status_code, t0, queue_wait, service, attempts)

        return Response(content=r.content, status_code=r.status_code,

                        media_type="application/json",

                        headers={"X-Request-Id": rid})

    finally:

        sched.release()





async def run_or_disconnect(request, coro):

    work = asyncio.ensure_future(coro)



    async def gone():

        while not await request.is_disconnected():

            await asyncio.sleep(0.1)



    watcher = asyncio.ensure_future(gone())

    try:

        await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)

        if work.done():

            return work.result()

        work.cancel()                              # closes httpx -> vLLM aborts

        await asyncio.gather(work, return_exceptions=True)

        return None

    finally:

        watcher.cancel()

        if not work.done():

            work.cancel()





async def probe_loop(client):

    while True:

        try:

            r = await client.get("/health", timeout=1.0)

            health.ok() if r.status_code == 200 else health.fail()

        except httpx.HTTPError:

            health.fail()

        await asyncio.sleep(HEALTH_INTERVAL)





@asynccontextmanager

async def lifespan(app):

    client = httpx.AsyncClient(base_url=VLLM_URL,

                               timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=CONNECT_TIMEOUT))

    app.state.client = client

    probe = asyncio.create_task(probe_loop(client))

    yield

    probe.cancel()

    await client.aclose()





app = FastAPI(lifespan=lifespan)





@app.post("/v1/chat/completions")

async def chat(request: Request):

    global retry_tokens

    stats["received"] += 1

    retry_tokens = min(10.0, retry_tokens + RETRY_RATIO)

    t0 = time.perf_counter()

    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]

    try:

        body = await request.json()

    except Exception:

        return JSONResponse({"error": "invalid JSON"}, status_code=400)



    if not health.up:

        return fail("rejected_unhealthy", 503, rid, t0, retry_after=2)



    resp = await run_or_disconnect(request, process(request.app.state.client, body, rid, t0))

    if resp is None:

        return fail("client_disconnected", 499, rid, t0)

    return resp





@app.get("/health")

async def gateway_health():

    return JSONResponse({"gateway": "up", "vllm": "up" if health.up else "down",

                         "active": sched.active, "waiting": sched.waiting},

                        status_code=200 if health.up else 503)





@app.get("/metrics")

async def metrics():

    return {**stats, "active_now": sched.active, "waiting_now": sched.waiting,

            "vllm_up": health.up, "down_since": health.down_since,

            "outages_s": health.outages[-5:], "retry_tokens": round(retry_tokens, 2),

            "config": {"max_active": MAX_ACTIVE, "max_queue": MAX_QUEUE,

                       "upstream_timeout": UPSTREAM_TIMEOUT, "max_retries": MAX_RETRIES,

                       "retry_ratio": RETRY_RATIO, "health_interval": HEALTH_INTERVAL,

                       "fail_threshold": FAIL_THRESHOLD}}
