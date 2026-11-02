# bench4.py

"""Phase 4 reliability bench — self-contained.



Starts a fault-injectable fake vLLM (in a thread) and the gateway (subprocess),

runs burst / disconnect / timeout / retry / outage, prints PASS/FAIL,

and appends rows to results/phase4_summary.csv.



    python bench4.py all

    python bench4.py outage --label run2

"""

import argparse

import asyncio

import csv

import os

import random

import statistics

import subprocess

import sys

import threading

import time

import uuid

from collections import Counter



import httpx

import uvicorn

from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse, Response



# ------------------------------ fake vLLM ------------------------------

FAULT_DEFAULTS = {"delay_per_token": 1 / 400, "extra_delay": 0.0,

                  "fail_next": 0, "fail_status": 503}

FAULT = dict(FAULT_DEFAULTS)

COUNTS = Counter()

STARTED = Counter()

COMPLETED = Counter()



fake_app = FastAPI()





@fake_app.post("/v1/chat/completions")

async def fake_chat(r: Request):

    body = await r.json()

    rid = r.headers.get("x-request-id", "?")

    STARTED[rid] += 1

    COUNTS["started"] += 1

    if FAULT["fail_next"] > 0:

        FAULT["fail_next"] -= 1

        COUNTS["failed"] += 1

        return JSONResponse({"error": "injected"}, status_code=int(FAULT["fail_status"]))

    end = time.monotonic() + body.get("max_tokens", 16) * FAULT["delay_per_token"] + FAULT["extra_delay"]

    while time.monotonic() < end:

        if await r.is_disconnected():

            COUNTS["aborted"] += 1

            return Response(status_code=499)

        await asyncio.sleep(0.05)

    COUNTS["completed"] += 1

    COMPLETED[rid] += 1

    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}





@fake_app.get("/health")

async def fake_health():

    return {}





def reset_fake():

    FAULT.update(FAULT_DEFAULTS)

    COUNTS.clear(); STARTED.clear(); COMPLETED.clear()





class FakeVLLM:

    def __init__(self, port):

        self.port = port

        self.server = None

        self.thread = None

        self._lock = threading.Lock()



    def start(self):

        with self._lock:

            if self.thread and self.thread.is_alive():

                return

            cfg = uvicorn.Config(fake_app, host="127.0.0.1", port=self.port,

                                 log_level="warning", access_log=False)

            self.server = uvicorn.Server(cfg)

            self.server.install_signal_handlers = lambda: None   # thread-safe no-op

            self.thread = threading.Thread(target=self.server.run, daemon=True, name="fake-vllm")

            self.thread.start()

        for _ in range(500):

            if self.server.started:

                return

            if not self.thread.is_alive():

                raise RuntimeError("fake vLLM thread died during startup")

            time.sleep(0.01)

        raise RuntimeError("fake vLLM failed to start")



    def stop(self):

        with self._lock:

            if self.server is None:

                return

            self.server.should_exit = True

            self.server.force_exit = True                      # abrupt: in-flight reset

            if self.thread:

                self.thread.join(timeout=5)

            self.server = None

            self.thread = None



    @property

    def url(self):

        return f"http://127.0.0.1:{self.port}"





# ------------------------------ gateway / helpers ------------------------------

ROWS, CHECKS = [], []

LABEL = ""

SCENARIO_NAMES = ["burst", "disconnect", "timeout", "retry", "outage"]





def spawn_gateway(port, env):

    return subprocess.Popen(

        [sys.executable, "-m", "uvicorn", "gateway4:app",

         "--port", str(port), "--log-level", "warning", "--no-access-log"],

        env=env,

    )





async def wait_until_up(gw_url, timeout=15):

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < timeout:

        try:

            async with httpx.AsyncClient(timeout=1) as c:

                r = await c.get(gw_url + "/metrics")

            if r.status_code == 200:

                return True

        except httpx.HTTPError:

            pass

        await asyncio.sleep(0.2)

    return False





async def wait_vllm_seen_up(gw_url, timeout=15):

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < timeout:

        try:

            async with httpx.AsyncClient(timeout=1) as c:

                m = (await c.get(gw_url + "/metrics")).json()

            if m.get("vllm_up"):

                return True

        except httpx.HTTPError:

            pass

        await asyncio.sleep(0.2)

    return False





async def get_metrics(gw_url):

    async with httpx.AsyncClient(timeout=5) as c:

        return (await c.get(gw_url + "/metrics")).json()





async def wait_idle(gw_url, timeout=15):

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < timeout:

        m = await get_metrics(gw_url)

        if m["vllm_up"] and m["active_now"] == 0 and m["waiting_now"] == 0:

            return True

        await asyncio.sleep(0.1)

    return False





async def reset_all(gw_url):

    reset_fake()

    await wait_idle(gw_url)





def payload(model, tokens):

    return {"model": model,

            "messages": [{"role": "user", "content": "Write about queues."}],

            "max_tokens": tokens, "min_tokens": tokens, "temperature": 0}





async def send(gw_url, model, tokens, rid=None, read_timeout=60.0):

    rid = rid or uuid.uuid4().hex[:12]

    t = time.perf_counter()

    try:

        async with httpx.AsyncClient(timeout=read_timeout) as c:

            r = await c.post(f"{gw_url}/v1/chat/completions", json=payload(model, tokens),

                             headers={"X-Request-Id": rid})

        status = r.status_code

    except httpx.TimeoutException:

        status = "client_timeout"

    except httpx.HTTPError:

        status = "conn_error"

    return {"rid": rid, "status": status, "ms": (time.perf_counter() - t) * 1000, "t": t}





def rates(rs):

    n = len(rs) or 1

    c = Counter("ok" if r["status"] == 200 else "shed" if r["status"] == 429

                else "timeout" if r["status"] == 504 else "unhealthy" if r["status"] == 503

                else "error" for r in rs)

    return {"success_rate": c["ok"] / n, "shed_rate": c["shed"] / n,

            "timeout_rate": c["timeout"] / n, "unhealthy_rate": c["unhealthy"] / n,

            "error_rate": c["error"] / n}





def metric(scenario, name, value):

    value = round(value, 3) if isinstance(value, float) else value

    print(f"  {name:34} {value}")

    ROWS.append({"label": LABEL, "scenario": scenario, "metric": name, "value": value})





def check(scenario, name, cond, detail=""):

    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    CHECKS.append(bool(cond))





# ------------------------------ scenarios ------------------------------

async def scenario_burst(gw_url, args):

    s = "burst"

    print("\n== burst: steady -> spike -> steady ==")

    await reset_all(gw_url)

    cfg = (await get_metrics(gw_url))["config"]

    rng = random.Random(args.seed)

    plan = [("before", args.base_rate, 5.0), ("burst", args.burst_rate, 3.0),

            ("after", args.base_rate, 5.0)]

    burst_end = plan[0][2] + plan[1][2]

    arrivals, t = [], 0.0

    for name, rate, dur in plan:

        end = t + dur

        while True:

            t += rng.expovariate(rate)

            if t >= end:

                break

            arrivals.append((t, name))

        t = end



    start = time.perf_counter()

    seen = {"max_wait": 0, "drained": None}



    async def poll():

        while True:

            m = await get_metrics(gw_url)

            seen["max_wait"] = max(seen["max_wait"], m["waiting_now"])

            el = time.perf_counter() - start

            if el > burst_end and seen["drained"] is None and m["waiting_now"] == 0:

                seen["drained"] = el

            await asyncio.sleep(0.1)



    async def fire(at, name):

        await asyncio.sleep(max(0.0, at - (time.perf_counter() - start)))

        r = await send(gw_url, args.model, args.tokens)

        r["phase"] = name

        return r



    poller = asyncio.create_task(poll())

    results = await asyncio.gather(*[fire(at, n) for at, n in arrivals])

    poller.cancel()



    for k, v in rates(results).items():

        metric(s, k, v)

    for name, _, _ in plan:

        rs = [r for r in results if r["phase"] == name]

        metric(s, f"{name}_success_rate", rates(rs)["success_rate"])

    metric(s, "max_queue_depth_seen", seen["max_wait"])

    metric(s, "recovery_s_after_burst",

           (seen["drained"] or float("nan")) - burst_end)

    during = [r for r in results if r["phase"] == "burst"]

    after_ok = rates([r for r in results if r["phase"] == "after"])["success_rate"]

    check(s, "queue stayed bounded", seen["max_wait"] <= cfg["max_queue"],

          f"{seen['max_wait']} <= {cfg['max_queue']}")

    check(s, "burst overflow shed with 429, no errors/timeouts",

          all(r["status"] in (200, 429) for r in during))

    check(s, "traffic after burst served normally", after_ok >= 0.95, f"{after_ok:.0%}")





async def scenario_disconnect(gw_url, args):

    s = "disconnect"

    print("\n== disconnect: clients hang up while queued / running ==")

    await reset_all(gw_url)

    cfg = (await get_metrics(gw_url))["config"]

    await asyncio.gather(*[send(gw_url, args.model, 400, read_timeout=0.3) for _ in range(5)])

    await asyncio.sleep(1.5)

    m = await get_metrics(gw_url)

    running = min(5, cfg["max_active"])

    metric(s, "disconnects_detected", m["client_disconnected"])

    metric(s, "requests_that_reached_vllm", COUNTS["started"])

    metric(s, "aborted_at_vllm", COUNTS["aborted"])

    check(s, "gateway noticed all 5 disconnects", m["client_disconnected"] == 5)

    check(s, "queued requests never reached vLLM", COUNTS["started"] == running,

          f"{COUNTS['started']} == {running}")

    check(s, "running request aborted at vLLM (GPU freed)", COUNTS["aborted"] == running)

    check(s, "no leaked slots or queue entries",

          m["active_now"] == 0 and m["waiting_now"] == 0)

    follow = await send(gw_url, args.model, 32)

    check(s, "next request served normally",

          follow["status"] == 200 and follow["ms"] < 1500, f"{follow['ms']:.0f} ms")





async def scenario_timeout(gw_url, args):

    s = "timeout"

    print("\n== timeout: vLLM slower than UPSTREAM_TIMEOUT ==")

    await reset_all(gw_url)

    cfg = (await get_metrics(gw_url))["config"]

    FAULT["extra_delay"] = cfg["upstream_timeout"] + 2

    m0 = await get_metrics(gw_url)

    rs = await asyncio.gather(*[send(gw_url, args.model, 32) for _ in range(3)])

    await asyncio.sleep(0.5)

    m1 = await get_metrics(gw_url)

    for k, v in rates(rs).items():

        metric(s, k, v)

    metric(s, "attempts_at_vllm", COUNTS["started"])

    check(s, "every request got 504", all(r["status"] == 504 for r in rs))

    check(s, "timeouts are not retried",

          COUNTS["started"] == 3 and m1["retries"] == m0["retries"])

    check(s, "vLLM work aborted after the gateway gave up", COUNTS["aborted"] == 3)

    check(s, "no leaked slots", m1["active_now"] == 0 and m1["waiting_now"] == 0)

    FAULT["extra_delay"] = 0

    follow = await send(gw_url, args.model, 32)

    check(s, "gateway healthy again afterwards", follow["status"] == 200)





async def scenario_retry(gw_url, args):

    s = "retry"

    print("\n== retry: transient 503s and retry-storm control ==")

    await reset_all(gw_url)

    cfg = (await get_metrics(gw_url))["config"]

    m0 = await get_metrics(gw_url)

    FAULT["fail_next"] = 2

    FAULT["fail_status"] = 503

    rid = uuid.uuid4().hex[:12]

    r = await send(gw_url, args.model, 32, rid=rid)

    m1 = await get_metrics(gw_url)

    metric(s, "attempts_for_one_request", STARTED[rid])

    metric(s, "completed_executions", COMPLETED[rid])

    check(s, "transient failure hidden from client", r["status"] == 200)

    check(s, "retried exactly twice", m1["retries"] - m0["retries"] == 2)

    check(s, "no duplicate work", COMPLETED[rid] == 1)



    FAULT["fail_next"] = 1000

    n = 20

    await asyncio.gather(*[send(gw_url, args.model, 32) for _ in range(n)])

    m2 = await get_metrics(gw_url)

    spent = m2["retries"] - m1["retries"]

    denied = m2["retries_denied"] - m1["retries_denied"]

    naive = n * cfg["max_retries"]

    metric(s, "retries_during_storm", spent)

    metric(s, "retries_denied_by_budget", denied)

    metric(s, "upstream_calls_per_request", (n + spent) / n)

    check(s, "retry budget capped amplification",

          spent < naive and denied > 0, f"{spent} retries vs {naive} without budget")

    check(s, "no request ever completed twice",

          all(v <= 1 for v in COMPLETED.values()))





async def scenario_outage(gw_url, fake, args):

    s = "outage"

    print("\n== outage: vLLM crashes mid-traffic, then recovers ==")

    await reset_all(gw_url)

    cfg = (await get_metrics(gw_url))["config"]

    m0 = await get_metrics(gw_url)

    ev = {}

    start = time.perf_counter()



    async def chaos():

        await asyncio.sleep(3)

        await asyncio.to_thread(fake.stop)

        ev["down"] = time.perf_counter()

        await asyncio.sleep(5)

        await asyncio.to_thread(fake.start)

        ev["up"] = time.perf_counter()



    async def watch():

        while True:

            try:

                async with httpx.AsyncClient(timeout=2) as c:

                    r = await c.get(gw_url + "/health")

                now = time.perf_counter()

                if "down" in ev and "gw_down" not in ev and r.status_code == 503:

                    ev["gw_down"] = now

                if "up" in ev and "gw_up" not in ev and r.status_code == 200:

                    ev["gw_up"] = now

            except httpx.HTTPError:

                pass

            await asyncio.sleep(0.1)



    async def fire(i):

        await asyncio.sleep(max(0.0, i * 0.5 - (time.perf_counter() - start)))

        return await send(gw_url, args.model, 32)



    helpers = [asyncio.create_task(chaos()), asyncio.create_task(watch())]

    results = await asyncio.gather(*[fire(i) for i in range(28)])   # 2 req/s for 14 s

    await helpers[0]

    await asyncio.sleep(0.5)

    helpers[1].cancel()

    m1 = await get_metrics(gw_url)



    for k, v in rates(results).items():

        metric(s, k, v)

    metric(s, "status_503_failfast", sum(r["status"] == 503 for r in results))

    metric(s, "status_502_connection_errors", sum(r["status"] == 502 for r in results))

    detect = ev.get("gw_down", float("nan")) - ev["down"]

    recover = ev.get("gw_up", float("nan")) - ev["up"]

    after_up = [r for r in results if r["status"] == 200 and r["t"] >= ev["up"]]

    client_recover = ((min(r["t"] + r["ms"] / 1000 for r in after_up) - ev["up"])

                      if after_up else float("nan"))

    metric(s, "detection_s (gateway marks down)", detect)

    metric(s, "recovery_s (gateway marks up)", recover)

    metric(s, "recovery_s (first client success)", client_recover)

    ff = [r["ms"] for r in results if r["status"] == 503]

    if ff:

        metric(s, "failfast_median_ms", statistics.median(ff))

    metric(s, "retries", m1["retries"] - m0["retries"])

    check(s, "outage detected quickly",

          detect <= cfg["fail_threshold"] * cfg["health_interval"] + 1.5, f"{detect:.2f}s")

    check(s, "requests fail fast while down", not ff or statistics.median(ff) < 500)

    check(s, "service recovered by itself", bool(after_up))

    check(s, "no request completed twice",

          all(v <= 1 for v in COMPLETED.values()))





# ------------------------------ entrypoint ------------------------------

async def dispatch(name, gw_url, fake, args):

    if name == "burst":       await scenario_burst(gw_url, args)

    elif name == "disconnect": await scenario_disconnect(gw_url, args)

    elif name == "timeout":    await scenario_timeout(gw_url, args)

    elif name == "retry":      await scenario_retry(gw_url, args)

    elif name == "outage":     await scenario_outage(gw_url, fake, args)





async def run(args):

    global LABEL

    LABEL = args.label

    fake = FakeVLLM(args.upstream_port)

    fake.start()

    env = {**os.environ,

           "VLLM_URL": fake.url,

           "MAX_ACTIVE": "2", "MAX_QUEUE": "4",

           "UPSTREAM_TIMEOUT": "5", "HEALTH_INTERVAL": "0.25", "FAIL_THRESHOLD": "2",

           "RETRY_BACKOFF": "0.1", "LOG_PATH": "results/phase4_requests.jsonl"}

    gw = spawn_gateway(args.gateway_port, env)

    gw_url = f"http://127.0.0.1:{args.gateway_port}"

    try:

        if not await wait_until_up(gw_url, timeout=15):

            print("gateway did not come up"); return

        if not await wait_vllm_seen_up(gw_url, timeout=15):

            print("gateway never saw vLLM up"); return

        names = SCENARIO_NAMES if args.scenario == "all" else [args.scenario]

        for name in names:

            await dispatch(name, gw_url, fake, args)

        print(f"\n{sum(CHECKS)}/{len(CHECKS)} checks passed")

        os.makedirs("results", exist_ok=True)

        path = "results/phase4_summary.csv"

        new = not os.path.exists(path)

        with open(path, "a", newline="") as f:

            w = csv.DictWriter(f, fieldnames=["label", "scenario", "metric", "value"])

            if new:

                w.writeheader()

            w.writerows(ROWS)

    finally:

        gw.terminate()

        try:

            gw.wait(timeout=5)

        except subprocess.TimeoutExpired:

            gw.kill()

        fake.stop()





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("scenario", choices=["all", *SCENARIO_NAMES])

    p.add_argument("--label", default="")

    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")

    p.add_argument("--tokens", type=int, default=160)

    p.add_argument("--base-rate", type=float, default=1.0)

    p.add_argument("--burst-rate", type=float, default=15.0)

    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--upstream-port", type=int, default=8000)

    p.add_argument("--gateway-port", type=int, default=9000)

    return p.parse_args()





if __name__ == "__main__":

    asyncio.run(run(parse_args()))
