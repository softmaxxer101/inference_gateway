
import asyncio

import json

import os

import time

from contextlib import asynccontextmanager



import httpx

from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse, Response





# --------------------------------------------------

# Configuration

# --------------------------------------------------



VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")



MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "1"))

MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))



QUEUE_DEADLINE = float(os.getenv("QUEUE_DEADLINE", "5"))

UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "30"))



LOG_PATH = os.getenv("LOG_PATH", "results/requests.jsonl")





# --------------------------------------------------

# Shared state

# --------------------------------------------------



slots = asyncio.Semaphore(MAX_ACTIVE)



# Protects the queue-capacity admission check.

admission_lock = asyncio.Lock()



waiting = 0

active = 0



stats = {

    "received": 0,

    "completed": 0,

    "rejected_queue_full": 0,

    "rejected_deadline": 0,

    "upstream_timeout": 0,

    "upstream_error": 0,

}





# Ensure the log directory exists.

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)





# --------------------------------------------------

# Application lifecycle

# --------------------------------------------------



@asynccontextmanager

async def lifespan(app: FastAPI):

    app.state.client = httpx.AsyncClient(

        base_url=VLLM_URL,

        timeout=UPSTREAM_TIMEOUT,

    )



    try:

        yield

    finally:

        await app.state.client.aclose()





app = FastAPI(lifespan=lifespan)





# --------------------------------------------------

# Logging

# --------------------------------------------------



def log_event(

    outcome: str,

    status: int,

    queue_wait: float,

    service_time: float,

):

    event = {

        "ts": time.time(),

        "outcome": outcome,

        "status": status,

        "queue_wait_s": round(queue_wait, 4),

        "service_s": round(service_time, 4),

        "waiting_at_exit": waiting,

        "active_at_exit": active,

    }



    with open(LOG_PATH, "a", encoding="utf-8") as f:

        f.write(json.dumps(event) + "\n")





# --------------------------------------------------

# Rejection helper

# --------------------------------------------------



def reject(outcome: str, t0: float):

    log_event(

        outcome=outcome,

        status=429,

        queue_wait=time.perf_counter() - t0,

        service_time=0.0,

    )



    return JSONResponse(

        status_code=429,

        content={

            "error": "server overloaded",

            "reason": outcome,

        },

        headers={"Retry-After": "1"},

    )





# --------------------------------------------------

# Chat completions endpoint

# --------------------------------------------------



@app.post("/v1/chat/completions")

async def chat_completions(request: Request):

    global waiting, active



    stats["received"] += 1



    t0 = time.perf_counter()



    # Parse the incoming JSON request.

    try:

        body = await request.json()

    except Exception:

        return JSONResponse(

            status_code=400,

            content={"error": "invalid JSON request body"},

        )



    # Streaming is not supported by this gateway yet.

    if body.get("stream", False):

        return JSONResponse(

            status_code=400,

            content={

                "error": "streaming is not supported by this gateway yet"

            },

        )



    # --------------------------------------------------

    # 1. Atomically check and reserve admission capacity.

    # --------------------------------------------------



    async with admission_lock:

        total_capacity = MAX_ACTIVE + MAX_QUEUE



        if active + waiting >= total_capacity:

            stats["rejected_queue_full"] += 1

            return reject("queue_full", t0)



        waiting += 1



    # --------------------------------------------------

    # 2. Wait for an active-request slot.

    # --------------------------------------------------



    acquired = False



    try:

        try:

            async with asyncio.timeout(QUEUE_DEADLINE):

                await slots.acquire()



            acquired = True



        except TimeoutError:

            stats["rejected_deadline"] += 1

            return reject("queue_deadline", t0)



        finally:

            # This request is no longer waiting, whether it

            # acquired a slot, timed out, or was cancelled.

            waiting -= 1



        # --------------------------------------------------

        # 3. Forward the request to vLLM.

        # --------------------------------------------------



        queue_wait = time.perf_counter() - t0

        t1 = time.perf_counter()



        active += 1



        try:

            upstream = await request.app.state.client.post(

                "/v1/chat/completions",

                json=body,

            )



            service_time = time.perf_counter() - t1



            stats["completed"] += 1



            log_event(

                outcome="ok",

                status=upstream.status_code,

                queue_wait=queue_wait,

                service_time=service_time,

            )



            return Response(

                content=upstream.content,

                status_code=upstream.status_code,

                media_type=upstream.headers.get(

                    "content-type",

                    "application/json",

                ),

            )



        except httpx.TimeoutException:

            service_time = time.perf_counter() - t1



            stats["upstream_timeout"] += 1



            log_event(

                outcome="upstream_timeout",

                status=504,

                queue_wait=queue_wait,

                service_time=service_time,

            )



            return JSONResponse(

                status_code=504,

                content={"error": "upstream inference request timed out"},

            )



        except httpx.TransportError:

            service_time = time.perf_counter() - t1



            stats["upstream_error"] += 1



            log_event(

                outcome="upstream_error",

                status=502,

                queue_wait=queue_wait,

                service_time=service_time,

            )



            return JSONResponse(

                status_code=502,

                content={"error": "could not connect to inference server"},

            )



        finally:

            active -= 1



    finally:

        # Release a slot only if this request actually acquired it.

        if acquired:

            slots.release()





# --------------------------------------------------

# Health endpoint

# --------------------------------------------------



@app.get("/health")

async def health(request: Request):

    try:

        response = await request.app.state.client.get(

            "/health",

            timeout=2.0,

        )



        if response.is_success:

            return {

                "gateway": "up",

                "vllm": "up",

            }



        return JSONResponse(

            status_code=503,

            content={

                "gateway": "up",

                "vllm": "down",

                "upstream_status": response.status_code,

            },

        )



    except httpx.HTTPError:

        return JSONResponse(

            status_code=503,

            content={

                "gateway": "up",

                "vllm": "down",

            },

        )





# --------------------------------------------------

# Metrics endpoint

# --------------------------------------------------



@app.get("/metrics")

async def metrics():

    return {

        **stats,

        "waiting_now": waiting,

        "active_now": active,

        "max_active": MAX_ACTIVE,

        "max_queue": MAX_QUEUE,

    }

