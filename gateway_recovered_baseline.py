import asyncio

import json

import math

import os

import time

from collections import deque

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



SLA_FALLBACK_SERVICE_SECONDS = float(

    os.getenv("SLA_FALLBACK_SERVICE_SECONDS", "1.5")

)

SERVICE_HISTORY_SIZE = int(os.getenv("SERVICE_HISTORY_SIZE", "100"))

SLA_HEADER = "X-Request-Deadline-Ms"



if MAX_ACTIVE < 1:

    raise ValueError("MAX_ACTIVE must be at least 1")

if MAX_QUEUE < 0:

    raise ValueError("MAX_QUEUE cannot be negative")

if SERVICE_HISTORY_SIZE < 1:

    raise ValueError("SERVICE_HISTORY_SIZE must be at least 1")



# --------------------------------------------------

# Shared state

# --------------------------------------------------



slots = asyncio.Semaphore(MAX_ACTIVE)

admission_lock = asyncio.Lock()



waiting = 0

active = 0



service_time_history = deque(maxlen=SERVICE_HISTORY_SIZE)



stats = {

    "received": 0,

    "completed": 0,

    "rejected_queue_full": 0,

    "rejected_deadline": 0,

    "rejected_sla": 0,

    "upstream_timeout": 0,

    "upstream_error": 0,

    "sla_requests": 0,

    "sla_completed_within_budget": 0,

    "sla_completed_over_budget": 0,

}



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

# SLA estimation

# --------------------------------------------------



def estimate_service_time() -> float:

    """Use fallback before observations; then nearest-rank P90."""

    if not service_time_history:

        return SLA_FALLBACK_SERVICE_SECONDS



    observations = sorted(service_time_history)

    index = max(0, math.ceil(0.90 * len(observations)) - 1)

    return observations[index]





def estimate_queue_wait(

    active_requests: int,

    waiting_requests: int,

    service_seconds: float,

) -> float:

    """

    Estimate wait for a new request to obtain an active slot.

    Assumes similar service durations and FIFO-like slot usage.

    """

    requests_ahead = active_requests + waiting_requests

    work_before_slot = max(

        0,

        requests_ahead - MAX_ACTIVE + 1,

    )

    waves = math.ceil(work_before_slot / MAX_ACTIVE)

    return waves * service_seconds





def estimate_completion_time(

    active_requests: int,

    waiting_requests: int,

):

    """Return queue wait, service estimate, and total prediction."""

    service_seconds = estimate_service_time()

    queue_wait_seconds = estimate_queue_wait(

        active_requests=active_requests,

        waiting_requests=waiting_requests,

        service_seconds=service_seconds,

    )

    return (

        queue_wait_seconds,

        service_seconds,

        queue_wait_seconds + service_seconds,

    )





# --------------------------------------------------

# Request logging

# --------------------------------------------------



def log_event(

    outcome: str,

    status: int,

    queue_wait: float,

    service_time: float,

    total_latency: float,

    sla_budget_ms=None,

    remaining_budget_ms=None,

    predicted_completion_ms=None,

    estimated_service_time_ms=None,

    predicted_queue_wait_ms=None,

    active_at_admission=None,

    waiting_at_admission=None,

):

    """Append one JSON record to the request log."""

    record = {

        "timestamp": time.time(),

        "outcome": outcome,

        "status": status,

        "queue_wait": round(queue_wait, 6),

        "service_time": round(service_time, 6),

        "total_latency": round(total_latency, 6),

        "sla_budget_ms": sla_budget_ms,

        "remaining_budget_ms": remaining_budget_ms,

        "predicted_completion_ms": predicted_completion_ms,

        "estimated_service_time_ms": estimated_service_time_ms,

        "predicted_queue_wait_ms": predicted_queue_wait_ms,

        "active_at_admission": active_at_admission,

        "waiting_at_admission": waiting_at_admission,

    }



    with open(LOG_PATH, "a", encoding="utf-8") as file:

        file.write(json.dumps(record) + "\n")





# --------------------------------------------------

# Rejection helper

# --------------------------------------------------



def reject(outcome: str, t0: float, extra=None):

    """

    Return HTTP 429 and log supplied SLA details.

    This includes the budget and prediction fields for SLA rejections.

    """

    elapsed = time.perf_counter() - t0

    extra = extra or {}



    log_event(

        outcome=outcome,

        status=429,

        queue_wait=elapsed,

        service_time=0.0,

        total_latency=elapsed,

        sla_budget_ms=extra.get("sla_budget_ms"),

        remaining_budget_ms=extra.get("remaining_budget_ms"),

        predicted_completion_ms=extra.get("predicted_completion_ms"),

        estimated_service_time_ms=extra.get("estimated_service_time_ms"),

        predicted_queue_wait_ms=extra.get("predicted_queue_wait_ms"),

        active_at_admission=extra.get("active_at_admission"),

        waiting_at_admission=extra.get("waiting_at_admission"),

    )



    content = {

        "error": "request rejected by gateway",

        "reason": outcome,

    }



    # Preserve SLA details in the response body as well.

    content.update(extra)



    return JSONResponse(

        status_code=429,

        content=content,

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



    # Parse incoming JSON.

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



    # Parse optional total-latency budget in milliseconds.

    sla_budget_ms = None

    header_value = request.headers.get(SLA_HEADER)



    if header_value is not None:

        try:

            sla_budget_ms = int(header_value)

            if sla_budget_ms <= 0:

                raise ValueError

        except (TypeError, ValueError):

            return JSONResponse(

                status_code=400,

                content={

                    "error": f"{SLA_HEADER} must be a positive integer"

                },

            )

        stats["sla_requests"] += 1



    admission_details = {}



    # Atomically estimate SLA completion and reserve capacity.

    async with admission_lock:

        total_capacity = MAX_ACTIVE + MAX_QUEUE



        if sla_budget_ms is not None:

            elapsed_seconds = time.perf_counter() - t0

            remaining_budget_seconds = (

                sla_budget_ms / 1000.0 - elapsed_seconds

            )



            active_at_admission = active

            waiting_at_admission = waiting



            (

                predicted_queue_wait,

                estimated_service_time,

                predicted_seconds,

            ) = estimate_completion_time(

                active_requests=active_at_admission,

                waiting_requests=waiting_at_admission,

            )



            admission_details = {

                "sla_budget_ms": sla_budget_ms,

                "remaining_budget_ms": round(

                    max(0.0, remaining_budget_seconds) * 1000, 2

                ),

                "predicted_completion_ms": round(

                    predicted_seconds * 1000, 2

                ),

                "estimated_service_time_ms": round(

                    estimated_service_time * 1000, 2

                ),

                "predicted_queue_wait_ms": round(

                    predicted_queue_wait * 1000, 2

                ),

                "active_at_admission": active_at_admission,

                "waiting_at_admission": waiting_at_admission,

            }



            if predicted_seconds > remaining_budget_seconds:

                stats["rejected_sla"] += 1

                return reject(

                    "predicted_sla_miss",

                    t0,

                    extra=admission_details,

                )



        # Preserve original bounded-queue capacity check.

        if active + waiting >= total_capacity:

            stats["rejected_queue_full"] += 1

            return reject(

                "queue_full",

                t0,

                extra=admission_details,

            )



        waiting += 1



    # Wait for an active inference slot.

    acquired = False

    waiting_reserved = True



    try:

        try:

            async with asyncio.timeout(QUEUE_DEADLINE):

                await slots.acquire()

            acquired = True



        except TimeoutError:

            stats["rejected_deadline"] += 1

            return reject(

                "queue_deadline",

                t0,

                extra=admission_details,

            )



        finally:

            # Remove waiting reservation exactly once.

            if waiting_reserved:

                async with admission_lock:

                    waiting -= 1

                waiting_reserved = False



        # A semaphore slot has been acquired.

        async with admission_lock:

            active += 1



        queue_wait = time.perf_counter() - t0

        t1 = time.perf_counter()



        try:

            upstream = await request.app.state.client.post(

                "/v1/chat/completions",

                json=body,

            )



            service_time = time.perf_counter() - t1

            total_latency = time.perf_counter() - t0



            # Store observed upstream response time for future estimates.

            service_time_history.append(service_time)

            stats["completed"] += 1



            if sla_budget_ms is not None:

                if total_latency * 1000 <= sla_budget_ms:

                    stats["sla_completed_within_budget"] += 1

                else:

                    stats["sla_completed_over_budget"] += 1



            log_event(

                outcome="ok",

                status=upstream.status_code,

                queue_wait=queue_wait,

                service_time=service_time,

                total_latency=total_latency,

                sla_budget_ms=admission_details.get("sla_budget_ms"),

                remaining_budget_ms=admission_details.get(

                    "remaining_budget_ms"

                ),

                predicted_completion_ms=admission_details.get(

                    "predicted_completion_ms"

                ),

                estimated_service_time_ms=admission_details.get(

                    "estimated_service_time_ms"

                ),

                predicted_queue_wait_ms=admission_details.get(

                    "predicted_queue_wait_ms"

                ),

                active_at_admission=admission_details.get(

                    "active_at_admission"

                ),

                waiting_at_admission=admission_details.get(

                    "waiting_at_admission"

                ),

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

            total_latency = time.perf_counter() - t0

            stats["upstream_timeout"] += 1



            log_event(

                outcome="upstream_timeout",

                status=504,

                queue_wait=queue_wait,

                service_time=service_time,

                total_latency=total_latency,

                sla_budget_ms=admission_details.get("sla_budget_ms"),

                remaining_budget_ms=admission_details.get(

                    "remaining_budget_ms"

                ),

                predicted_completion_ms=admission_details.get(

                    "predicted_completion_ms"

                ),

                estimated_service_time_ms=admission_details.get(

                    "estimated_service_time_ms"

                ),

                predicted_queue_wait_ms=admission_details.get(

                    "predicted_queue_wait_ms"

                ),

                active_at_admission=admission_details.get(

                    "active_at_admission"

                ),

                waiting_at_admission=admission_details.get(

                    "waiting_at_admission"

                ),

            )



            return JSONResponse(

                status_code=504,

                content={

                    "error": "upstream inference request timed out"

                },

            )



        except httpx.TransportError:

            service_time = time.perf_counter() - t1

            total_latency = time.perf_counter() - t0

            stats["upstream_error"] += 1



            log_event(

                outcome="upstream_error",

                status=502,

                queue_wait=queue_wait,

                service_time=service_time,

                total_latency=total_latency,

                sla_budget_ms=admission_details.get("sla_budget_ms"),

                remaining_budget_ms=admission_details.get(

                    "remaining_budget_ms"

                ),

                predicted_completion_ms=admission_details.get(

                    "predicted_completion_ms"

                ),

                estimated_service_time_ms=admission_details.get(

                    "estimated_service_time_ms"

                ),

                predicted_queue_wait_ms=admission_details.get(

                    "predicted_queue_wait_ms"

                ),

                active_at_admission=admission_details.get(

                    "active_at_admission"

                ),

                waiting_at_admission=admission_details.get(

                    "waiting_at_admission"

                ),

            )



            return JSONResponse(

                status_code=502,

                content={

                    "error": "could not connect to inference server"

                },

            )



        finally:

            async with admission_lock:

                active -= 1



    finally:

        if acquired:

            slots.release()



        # Safety cleanup if cancellation occurs before normal cleanup.

        if waiting_reserved:

            async with admission_lock:

                waiting -= 1





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

            return {"gateway": "up", "vllm": "up"}



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

            content={"gateway": "up", "vllm": "down"},

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

        "service_time_samples": len(service_time_history),

        "estimated_service_time_seconds": round(

            estimate_service_time(), 4

        ),

    }
