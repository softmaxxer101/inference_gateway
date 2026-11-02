

"""

Fire N simultaneous requests with an SLA latency budget.



Usage:

    python load_test_sla.py

    python load_test_sla.py 20

    SLA_BUDGET_MS=1000 python load_test_sla.py 10

"""



import asyncio

import json

import math

import os

import statistics

import sys

import time

from collections import Counter



import httpx





URL = "http://localhost:9000/v1/chat/completions"

SLA_BUDGET_MS = int(os.getenv("SLA_BUDGET_MS", "3000"))



PAYLOAD = {

    "model": "Qwen/Qwen2.5-0.5B-Instruct",

    "messages": [

        {"role": "user", "content": "Reply with one short sentence."}

    ],

    "max_tokens": 64,

}





async def one(client: httpx.AsyncClient):

    start = time.perf_counter()



    try:

        response = await client.post(

            URL,

            json=PAYLOAD,

            headers={"X-Request-Deadline-Ms": str(SLA_BUDGET_MS)},

        )

        latency = time.perf_counter() - start



        try:

            body = response.json()

        except ValueError:

            body = {"raw_body": response.text}



        reason = ""



        if isinstance(body, dict):

            detail = body.get("detail", "")

            reason = body.get("reason", "")



            if isinstance(detail, dict):

                reason = detail.get("reason", reason)

            elif isinstance(detail, str):

                reason = detail



        return response.status_code, latency, reason, body



    except httpx.HTTPError as exc:

        latency = time.perf_counter() - start

        return "client_error", latency, str(exc), {}





def percentile_nearest_rank(values, percentile):

    """Calculate a percentile using the nearest-rank method."""

    ordered = sorted(values)



    if not ordered:

        return None



    rank = max(1, math.ceil(percentile * len(ordered)))

    return ordered[rank - 1]





async def main(n: int):

    async with httpx.AsyncClient(timeout=60) as client:

        results = await asyncio.gather(

            *[one(client) for _ in range(n)]

        )



    counts = Counter(code for code, _, _, _ in results)



    print(f"SLA budget: {SLA_BUDGET_MS} ms")

    print(f"Sent {n} simultaneous requests -> status counts: {dict(counts)}")



    successful = [

        latency

        for code, latency, _, _ in results

        if code == 200

    ]



    if successful:

        p95 = percentile_nearest_rank(successful, 0.95)



        print(

            f"200s: median {statistics.median(successful):.2f}s, "

            f"p95 {p95:.2f}s, "

            f"max {max(successful):.2f}s"

        )



        within_budget = sum(

            latency * 1000 <= SLA_BUDGET_MS

            for latency in successful

        )



        print(

            f"Successful responses within client-observed budget: "

            f"{within_budget}/{len(successful)}"

        )



    rejected = [

        (latency, reason, body)

        for code, latency, reason, body in results

        if code == 429

    ]



    if rejected:

        rejection_counts = Counter(

            reason or "reason_not_found_in_response"

            for _, reason, _ in rejected

        )



        print(f"429 rejection reasons: {dict(rejection_counts)}")

        print(

            f"Slowest 429 rejection: "

            f"{max(latency for latency, _, _ in rejected):.2f}s"

        )



        for _, reason, body in rejected:

            if not reason:

                print(

                    "Example rejection response body:",

                    json.dumps(body, ensure_ascii=False),

                )

                break



    client_errors = [

        (latency, reason)

        for code, latency, reason, _ in results

        if code == "client_error"

    ]



    if client_errors:

        print(f"Client/network errors: {len(client_errors)}")





if __name__ == "__main__":

    request_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10



    if request_count < 1:

        raise SystemExit("Request count must be at least 1.")



    if SLA_BUDGET_MS < 1:

        raise SystemExit("SLA_BUDGET_MS must be positive.")



    asyncio.run(main(request_count))


