
"""

Send N simultaneous requests to the LLM inference gateway.



Usage:

    python load_test.py             # 10 requests, no SLA budget

    python load_test.py 20          # 20 requests, no SLA budget

    python load_test.py 10 2000     # 10 requests, 2000 ms SLA budget

"""



import asyncio

import statistics

import sys

import time

from collections import Counter



import httpx



URL = "http://localhost:9000/v1/chat/completions"



PAYLOAD = {

    "model": "Qwen/Qwen2.5-0.5B-Instruct",

    "messages": [

        {"role": "user", "content": "Reply with one short sentence."}

    ],

    "max_tokens": 64,

}





async def one(client: httpx.AsyncClient, sla_ms=None):

    start = time.perf_counter()

    headers = {}



    if sla_ms is not None:

        headers["X-Request-Deadline-Ms"] = str(sla_ms)



    try:

        response = await client.post(URL, json=PAYLOAD, headers=headers)

        latency = time.perf_counter() - start



        try:

            body = response.json()

        except ValueError:

            body = {}



        return response.status_code, latency, body



    except httpx.HTTPError as exc:

        return "client_error", time.perf_counter() - start, {"error": str(exc)}





async def main(n: int, sla_ms=None):

    async with httpx.AsyncClient(timeout=60) as client:

        results = await asyncio.gather(

            *[one(client, sla_ms) for _ in range(n)]

        )



    codes = Counter(code for code, _, _ in results)



    print(f"\nSent {n} simultaneous requests")

    print(

        f"SLA budget: {sla_ms} ms"

        if sla_ms is not None

        else "SLA budget: disabled"

    )

    print(f"Status counts: {dict(codes)}")



    successful = sorted(

        latency for code, latency, _ in results if code == 200

    )



    if successful:

        p95_index = min(len(successful) - 1, int(0.95 * len(successful)))

        print(

            f"200s: median {statistics.median(successful):.2f}s, "

            f"p95 {successful[p95_index]:.2f}s, "

            f"max {successful[-1]:.2f}s"

        )



    rejected = [

        (latency, body)

        for code, latency, body in results

        if code == 429

    ]



    if rejected:

        print(

            f"429s: {len(rejected)}; "

            f"slowest rejection {max(t for t, _ in rejected):.2f}s"

        )



        reasons = Counter(

            body.get("reason", body.get("error", "reason_not_in_body"))

            for _, body in rejected

        )

        print(f"429 reasons: {dict(reasons)}")



    errors = [

        body for code, _, body in results if code == "client_error"

    ]



    if errors:

        print(f"Client errors: {len(errors)}")

        for error in errors[:3]:

            print(f"  {error.get('error', 'unknown error')}")





if __name__ == "__main__":

    if len(sys.argv) > 3:

        raise SystemExit(

            "Usage: python load_test.py [request_count] [sla_budget_ms]"

        )



    request_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    sla_budget = int(sys.argv[2]) if len(sys.argv) > 2 else None



    if request_count < 1:

        raise SystemExit("request_count must be at least 1")



    if sla_budget is not None and sla_budget < 1:

        raise SystemExit("sla_budget_ms must be a positive integer")



    asyncio.run(main(request_count, sla_budget))

