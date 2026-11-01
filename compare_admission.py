
import asyncio

import json

import statistics

import time

from collections import Counter

from datetime import datetime, timezone

from pathlib import Path



import httpx



URL = "http://127.0.0.1:9000/v1/chat/completions"

METRICS_URL = "http://127.0.0.1:9000/metrics"



TRIALS = 3

REQUESTS = 10

SLA_BUDGET_MS = 2000



OUTPUT_DIR = Path("results")



PAYLOAD = {

    "model": "Qwen/Qwen2.5-0.5B-Instruct",

    "messages": [

        {"role": "user", "content": "Reply with one short sentence."}

    ],

    "max_tokens": 64,

}





async def get_metrics(client):

    response = await client.get(METRICS_URL)

    response.raise_for_status()

    return response.json()





async def send_one(client, use_sla):

    start = time.perf_counter()



    headers = {}

    if use_sla:

        headers["X-Request-Deadline-Ms"] = str(SLA_BUDGET_MS)



    try:

        response = await client.post(

            URL,

            json=PAYLOAD,

            headers=headers,

        )



        try:

            body = response.json()

        except ValueError:

            body = {}



        return {

            "status": response.status_code,

            "latency_seconds": round(

                time.perf_counter() - start, 4

            ),

            "reason": body.get("reason"),

        }



    except httpx.HTTPError as exc:

        return {

            "status": "client_error",

            "latency_seconds": round(

                time.perf_counter() - start, 4

            ),

            "reason": str(exc),

        }





def summarize(results):

    statuses = Counter(str(r["status"]) for r in results)



    successful = sorted(

        r["latency_seconds"]

        for r in results

        if r["status"] == 200

    )



    rejected = [

        r for r in results if r["status"] == 429

    ]



    reasons = Counter(

        r["reason"] or "unspecified"

        for r in rejected

    )



    result = {

        "total_requests": len(results),

        "status_counts": dict(statuses),

        "successful_requests": len(successful),

        "rejected_429": len(rejected),

        "rejection_reasons": dict(reasons),

        "acceptance_rate": round(

            (len(results) - len(rejected)) / len(results), 3

        ),

        "success_rate": round(

            len(successful) / len(results), 3

        ),

        "median_latency_seconds": None,

        "p95_latency_seconds": None,

        "max_latency_seconds": None,

    }



    if successful:

        p95_index = min(

            len(successful) - 1,

            int(0.95 * len(successful)),

        )



        result.update({

            "median_latency_seconds": round(

                statistics.median(successful), 4

            ),

            "p95_latency_seconds": round(

                successful[p95_index], 4

            ),

            "max_latency_seconds": round(

                successful[-1], 4

            ),

        })



    return result





def metric_delta(before, after):

    if not before or not after:

        return None



    return {

        key: after[key] - before[key]

        for key in before

        if key in after

        and isinstance(before[key], (int, float))

        and isinstance(after[key], (int, float))

    }





async def run_policy(client, policy, use_sla):

    trials = []



    print(f"\n{'=' * 55}")

    print(f"POLICY: {policy}")

    print(f"{'=' * 55}")



    for trial in range(1, TRIALS + 1):

        before = await get_metrics(client)



        results = await asyncio.gather(

            *[

                send_one(client, use_sla)

                for _ in range(REQUESTS)

            ]

        )



        after = await get_metrics(client)



        record = {

            "trial": trial,

            "summary": summarize(results),

            "gateway_metric_delta": metric_delta(

                before, after

            ),

            "responses": results,

        }



        trials.append(record)



        print(f"\nTrial {trial}/{TRIALS}")

        print(json.dumps(record["summary"], indent=2))

        print("Gateway metric changes:")

        print(json.dumps(record["gateway_metric_delta"], indent=2))



        await asyncio.sleep(3)



    return trials





async def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



    timestamp = datetime.now(timezone.utc).strftime(

        "%Y%m%dT%H%M%SZ"

    )



    async with httpx.AsyncClient(timeout=60) as client:

        health = await client.get(

            "http://127.0.0.1:9000/health"

        )

        health.raise_for_status()



        print("Gateway health check passed.")



        # First: capacity-only admission.

        capacity_results = await run_policy(

            client,

            "capacity_based",

            use_sla=False,

        )



        # Second: SLA-aware admission.

        sla_results = await run_policy(

            client,

            "sla_aware",

            use_sla=True,

        )



    output = {

        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "configuration": {

            "max_active": 1,

            "max_queue": 9,

            "requests_per_trial": REQUESTS,

            "trials": TRIALS,

            "sla_budget_ms": SLA_BUDGET_MS,

        },

        "capacity_based": capacity_results,

        "sla_aware": sla_results,

    }



    output_file = OUTPUT_DIR / (

        f"admission_comparison_{timestamp}.json"

    )



    output_file.write_text(json.dumps(output, indent=2))



    print("\nComparison experiment completed.")

    print(f"Results saved to: {output_file}")





if __name__ == "__main__":

    asyncio.run(main())

