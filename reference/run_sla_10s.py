
import asyncio

import json

import statistics

import time

from collections import Counter

from datetime import datetime, timezone

from pathlib import Path



import httpx



URL = "http://localhost:9000/v1/chat/completions"

METRICS_URL = "http://127.0.0.1:9000/metrics"



BUDGETS_MS = [10000]

TRIALS_PER_BUDGET = 3

REQUESTS_PER_TRIAL = 10

OUTPUT_DIR = Path("results")



PAYLOAD = {

    "model": "Qwen/Qwen2.5-0.5B-Instruct",

    "messages": [

        {"role": "user", "content": "Reply with one short sentence."}

    ],

    "max_tokens": 64,

}





async def send_one(client, sla_ms):

    start = time.perf_counter()

    headers = {"X-Request-Deadline-Ms": str(sla_ms)}



    try:

        response = await client.post(URL, json=PAYLOAD, headers=headers)

        latency = time.perf_counter() - start



        try:

            body = response.json()

        except ValueError:

            body = {}



        return {

            "status": response.status_code,

            "latency_seconds": round(latency, 4),

            "reason": body.get("reason"),

        }



    except httpx.HTTPError as exc:

        return {

            "status": "client_error",

            "latency_seconds": round(time.perf_counter() - start, 4),

            "reason": str(exc),

        }





async def get_metrics(client):

    try:

        response = await client.get(METRICS_URL)

        response.raise_for_status()

        return response.json()

    except (httpx.HTTPError, ValueError) as exc:

        print(f"Warning: could not read gateway metrics: {exc}")

        return None





def summarize(results):

    status_counts = Counter(str(r["status"]) for r in results)

    successful = sorted(

        r["latency_seconds"]

        for r in results

        if r["status"] == 200

    )



    rejection_reasons = Counter(

        r["reason"] or "unspecified"

        for r in results

        if r["status"] == 429

    )



    summary = {

        "status_counts": dict(status_counts),

        "success_rate": round(len(successful) / len(results), 3),

        "rejection_reasons": dict(rejection_reasons),

        "successful_latency_median_seconds": None,

        "successful_latency_p95_seconds": None,

        "successful_latency_max_seconds": None,

    }



    if successful:

        p95_index = min(len(successful) - 1, int(0.95 * len(successful)))

        summary.update({

            "successful_latency_median_seconds": round(

                statistics.median(successful), 4

            ),

            "successful_latency_p95_seconds": round(

                successful[p95_index], 4

            ),

            "successful_latency_max_seconds": round(successful[-1], 4),

        })



    return summary





async def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_file = OUTPUT_DIR / f"sla_experiments_{timestamp}.json"



    records = []



    async with httpx.AsyncClient(timeout=60) as client:

        health = await client.get("http://127.0.0.1:9000/health")

        health.raise_for_status()

        print("Gateway health check passed.")



        for budget_ms in BUDGETS_MS:

            for trial in range(1, TRIALS_PER_BUDGET + 1):

                metrics_before = await get_metrics(client)



                print(

                    f"\nBudget={budget_ms} ms | "

                    f"Trial {trial}/{TRIALS_PER_BUDGET} | "

                    f"{REQUESTS_PER_TRIAL} simultaneous requests"

                )



                results = await asyncio.gather(

                    *[

                        send_one(client, budget_ms)

                        for _ in range(REQUESTS_PER_TRIAL)

                    ]

                )



                metrics_after = await get_metrics(client)

                summary = summarize(results)



                record = {

                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),

                    "sla_budget_ms": budget_ms,

                    "trial": trial,

                    "requests": REQUESTS_PER_TRIAL,

                    "summary": summary,

                    "responses": results,

                    "gateway_metrics_before": metrics_before,

                    "gateway_metrics_after": metrics_after,

                }

                records.append(record)



                print(json.dumps(summary, indent=2))



                # Save after every trial, so partial progress is preserved.

                output_file.write_text(json.dumps(records, indent=2))



                # Give the gateway a short rest between trials.

                await asyncio.sleep(2)



    print(f"\nAll trials finished. Results saved to: {output_file}")





if __name__ == "__main__":

    asyncio.run(main())

