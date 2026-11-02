
import json

import math

import statistics

from pathlib import Path



LOG_PATH = Path("results/requests.jsonl")

HISTORY_SIZE = 20

MIN_HISTORY = 5





def nearest_rank_percentile(values, percentile):

    ordered = sorted(values)

    index = max(0, math.ceil(percentile * len(ordered)) - 1)

    return ordered[index]





def load_service_times(path):

    observations = []



    with path.open(encoding="utf-8") as f:

        for line in f:

            try:

                row = json.loads(line)

            except json.JSONDecodeError:

                continue



            if row.get("status") != 200:

                continue



            service_time = row.get("service_time")



            if service_time is None:

                continue



            try:

                service_time = float(service_time)

            except (ValueError, TypeError):

                continue



            if service_time > 0:

                observations.append(service_time)



    return observations





def evaluate(name, observations, estimator):

    errors = []



    for i in range(MIN_HISTORY, len(observations)):

        history = observations[max(0, i - HISTORY_SIZE):i]

        actual = observations[i]

        predicted = estimator(history)

        errors.append(actual - predicted)



    absolute_errors = [abs(error) for error in errors]



    underpredictions = sum(error > 0 for error in errors)

    overpredictions = sum(error < 0 for error in errors)

    exact = sum(error == 0 for error in errors)



    print(f"\n{name}")

    print(f"  Predictions evaluated: {len(errors)}")

    print(f"  Mean absolute error:   {statistics.mean(absolute_errors):.3f} s")

    print(f"  Mean signed error:     {statistics.mean(errors):+.3f} s")

    print(f"  Underpredictions:      {underpredictions}")

    print(f"  Overpredictions:       {overpredictions}")

    print(f"  Exact predictions:     {exact}")





def main():

    if not LOG_PATH.exists():

        raise SystemExit(f"Log file not found: {LOG_PATH}")



    observations = load_service_times(LOG_PATH)



    if len(observations) <= MIN_HISTORY:

        raise SystemExit("Not enough service-time observations.")



    print(f"Service-time observations: {len(observations)}")

    print(f"Rolling history size:      {HISTORY_SIZE}")

    print(f"Minimum history required:  {MIN_HISTORY}")



    evaluate("Mean estimator", observations, statistics.mean)

    evaluate(

        "P90 estimator",

        observations,

        lambda history: nearest_rank_percentile(history, 0.90),

    )

    evaluate(

        "P95 estimator",

        observations,

        lambda history: nearest_rank_percentile(history, 0.95),

    )





if __name__ == "__main__":

    main()

