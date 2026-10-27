# Building an LLM Inference Gateway: Queues, Limits & Timeouts



A lightweight inference gateway built with **FastAPI and vLLM** to explore production-oriented LLM serving concepts, including concurrency control, bounded queues, overload handling, upstream timeouts, request logging, and metrics.



This project focuses on understanding how a gateway manages incoming requests when inference capacity is limited, and how its behavior can be measured through controlled load tests.



## 🎯 Project Goals



- Control the number of concurrent inference requests.

- Implement bounded queue admission and reject excess requests.

- Limit how long requests can wait for an available processing slot.

- Handle upstream inference-server timeouts.

- Record request outcomes, queue wait times, and service times.

- Expose metrics to observe gateway behavior under load.

- Experiment with trade-offs between request acceptance, rejection, and latency.



## 🏗️ Architecture



```text

                   ┌─────────────────────┐

                   │   Load Test Client  │

                   │    load_test.py     │

                   └──────────┬──────────┘

                              │

                              ▼

                   ┌─────────────────────┐

                   │   FastAPI Gateway   │

                   │                     │

                   │  Admission Control  │

                   │  Concurrency Limit  │

                   │  Bounded Queue      │

                   │  Timeout Handling   │

                   │  Request Logging    │

                   └──────────┬──────────┘

                              │

                              ▼

                   ┌─────────────────────┐

                   │     vLLM Server     │

                   │   LLM Inference     │

                   └─────────────────────┘

```



The gateway sits between the load-test client and vLLM. It regulates access to the inference server and records the outcome of each request.



## 🛠️ Tech Stack



- **Python** — implementation and experiments

- **FastAPI** — gateway API

- **Uvicorn** — ASGI server

- **httpx** — asynchronous HTTP requests

- **vLLM** — LLM inference serving

- **JSONL** — structured request logs

- **Linux on WSL Ubuntu** — local development environment



## 📂 Project Structure



```text

production_vllm/

├── gateway.py       # FastAPI inference gateway

├── load_test.py     # Concurrent request load tester

├── results/         # Experiment logs in JSONL format

└── README.md        # Project documentation

```



## ⚙️ Gateway Features



### 1. Concurrency Control



Uses an asynchronous semaphore to limit the number of requests actively processed by the gateway.



```bash

MAX_ACTIVE=1

```



This setting allows one request at a time to occupy an active gateway slot.



### 2. Bounded Queue



Limits how many additional requests can wait for an active slot.



```bash

MAX_QUEUE=4

```



Requests arriving when the active capacity and waiting queue are full are rejected rather than admitted without limit.



### 3. Queue Deadline



Limits how long a request can wait to acquire an active slot.



```bash

QUEUE_DEADLINE=5

```



Requests that exceed this waiting deadline receive an HTTP 429 response.



### 4. Upstream Timeout



Limits how long the gateway waits for a response from vLLM.



```bash

UPSTREAM_TIMEOUT=30

```



An upstream timeout is handled as an HTTP 504 response.



### 5. Metrics and Logging



The gateway exposes a `/metrics` endpoint with counters and current-state values, including:



- Total requests received

- Completed upstream responses

- Queue-full rejections

- Queue-deadline rejections

- Upstream timeouts and errors

- Current active requests

- Current waiting requests

- Configured concurrency and queue limits



Request events are also written to JSONL logs for later analysis.



## 🖥️ Running Experiments with Four Terminals



I used **four terminals in my WSL Ubuntu environment** to separate the gateway, inference server, load generator, and monitoring tasks.



| Terminal | Responsibility | Example |

|---|---|---|

| Terminal 1 | Run the vLLM inference server | Start the model-serving process |

| Terminal 2 | Run the FastAPI gateway | `uvicorn gateway:app --port 9000` |

| Terminal 3 | Generate concurrent requests | `python load_test.py 20` |

| Terminal 4 | Inspect metrics and experiment logs | `curl -s http://localhost:9000/metrics` |



This separation makes it easier to observe how the gateway behaves while inference requests are being processed.



**Note:** The exact commands and environment variables depend on the model configuration and the experiment being run.



## 🚀 Running the Gateway



### 1. Activate the Python environment



```bash

source venv/bin/activate

```



### 2. Start the vLLM server



Start vLLM using your chosen model and its appropriate serving configuration. The gateway expects the OpenAI-compatible API to be available at:



```text

http://localhost:8000

```



If vLLM is running on a different address, configure `VLLM_URL` accordingly.



### 3. Start the gateway



```bash

uvicorn gateway:app --port 9000

```



### 4. Run a load test



In another terminal:



```bash

python load_test.py 20

```



This sends 20 simultaneous requests to the gateway.



### 5. Inspect metrics



```bash

curl -s http://localhost:9000/metrics

```



### 6. Inspect the logs



```bash

cat results/requests.jsonl

```



The log filename can be changed using the `LOG_PATH` environment variable.



## 🧪 Experiments Completed



### Experiment 1: Concurrency and Admission Control



Explored how the gateway limits active requests and handles requests that exceed its admission capacity.



### Experiment 2: Queue Capacity vs. Overload



Kept the active-request limit fixed while changing the queue capacity.



| Queue Capacity | Requests Sent | Successful | Rejected |

|---:|---:|---:|---:|

| 1 | 20 | 2 | 18 |

| 4 | 20 | 5 | 15 |

| 9 | 20 | 10 | 10 |



In these individual runs, larger queues allowed more requests to be accepted, but successful-request latency increased as the queue grew.



These results are observations from small experiments, not statistically conclusive performance benchmarks.



### Experiment 3: Queue Deadline



Tested a one-second queue deadline with one active slot and a queue capacity of nine.



In the latest run:

- 3 requests succeeded.

- 10 requests were rejected because the queue was full.

- 7 requests were rejected after exceeding the queue deadline.



This demonstrated the difference between rejecting a request immediately due to capacity and rejecting it after excessive queue waiting.



### Experiment 4: Upstream Timeout Handling



Configured an intentionally short upstream timeout:



```bash

UPSTREAM_TIMEOUT=0.001

```



Five simultaneous requests returned HTTP 504, and the gateway recorded five upstream timeouts.



This was a fault-injection test to validate timeout handling, not a realistic production configuration.



### Experiment 5: Recovery After Timeout Configuration Change



Restored the upstream timeout to 30 seconds and restarted the gateway.



The subsequent request returned HTTP 200 with an observed latency of approximately 0.91 seconds.



This confirmed that the gateway worked after restarting with the restored configuration. It did not independently establish recovery within the same running process.



## 📊 Key Learnings



- Concurrency limits control how many requests can actively occupy gateway slots.

- Bounded queues prevent unlimited admission of waiting requests.

- Larger queues can absorb larger bursts but may increase waiting time.

- Queue deadlines and upstream timeouts protect different stages of request processing.

- HTTP 429 and HTTP 504 represent different failure conditions in this gateway.

- Metrics and structured logs help explain why requests succeed or fail.

- Controlled experiments are useful for investigating production-serving trade-offs.



## 🔭 Future Work



The following features are planned but have not yet been implemented or validated:



- [ ] SLA-aware admission control based on request latency budgets

- [ ] Improved latency-percentile calculations and repeated benchmark runs

- [ ] Priority-based request scheduling

- [ ] More realistic upstream failure and recovery tests

- [ ] Circuit-breaker behavior for an unhealthy inference server

- [ ] Further load testing and configuration tuning

- [ ] End-to-end evaluation of gateway behavior under different workloads



## ⚠️ Current Limitations



- The current experiments use small request batches and are not production-scale benchmarks.

- The load-test percentile calculation needs improvement for statistically reliable tail-latency analysis.

- The current gateway uses in-process concurrency and queue state; multiple Uvicorn worker processes would not share these limits automatically.

- Streaming requests are not supported by the current implementation.

- SLA-aware admission control and priority scheduling are future work.



## 👤 About This Project



This is a hands-on learning project focused on the engineering problems surrounding LLM inference serving: request admission, bounded resources, overload behavior, timeouts, observability, and performance evaluation.



The goal is to develop a practical understanding of inference infrastructure through implementation, experimentation, and measurement.
