# inference-serving-infra

curl → Go server → Python worker → response

# Inference Serving Infrastructure

A production-grade LLM inference serving system built from scratch in Go, exploring the core systems challenges behind serving large language models at scale: request scheduling, dynamic batching, concurrency control, memory management, and horizontal scaling.

The Go server acts as the control plane — handling HTTP routing, request queuing, batching, and scheduling — while a Python gRPC worker handles model inference via `llama-cpp-python`. This separation mirrors how production systems like vLLM, Triton, and TensorRT-LLM are architected.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Go HTTP Server                  │
                    │                                             │
  HTTP Request ───► │  Validate ─► Queue ─► Batch ─► Dispatch    │
                    │                                     │       │
                    │                                     ▼       │
                    │                              gRPC Client    │
                    └─────────────────────────────────┬───────────┘
                                                      │
                                                      │ gRPC (TCP)
                                                      │
                    ┌─────────────────────────────────▼───────────┐
                    │            Python Worker                     │
                    │                                             │
                    │   gRPC Server ─► llama-cpp-python ─► Model  │
                    └─────────────────────────────────────────────┘
```

## Project Structure

```
inference-serving-infra/
├── cmd/
│   └── server/
│       └── main.go                 # Server entry point
├── internal/
│   ├── api/
│   │   ├── handler.go              # HTTP handlers
│   │   └── types.go                # Request/response types
│   ├── worker/
│   │   └── client.go               # gRPC client wrapper
│   ├── queue/
│   │   └── queue.go                # Priority request queue
│   ├── batcher/
│   │   └── batcher.go              # Dynamic batch accumulator
│   ├── scheduler/
│   │   └── scheduler.go            # Continuous batching scheduler
│   ├── registry/
│   │   └── registry.go             # Multi-model management
│   ├── balancer/
│   │   └── balancer.go             # Load balancer & routing
│   └── metrics/
│       └── metrics.go              # Prometheus metrics
├── proto/
│   └── inference.proto             # gRPC service definition
├── gen/                            # Generated protobuf code
├── worker/
│   └── worker.py                   # Python inference worker
├── bench/
│   └── loadtest.go                 # Load testing tool
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

## Getting Started

### Prerequisites

- Go 1.21+
- Python 3.10+
- protoc with protoc-gen-go and protoc-gen-go-grpc
- A GGUF model file (e.g., TinyLlama 1.1B Q4_K_M)

### Setup

```bash
# Clone the repo
git clone https://github.com/yashp5/inference-serving-infra.git
cd inference-serving-infra

# Generate protobuf code
make proto

# Install Python dependencies
pip install grpcio grpcio-tools llama-cpp-python

# Download a model (example)
mkdir -p models/
# Place your .gguf file in models/

# Start the Python worker
python worker/worker.py --model-path models/tinyllama-1.1b-q4_k_m.gguf --port 50051

# Start the Go server (in a separate terminal)
go run cmd/server/main.go --worker-addr localhost:50051

# Test
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50, "temperature": 0.7}'
```

## Build Phases

The project is built incrementally. Each phase adds a layer of complexity that addresses a real production concern.

### Phase 1 — Single Model, Single Request Server ✅

The foundation: a Go HTTP server that proxies inference requests to a Python gRPC worker.

**What's built:**
- gRPC service definition (`proto/inference.proto`) with `GenerateRequest`/`GenerateResponse`
- Python worker that loads a GGUF model via `llama-cpp-python` and serves inference over gRPC
- Go HTTP server with `POST /v1/completions` endpoint
- Input validation (non-empty prompt, max_tokens bounds, temperature range)
- Per-request latency tracking: total time, gRPC call time, overhead
- Request ID generation (UUID) in the Go server

**Key design decisions:**
- Go handles the control plane (HTTP, routing, scheduling); Python handles the data plane (GPU/CPU inference). This separation is how production serving systems work.
- gRPC over TCP between Go and Python for clean serialization and future streaming support.
- Single gRPC connection created at server startup, not per request.

**Concepts:** HTTP server design, gRPC client/server, protobuf serialization, latency measurement

---

### Phase 2 — Request Queue and Concurrency Control

The model can only handle a limited number of concurrent requests. This phase adds admission control, queuing, and fairness.

**What to build:**
- In-memory priority queue using `container/heap` to buffer incoming requests
- Concurrency limiter using a semaphore pattern (buffered channel of size N) to cap in-flight requests to the worker
- Dispatcher goroutine that pulls from the queue and sends to available worker slots
- HTTP 429 (Too Many Requests) with `Retry-After` header when the queue is full
- Per-user rate limiting with token buckets (`sync.Map` of token bucket structs)
- Priority levels (high/medium/low) with starvation prevention
- `context.Context` propagation throughout — if a client disconnects, cancel in-flight work
- Configurable queue timeout: requests waiting longer than a deadline return HTTP 504
- Separate tracking of queue wait time vs. inference time in metrics

**Benchmarking:**
- Hit the server with increasing concurrency (1, 10, 50, 100, 500 concurrent requests)
- Measure p50, p95, p99 latency, throughput (req/s), queue depth over time, rejection rate
- Plot how these degrade as load increases

**Concepts:** Priority queues, semaphores, goroutine coordination, rate limiting, backpressure, context cancellation

---

### Phase 3 — Dynamic Batching

The single most impactful optimization in inference serving. Instead of processing one request at a time, accumulate multiple requests and process them as a batch.

**What to build:**
- Batch accumulator goroutine that collects requests using two triggers:
  - Batch reaches max size (e.g., 8 requests), OR
  - Max wait time expires (e.g., 50ms since the first request entered the batch)
- `select` statement with `time.After` for the timeout trigger
- Fan-out logic: each request in the batch has a response channel; when batch results come back, route each result to the correct waiting HTTP handler
- Pre-allocated send/receive buffers to minimize GC pressure during batching
- Configurable batch size and max wait time as command-line flags

**Benchmarking:**
- Compare throughput and latency against Phase 2 (no batching)
- Under high load: batching should dramatically increase throughput
- Under low load: should add minimal latency (just the max wait time at worst)
- Show both regimes in plots, document the tradeoff

**Key tradeoff:** Larger batches improve throughput but increase latency for early-arriving requests. Smaller max wait times reduce latency but send partially-full batches.

**Concepts:** Dynamic batching, fan-out/fan-in, channel-based coordination, throughput vs. latency tradeoffs

---

### Phase 4 — Continuous Batching

What makes systems like vLLM special. Standard batching waits for all requests in a batch to finish before starting the next batch. Continuous batching lets new requests join at every decode step.

**What to build:**
- Streaming gRPC connection (or WebSocket) between Go scheduler and Python worker for per-token communication
- Slot-based scheduler: maintain an array of active request slots in the current batch
- At every decode step, the worker reports which slots finished (hit EOS or max tokens)
- Scheduler immediately swaps in waiting requests to fill freed slots
- The batch stays as full as possible at all times

**Why it matters:** In LLM inference, different requests generate different numbers of tokens. With static batching, a request wanting 10 tokens holds its slot until the request wanting 500 tokens finishes. Continuous batching recovers those wasted cycles.

**Benchmarking:**
- Compare GPU utilization and throughput against static batching (Phase 3)
- Show improvement when generation lengths vary widely across requests

**Concepts:** Iteration-level scheduling, streaming RPC, slot management, GPU utilization optimization

---

### Phase 5 — Multi-Model Management and Memory

A real serving system runs multiple models and manages limited memory.

**What to build:**
- Model registry: knows available models (name, size, path on disk)
- Endpoints to load and unload models (`POST /v1/models/load`, `DELETE /v1/models/{name}`)
- Memory tracking: refuse to load a model if it would exceed the memory limit
- LRU eviction: when a request arrives for an unloaded model and there's no memory, evict the least-recently-used model
- Cold-start latency tracking (model loading time) as a separate metric
- Warm-up and preloading: on startup, preload the most commonly requested models
- Model isolation: each model gets its own worker process; Go server routes requests based on the `model` field in the request
- `map[string]*Worker` with proper mutex protection

**Concepts:** Resource management, LRU caches, process lifecycle management, mutex patterns

---

### Phase 6 — Observability and Metrics

No production system exists without observability.

**What to build:**
- `/metrics` endpoint in Prometheus format using `prometheus/client_golang`:
  - Request count (total, by model, by status code)
  - Latency histograms (queue time, inference time, total time)
  - Batch sizes (histogram)
  - Queue depth (gauge)
  - Active in-flight requests (gauge)
  - Model load/unload events (counter)
  - Error rates (counter by error type)
- Structured logging with `slog` (Go's standard structured logger):
  - Every request logged with: request ID, timestamps at each stage (received, queued, dispatched, completed), model name, token counts, latency breakdown
  - JSON-formatted output
- Health check endpoints:
  - `GET /healthz` — is the server up?
  - `GET /readyz` — is at least one model loaded and ready?
  - `GET /livez` — are workers responsive?
- Dashboard: either a Grafana JSON config or a Go-served HTML page with Chart.js polling a `/stats` endpoint

**Concepts:** Prometheus metrics, structured logging, health check patterns (Kubernetes-style), observability best practices

---

### Phase 7 — Horizontal Scaling and Load Balancing

This is where it becomes a distributed systems project.

**What to build:**
- Multiple server instances: run 2-4 instances, each managing its own workers
- Go load balancer that sits in front and routes requests with smart routing:
  - Model affinity: prefer sending a request to a server that already has the model warm
  - Least-loaded: fall back to the server with the shortest queue if the preferred one is overloaded
  - Consistent hashing: map model names to server instances; when a server goes down, only its models need redistribution
- Service discovery: each server instance registers with the load balancer on startup and sends periodic heartbeats; missed heartbeats → stop routing to that instance
- Fault tolerance: demonstrate that removing a server only disrupts a fraction of requests

**Benchmarking:**
- Compare throughput of 1 vs 2 vs 4 server instances
- Measure cache hit rate (requests served by a server with the model already loaded)
- Show fault tolerance: kill a server, measure impact on in-flight requests

**Concepts:** Load balancing strategies, consistent hashing, service discovery, heartbeats, fault tolerance, distributed systems

---

### Phase 8 — KV Cache Management (Bonus)

Advanced but extremely relevant to current LLM infrastructure.

**What to build:**
- Cache layer in Go that maps prompt prefixes (hashed) to cached KV state
- On new request, check if any prefix of the prompt has a cached KV state
- If cache hit, tell the worker to resume from cached state instead of recomputing the full prefillA
- LRU eviction for the cache (KV states are large)
- Cache hit/miss rate metrics

**Why it matters:** If a user sends a follow-up message with the same system prompt prefix, reusing the KV cache avoids recomputing the entire prefill — a major latency win in production chat systems.

**Concepts:** Prefix caching, hash-based lookup, memory management, prefill optimization

---

## API

### POST /v1/completions

```json
// Request
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}

// Response
{
  "request_id": "req_abc123",
  "generated_text": "in a land far away...",
  "tokens_generated": 42,
  "inference_time_ms": 850,
  "queue_time_ms": 12,
  "total_time_ms": 870
}
```

### GET /healthz
Returns 200 if the server is running.

### GET /readyz
Returns 200 if at least one model is loaded and ready to serve.

### GET /metrics
Prometheus-formatted metrics.

## Benchmarking

The `bench/` directory contains a load testing tool that sweeps across concurrency levels and measures latency percentiles, throughput, and error rates.

```bash
# Run load test with 50 concurrent requests, 1000 total
go run bench/loadtest.go --concurrency 50 --total 1000 --url http://localhost:8080/v1/completions
```

Results are output as CSV for plotting.

## Reading List

Papers and resources that inform the design of this system:

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023) — The foundational vLLM paper. Explains why KV cache memory management matters and how virtual memory concepts apply to inference serving.
- **How Continuous Batching Enables 23x Throughput in LLM Inference** (Anyscale blog) — Explains static vs. dynamic vs. continuous batching with benchmarks. Motivation for Phases 3 and 4.
- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (vLLM blog, 2025) — Walks through vLLM's scheduler, engine core, and prefill/decode handling. Reference architecture for Phase 4.
- **Go Concurrency Patterns** and **Advanced Go Concurrency Patterns** (Go blog) — The select-with-timeout, fan-out/fan-in, and pipeline patterns used throughout the batching and scheduling logic.
- **Designing Data-Intensive Applications** (Kleppmann) — Chapters on replication, partitioning, and distributed systems apply directly to Phase 7.

## License

MIT
