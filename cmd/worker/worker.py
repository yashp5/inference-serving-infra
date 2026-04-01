"""
Keep it minimal — a single Python file that does three things:
    loads the model on startup, implements the gRPC Generate method, and serves on a Unix domain socket (or a TCP port, your call).
For the model, use llama-cpp-python with a small GGUF model if you don't have a GPU.
TinyLlama 1.1B Q4 quantized is around 600MB and runs fine on CPU.
If you want even lighter for iteration speed, GPT-2 via transformers works too but is less representative of real LLM inference.
"""

from concurrent import futures
import logging
import os
import signal
import time

from grpc_reflection.v1alpha import reflection
import inference_pb2
import inference_pb2_grpc
from llama_cpp import Llama

import grpc

# ------ Logging ---------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("worker")

# --- Config (overrride via env vars) ------------------------

PORT = os.getenv("WORKER_PORT", "50051")
MAX_WORKERS = int(os.getenv("WORKER_MAX_THREADS", "4"))


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self):
        super().__init__()
        self.llm = Llama(
            model_path="/Users/yash/ml-infra/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=512,
        )

    def Generate(self, request, context):
        log.info(
            "Generate called request_id=%s prompt_len=%d max_tokens=%d temperature=%.2f",
            request.request_id,
            len(request.prompt),
            request.max_tokens,
            request.temperature,
        )

        # validate
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")

        if not request.prompt:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt is required")

        if request.max_tokens <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "max_tokens must be > 0")

        if not (0.0 <= request.temperature <= 2.0):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "temperature must be between 0.0 and 2.0",
            )

        # inference
        start_ms = time.monotonic()

        generated_text, tokens_generated = self._run_inference(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        inference_time_ms = int((time.monotonic() - start_ms) * 1000)

        # build response
        log.info(
            "Generate done request_id=%s tokens=%d latency_ms=%d",
            request.request_id,
            tokens_generated,
            inference_time_ms,
        )

        return inference_pb2.GenerateResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            inference_time_ms=inference_time_ms,
        )

    def _run_inference(self, prompt: str, max_tokens: int, temperature: float):
        output = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
        generated_text = output["choices"][0]["text"]
        tokens_generated = output["usage"]["completion_tokens"]
        return (generated_text, tokens_generated)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))

    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)

    SERVICE_NAMES = (
        inference_pb2.DESCRIPTOR.services_by_name["Inference"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    log.info("worker listening on port %s (thread=%d)", PORT, MAX_WORKERS)

    # --- Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        log.info("Shutdown signal received, stopping server...")
        server.stop(grace=5).wait()
        log.info("Server stoppe.")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
