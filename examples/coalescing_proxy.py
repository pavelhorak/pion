import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import redis
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, Response

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://127.0.0.1:8080").rstrip("/")
UPSTREAM_CHAT_URL = f"{UPSTREAM_URL}/v1/chat/completions"

app = FastAPI()
redis_client = redis.Redis(host="127.0.0.1", port=1974, decode_responses=False)

_metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "coalesced_requests": 0,
    "new_inference_calls": 0,
}
_metrics_lock = threading.Lock()


def _incr(metric: str, delta: int = 1) -> None:
    with _metrics_lock:
        _metrics[metric] += delta


def _snapshot_metrics() -> dict[str, int]:
    with _metrics_lock:
        return dict(_metrics)


def _canonical_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _response_from_cache(cached: bytes) -> Response:
    return Response(content=cached, media_type="application/json")


@app.post("/v1/chat/completions")
def chat_completions(payload: Any = Body(...)) -> Response:
    _incr("requests_total")
    req_hash = _canonical_hash(payload)
    inflight_key = f"inflight:{req_hash}"
    done_key = f"done:{req_hash}"

    try:
        cached = redis_client.get(done_key)
    except redis.RedisError:
        return Response(content=b"Redis error", status_code=503)

    if cached is not None:
        _incr("cache_hits")
        return _response_from_cache(cached)

    try:
        won_race = redis_client.set(inflight_key, b"1", nx=True, ex=30)
    except redis.RedisError:
        return Response(content=b"Redis error", status_code=503)

    if won_race:
        try:
            cached = redis_client.get(done_key)
        except redis.RedisError:
            cached = None
        if cached is not None:
            _incr("cache_hits")
            redis_client.delete(inflight_key)
            return _response_from_cache(cached)

        _incr("new_inference_calls")
        try:
            upstream_resp = httpx.post(UPSTREAM_CHAT_URL, json=payload, timeout=30.0)
            body = upstream_resp.content
            redis_client.set(done_key, body, ex=60)
            return Response(
                content=body,
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type", "application/json"),
            )
        finally:
            redis_client.delete(inflight_key)

    _incr("coalesced_requests")
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            cached = redis_client.get(done_key)
        except redis.RedisError:
            cached = None
        if cached is not None:
            return _response_from_cache(cached)
        time.sleep(0.1)

    upstream_resp = httpx.post(UPSTREAM_CHAT_URL, json=payload, timeout=30.0)
    body = upstream_resp.content
    redis_client.set(done_key, body, ex=60)
    return Response(
        content=body,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )


@app.get("/metrics")
def metrics() -> JSONResponse:
    snapshot = _snapshot_metrics()
    saved = snapshot["cache_hits"] + snapshot["coalesced_requests"]
    total = snapshot["requests_total"]
    hit_rate = round((saved / total) * 100, 1) if total else 0.0
    snapshot["saved_inference_calls"] = saved
    snapshot["hit_rate_pct"] = hit_rate
    return JSONResponse(snapshot)


class _UpstreamHandler(BaseHTTPRequestHandler):
    calls = 0
    lock = threading.Lock()

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        with _UpstreamHandler.lock:
            _UpstreamHandler.calls += 1
        time.sleep(0.2)
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "stub response"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _start_stub_server(port: int = 8080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _start_proxy_server(port: int = 8090) -> uvicorn.Server:
    print(f"Coalescing proxy listening on http://127.0.0.1:{port} -> {UPSTREAM_URL}")
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def _wait_for_url(url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {url}")


def _post_chat(payload: dict[str, Any]) -> httpx.Response:
    return httpx.post("http://127.0.0.1:8090/v1/chat/completions", json=payload, timeout=10.0)


if __name__ == "__main__":
    try:
        redis_client.ping()
    except redis.RedisError:
        print("Error: Pion is not running on port 1974. Start pion-server and re-run.")
        raise SystemExit(1)

    stub_server = _start_stub_server()
    proxy_server = _start_proxy_server()
    _wait_for_url("http://127.0.0.1:8090/metrics")

    payload_a = {"model": "demo", "messages": [{"role": "user", "content": "Hello"}]}
    payload_b = {"model": "demo", "messages": [{"role": "user", "content": "Different"}]}

    for payload in (payload_a, payload_b):
        req_hash = _canonical_hash(payload)
        redis_client.delete(f"done:{req_hash}")
        redis_client.delete(f"inflight:{req_hash}")

    metrics_before = httpx.get("http://127.0.0.1:8090/metrics", timeout=2.0).json()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_post_chat, payload_a) for _ in range(10)]
        for future in futures:
            future.result()
    metrics_after = httpx.get("http://127.0.0.1:8090/metrics", timeout=2.0).json()

    delta_calls = _UpstreamHandler.calls
    delta_coalesced = metrics_after["coalesced_requests"] - metrics_before["coalesced_requests"]
    delta_new = metrics_after["new_inference_calls"] - metrics_before["new_inference_calls"]

    print(f"Scenario 1: stub inference calls = {delta_calls} (expected 1)")
    print(f"Scenario 1: coalesced requests = {delta_coalesced} (expected ~9)")
    assert delta_new == 1, "Expected exactly 1 new inference call for identical requests"
    assert delta_coalesced >= 8, "Expected at least 8 coalesced requests for identical requests"

    metrics_before = metrics_after
    start_calls = _UpstreamHandler.calls
    batch = [
        (payload_a, 0.00),
        (payload_a, 0.02),
        (payload_a, 0.04),
        (payload_b, 0.01),
        (payload_b, 0.03),
    ]

    def send_with_delay(payload: dict[str, Any], delay: float) -> None:
        time.sleep(delay)
        _post_chat(payload)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(send_with_delay, payload, delay) for payload, delay in batch]
        for future in futures:
            future.result()

    metrics_after = httpx.get("http://127.0.0.1:8090/metrics", timeout=2.0).json()
    delta_new = metrics_after["new_inference_calls"] - metrics_before["new_inference_calls"]
    delta_calls = _UpstreamHandler.calls - start_calls

    print(f"Scenario 2: stub inference calls = {delta_calls} (expected 2)")
    print(f"Scenario 2: new inference calls = {delta_new} (expected <= 2)")
    assert delta_new <= 2, "Expected at most 2 new inference calls for two unique prompts"

    final_metrics = httpx.get("http://127.0.0.1:8090/metrics", timeout=2.0).json()
    print("Final /metrics:")
    print(json.dumps(final_metrics, indent=2))

    proxy_server.should_exit = True
    stub_server.shutdown()
