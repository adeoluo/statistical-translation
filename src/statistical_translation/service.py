"""A dependency-free HTTP service exposing the trained alignment models.

The service trains Model 1 and Model 2 once at startup over the bundled corpus and
then aligns sentence pairs, translates words, and reports held-out alignment error
rate. A pure handler is separated from the socket binding for testing.

Endpoints
---------
``GET  /health``     liveness and training-corpus size.
``GET  /ready``      readiness once the models are trained.
``GET  /metrics``    per-route request, error, and latency metrics (Prometheus text).
``POST /align``      body ``{"source", "target"}``; word alignment (Model 2).
``POST /translate``  body ``{"word", "k"?}``; most probable translations (Model 1).
``GET  /evaluate``   alignment error rate for both models on held-out data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from statistical_translation.corpus import SentencePair
from statistical_translation.data import gold_pairs, train_pairs
from statistical_translation.evaluation import alignment_error_rate
from statistical_translation.ibm1 import IBMModel1
from statistical_translation.ibm2 import IBMModel2
from statistical_translation.observability import ServiceMetrics

JsonObject = dict[str, Any]
_MAX_CHARS = 2000
_MAX_K = 25


@dataclass(frozen=True)
class TranslationService:
    """Holds the trained models and answers alignment and translation requests."""

    model1: IBMModel1
    model2: IBMModel2
    corpus_size: int
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, iterations: int = 15) -> TranslationService:
        pairs = train_pairs()
        model1 = IBMModel1()
        model1.train(pairs, iterations=iterations)
        model2 = IBMModel2()
        model2.train(pairs, iterations=iterations)
        return cls(model1=model1, model2=model2, corpus_size=len(pairs))

    def align(self, source: str, target: str) -> JsonObject:
        pair = SentencePair.from_text(source, target)
        source_tokens = pair.source_with_null()
        links = []
        for target_index, source_index in self.model2.align(pair).pairs:
            source_word = source_tokens[source_index] if source_index > 0 else None
            links.append(
                {
                    "target": pair.target[target_index],
                    "source": source_word,
                    "target_index": target_index,
                    "source_index": source_index,
                }
            )
        return {"source": list(pair.source), "target": list(pair.target), "alignment": links}

    def translate(self, word: str, k: int) -> JsonObject:
        results = self.model1.translate_word(word, k=k)
        return {
            "word": word,
            "translations": [{"word": w, "probability": round(p, 6)} for w, p in results],
        }

    def evaluate(self) -> JsonObject:
        gold = gold_pairs()
        return {
            "model1_aer": round(alignment_error_rate(self.model1, gold).aer, 6),
            "model2_aer": round(alignment_error_rate(self.model2, gold).aer, 6),
        }


def _text(body: Mapping[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    if len(value) > _MAX_CHARS:
        raise ValueError(f"'{key}' exceeds {_MAX_CHARS} characters")
    return value


def _route(
    method: str, path: str, body: Mapping[str, Any], service: TranslationService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "corpus_size": service.corpus_size}
    if method == "GET" and path == "/evaluate":
        return 200, service.evaluate()
    if method == "POST" and path == "/align":
        try:
            source = _text(body, "source")
            target = _text(body, "target")
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.align(source, target)
    if method == "POST" and path == "/translate":
        try:
            word = _text(body, "word")
        except ValueError as error:
            return 400, {"error": str(error)}
        k = body.get("k", 3)
        if not isinstance(k, int) or not 1 <= k <= _MAX_K:
            return 400, {"error": f"'k' must be an integer in [1, {_MAX_K}]"}
        return 200, service.translate(word, k)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: TranslationService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "corpus_size": service.corpus_size}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: TranslationService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": TranslationService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"statistical-translation service listening on http://{host}:{port}")
    server.serve_forever()
