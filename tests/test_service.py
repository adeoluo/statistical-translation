"""Tests for the dependency-free HTTP alignment service (pure handler layer)."""

from __future__ import annotations

from statistical_translation.service import TranslationService, handle

_SERVICE = TranslationService.trained(iterations=12)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and payload["corpus_size"] > 0
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_align() -> None:
    status, payload = handle(
        "POST", "/align", {"source": "the red house", "target": "la maison rouge"}, _SERVICE
    )
    assert status == 200
    assert any(link["source"] == "house" for link in payload["alignment"])


def test_translate() -> None:
    status, payload = handle("POST", "/translate", {"word": "house", "k": 2}, _SERVICE)
    assert status == 200
    assert payload["translations"][0]["word"] == "maison"


def test_evaluate() -> None:
    status, payload = handle("GET", "/evaluate", {}, _SERVICE)
    assert status == 200
    assert 0.0 <= payload["model1_aer"] <= 1.0


def test_rejects_bad_input() -> None:
    status, _ = handle("POST", "/align", {"source": "", "target": "x"}, _SERVICE)
    assert status == 400
    status, _ = handle("POST", "/translate", {"word": "house", "k": 0}, _SERVICE)
    assert status == 400


def test_metrics_and_unknown_route() -> None:
    service = TranslationService.trained(iterations=8)
    handle("POST", "/translate", {"word": "book"}, service)
    text = service.metrics.render_prometheus()
    assert "service_requests_total" in text
    assert 'route="POST /translate"' in text
    status, payload = handle("GET", "/nope", {}, service)
    assert status == 404 and "error" in payload
