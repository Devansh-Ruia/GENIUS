"""Offline wire validation and import isolation of the shared scoring client."""

from __future__ import annotations

from urllib.error import URLError
import traceback

import pytest
from pydantic import ValidationError

from _factories import placeholder_variant
from test_isolation import closure
from test_scoring_service import request, service
from vus_panel.schemas import AnswerKeyError
from vus_panel.scoring import client as client_module
from vus_panel.scoring.client import ScoringClient, ScoringClientError
from vus_panel.scoring.contracts import ScoreRequest, ScoreResponse, ServiceMetadata


def stub_client(monkeypatch):
    instance = service()
    calls = []
    client = ScoringClient("http://test.invalid")

    def transport(path, payload=None):
        calls.append((path, payload))
        if path == "/metadata":
            return instance.metadata.model_dump(mode="json")
        return instance.score(ScoreRequest.model_validate(payload)).model_dump(mode="json")

    monkeypatch.setattr(client, "_request", transport)
    return client, instance, calls


def test_client_sends_only_five_fields_and_preserves_partial_failure(monkeypatch):
    client, instance, calls = stub_client(monkeypatch)
    instance.backend.fail_at = 4
    result = client.score(request())
    assert result.status == "error" and result.ref_total is not None
    assert calls[-1] == ("/score", request().model_dump())
    assert len(calls[-1][1]) == 5


def test_client_rejects_answer_key_before_transport():
    with pytest.raises(AnswerKeyError):
        ScoreRequest.from_variant(placeholder_variant(findlay_score=0))


@pytest.mark.parametrize("field,value", [
    ("delta", 15), ("ref_total", float("inf")), ("alt_total", float("nan")),
    ("variant_id", "another-variant"), ("backend_seconds", -1),
])
def test_response_rejects_malformed_values(field, value):
    payload = service().score(request()).model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ScoreResponse.model_validate(payload)


def test_context_coordinates_validated():
    payload = service().score(request()).model_dump()
    payload["context"]["start"] += 1
    payload["context"]["end"] += 1
    with pytest.raises(ValidationError, match="centered"):
        ScoreResponse.model_validate(payload)


def test_client_rejects_changed_metadata_before_next_score(monkeypatch):
    client, instance, calls = stub_client(monkeypatch)
    client.score(request())
    instance.metadata = instance.metadata.model_copy(update={"session_id": "new-process"})
    with pytest.raises(ScoringClientError, match="changed") as error:
        client.score(request())
    assert error.value.fatal
    assert sum(path == "/score" for path, _ in calls) == 1


def test_client_retains_response_for_mid_request_session_drift(monkeypatch):
    client, instance, _ = stub_client(monkeypatch)
    payload = instance.score(request()).model_dump(mode="json")
    payload["session_id"] = "different-session"
    monkeypatch.setattr(client, "_request", lambda path, data=None: instance.metadata.model_dump(mode="json")
                        if path == "/metadata" else payload)
    with pytest.raises(ScoringClientError) as error:
        client.score(request())
    assert error.value.fatal and error.value.response is not None


def test_transport_failure_never_retried(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise URLError("offline")

    monkeypatch.setattr(client_module, "urlopen", fail)
    with pytest.raises(ScoringClientError) as error:
        ScoringClient("http://test.invalid").score(request())
    assert not error.value.fatal and len(calls) == 1


def test_invalid_json_is_fatal(monkeypatch):
    class BadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"not json"

    monkeypatch.setattr(client_module, "urlopen", lambda *args, **kwargs: BadResponse())
    with pytest.raises(ScoringClientError) as error:
        ScoringClient("http://test.invalid").metadata()
    assert error.value.fatal


def test_client_import_closure_excludes_gpu_and_evaluation():
    reached = closure("vus_panel.scoring.client")
    assert not any(name.startswith("vus_panel.eval") for name in reached)
    assert "vus_panel.scoring.service" not in reached
    assert "vus_panel.scoring.backend" not in reached


def test_nested_nonfinite_metadata_is_rejected():
    payload = service().metadata.model_dump()
    payload["precision"]["unexpected"] = float("nan")
    with pytest.raises(ValueError):
        ServiceMetadata.model_validate(payload)


@pytest.mark.parametrize("field", list(ServiceMetadata.model_fields))
def test_client_rejects_missing_metadata_fields_instead_of_applying_defaults(monkeypatch, field):
    payload = service().metadata.model_dump(mode="json")
    del payload[field]
    client = ScoringClient("http://test.invalid")
    monkeypatch.setattr(client, "_request", lambda *args: payload)
    with pytest.raises(ScoringClientError) as error:
        client.metadata()
    assert error.value.fatal


@pytest.mark.parametrize("change", ["absent", "partial", "coerced"])
def test_client_requires_complete_explicit_response_protocol(monkeypatch, change):
    client, instance, _ = stub_client(monkeypatch)
    payload = instance.score(request()).model_dump(mode="json")
    if change == "absent":
        del payload["protocol"]
    elif change == "partial":
        del payload["protocol"]["prepend_bos"]
    else:
        payload["protocol"]["prepend_bos"] = 0
    monkeypatch.setattr(client, "_request", lambda path, data=None: instance.metadata.model_dump(mode="json")
                        if path == "/metadata" else payload)
    with pytest.raises(ScoringClientError, match="protocol") as error:
        client.score(request())
    assert error.value.fatal


@pytest.mark.parametrize("change", ["missing_name", "incorrect_hash"])
def test_client_rejects_incomplete_or_inconsistent_warmup(monkeypatch, change):
    client = ScoringClient("http://test.invalid")
    payload = service().metadata.model_dump(mode="json")
    if change == "missing_name":
        del payload["warmup"]["name"]
    else:
        payload["warmup"]["ref_sha256"] = "f" * 64
    monkeypatch.setattr(client, "_request", lambda *args: payload)
    with pytest.raises(ScoringClientError) as error:
        client.metadata()
    assert error.value.fatal


@pytest.mark.parametrize("endpoint", ["/metadata", "/score"])
def test_invalid_backend_answer_key_never_appears_in_error_or_traceback(monkeypatch, endpoint):
    instance = service()
    metadata = instance.metadata.model_dump(mode="json")
    response = instance.score(request()).model_dump(mode="json")
    payload = metadata if endpoint == "/metadata" else response
    payload["findlay_function_class"] = "private-label-marker"
    payload["findlay_score"] = -987654321
    client = ScoringClient("http://test.invalid")
    monkeypatch.setattr(client, "_request", lambda path, data=None: metadata if path == "/metadata" else response)
    with pytest.raises(ScoringClientError) as error:
        client.score(request())
    rendered = "".join(traceback.format_exception(error.value))
    assert "private-label-marker" not in rendered
    assert "987654321" not in rendered
    assert error.value.response is None and error.value.fatal
