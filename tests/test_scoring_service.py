"""Offline service protocol checks using abstract sequences and deterministic stubs."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vus_panel.scoring.backend import Evo2Backend
from vus_panel.scoring.contracts import ScoreRequest
from vus_panel.scoring.service import ScoringService, create_app, matched_contexts


def request(**kwargs):
    return ScoreRequest(chrom="17", pos=4097, ref="A", alt="C", genome_build="GRCh37", **kwargs)


class StubBackend:
    dependencies = {"test_backend": "offline"}
    gpu = [{"name": "abstract test device"}]

    def __init__(self):
        self.calls = []
        self.syncs = 0
        self.fail_at = None
        self.settings = {
            "parameter_dtypes": ["torch.bfloat16"], "logit_dtype": "torch.bfloat16",
            "log_softmax_dtype": "torch.bfloat16", "use_fp8_input_projections": False,
            "projection_mode": "bf16", "reduction_dtype": "float32", "effective_config_sha256": "a" * 64,
        }

    def score(self, sequence):
        self.calls.append(sequence)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("injected inference failure")
        # Includes every next-token term and a downstream term affected by context.
        terms = np.full(8191, -1, dtype=np.float32)
        terms[-1] = -3 if sequence[4096] == "C" else -1
        return float(np.sum(terms))

    def synchronize(self):
        self.syncs += 1

    def precision(self):
        return deepcopy(self.settings)

    def freeze_precision(self):
        return self.precision()


def service():
    backend = StubBackend()
    return ScoringService(backend, "ACGT" * 3000, "b" * 64, 0.25)


def test_matched_context_is_forward_exact_size_and_index():
    sequence = "ACGT" * 3000
    ref, alt, context = matched_contexts(request(), sequence)
    assert len(ref) == len(alt) == 8192
    assert ref == sequence[:8192]
    assert [index for index in range(8192) if ref[index] != alt[index]] == [4096]
    assert (context.start, context.end) == (1, 8192)
    assert ref[4096] == "A" and alt[4096] == "C"


@pytest.mark.parametrize("updates,sequence", [
    ({"pos": 4096}, "ACGT" * 3000),
    ({"pos": 10001}, "ACGT" * 3000),
    ({"ref": "G"}, "ACGT" * 3000),
    ({}, "N" + "ACGT" * 3000),
])
def test_context_rejects_bounds_ref_mismatch_and_unsupported_bases(updates, sequence):
    variant = request().model_copy(update=updates)
    with pytest.raises(ValueError):
        matched_contexts(variant, sequence)


def test_warmup_separate_once_and_no_reference_cache():
    instance = service()
    backend = instance.backend
    assert len(backend.calls) == 2
    assert backend.calls[0] == "ACGT" * 2048
    assert instance.metadata.warmup.ref_total == -8191
    assert instance.metadata.warmup.alt_total == -8193
    first = instance.score(request())
    second = instance.score(request())
    assert len(backend.calls) == 6
    assert backend.syncs == 12
    assert first.status == second.status == "ok"
    assert first.ref_total == -8191 and first.alt_total == -8193 and first.delta == -2
    assert first.delta == first.alt_total - first.ref_total
    assert first.backend_seconds >= first.ref_seconds + first.alt_seconds


def test_alt_failure_retains_saved_ref_and_failure_timing():
    instance = service()
    instance.backend.fail_at = 4
    result = instance.score(request())
    assert result.status == "error" and result.error_code == "inference"
    assert result.ref_total == -8191 and result.alt_total is None and result.delta is None
    assert result.ref_seconds >= 0 and result.alt_seconds >= 0
    assert instance.backend.syncs == 8


def test_ref_failure_does_not_attempt_alt():
    instance = service()
    instance.backend.fail_at = 3
    result = instance.score(request())
    assert result.status == "error"
    assert result.ref_total is result.alt_total is result.delta is None
    assert len(instance.backend.calls) == 3


def test_precision_drift_rejects_before_scoring():
    instance = service()
    instance.backend.settings["parameter_dtypes"] = ["torch.float32"]
    result = instance.score(request())
    assert result.status == "error" and result.error_code == "metadata_drift"
    assert len(instance.backend.calls) == 2
    with TestClient(create_app(instance)) as client:
        assert client.get("/metadata").status_code == 409


@pytest.mark.parametrize("field", ["findlay_score", "findlay_function_class", "hgvs_c", "labels"])
def test_http_request_rejects_extra_fields_even_null(field):
    instance = service()
    with TestClient(create_app(instance)) as client:
        response = client.post("/score", json={**request().model_dump(), field: None})
    assert response.status_code == 422
    assert len(instance.backend.calls) == 2


def test_http_routes_and_frozen_metadata():
    instance = service()
    with TestClient(create_app(instance)) as client:
        first = client.get("/metadata").json()
        result = client.post("/score", json=request().model_dump()).json()
        second = client.get("/metadata").json()
    assert first == second
    assert result["metadata_sha256"] == instance.metadata.sha256
    assert result["status"] == "ok"


def test_gpu_adapter_passes_exact_pinned_scoring_arguments():
    calls = []

    def score(sequences, **kwargs):
        calls.append((sequences, kwargs))
        return [-8191.0]

    backend = Evo2Backend.__new__(Evo2Backend)
    backend.evo = SimpleNamespace(score_sequences=score)
    backend._frozen_precision = None
    assert backend.score("ACGT" * 2048) == -8191
    assert calls == [(["ACGT" * 2048], {"batch_size": 1, "prepend_bos": False, "reduce_method": "sum",
                                        "average_reverse_complement": False})]


def test_nonfinite_inference_rejected():
    instance = service()
    instance.backend.score = lambda sequence: float("nan")
    result = instance.score(request())
    assert result.status == "error" and result.ref_total is None


def test_wrapped_precision_drift_still_fatal_and_retains_ref():
    instance = service()
    original_score = instance.backend.score

    def changed(sequence):
        if len(instance.backend.calls) == 3:
            instance.backend.settings["logit_dtype"] = "torch.float32"
            raise RuntimeError("Arc wrapped a precision error")
        return original_score(sequence)

    instance.backend.score = changed
    result = instance.score(request())
    assert result.status == "error" and result.error_code == "metadata_drift"
    assert result.ref_total == -8191 and result.alt_total is None
