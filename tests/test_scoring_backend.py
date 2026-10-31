"""Pinned artifact verification without weights, CUDA, or remote calls."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from vus_panel.scoring import backend
from vus_panel.scoring.contracts import CHECKPOINT_REVISION


@pytest.fixture
def checkpoint_stub(monkeypatch):
    content = b"abstract unit-test checkpoint bytes"
    monkeypatch.setattr(backend, "CHECKPOINT_SIZE", len(content))
    monkeypatch.setattr(backend, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    return content


def test_materialization_is_revision_specific_verified_and_reused(tmp_path, monkeypatch, checkpoint_stub):
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(checkpoint_stub)

    monkeypatch.setattr(backend, "urlopen", download)
    path = backend.materialize_checkpoint(tmp_path)
    assert path.parent.name == CHECKPOINT_REVISION
    assert CHECKPOINT_REVISION in calls[0]
    assert path.read_bytes() == checkpoint_stub
    assert backend.materialize_checkpoint(tmp_path) == path
    assert len(calls) == 1
    assert json.loads(path.with_suffix(".json").read_text())["sha256"] == hashlib.sha256(checkpoint_stub).hexdigest()


def test_existing_corrupt_checkpoint_never_overwritten(tmp_path, monkeypatch, checkpoint_stub):
    path = tmp_path / CHECKPOINT_REVISION / "evo2_7b.pt"
    path.parent.mkdir()
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="pinned"):
        backend.materialize_checkpoint(tmp_path)
    assert path.read_bytes() == b"bad"


def test_download_hash_mismatch_does_not_publish_checkpoint(tmp_path, monkeypatch, checkpoint_stub):
    monkeypatch.setattr(backend, "urlopen", lambda *args, **kwargs: io.BytesIO(b"x" * len(checkpoint_stub)))
    with pytest.raises(ValueError, match="SHA256"):
        backend.materialize_checkpoint(tmp_path)
    assert not (tmp_path / CHECKPOINT_REVISION / "evo2_7b.pt").exists()


def test_stale_partial_download_refused(tmp_path, monkeypatch, checkpoint_stub):
    path = tmp_path / CHECKPOINT_REVISION / "evo2_7b.pt.partial"
    path.parent.mkdir()
    path.write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        backend.materialize_checkpoint(tmp_path)
    assert path.read_bytes() == b"partial"


def test_source_verification_checks_actual_file_bytes(tmp_path, monkeypatch):
    source = tmp_path / "scoring.py"
    source.write_bytes(b"verified source")
    monkeypatch.setattr(backend, "SOURCE_HASHES", {"scoring.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    backend.verify_source(tmp_path)
    source.write_bytes(b"modified source")
    with pytest.raises(ValueError, match="scoring.py"):
        backend.verify_source(tmp_path)
