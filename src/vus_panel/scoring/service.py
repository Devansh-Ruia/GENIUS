"""Single-process service for the frozen 8192-base genomic-forward pilot."""

from __future__ import annotations

import argparse
import hashlib
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from vus_panel.scoring.backend import Evo2Backend, PrecisionChanged, materialize_checkpoint, sha256_file
from vus_panel.scoring.contracts import ContextRecord, ScoreRequest, ScoreResponse, ServiceMetadata, WarmupRecord


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_reference(path: Path, expected_sequence_sha256: str) -> str:
    with path.open(encoding="ascii") as handle:
        header = next(handle, "")
        if not header.startswith(">NC_000017.10 "):
            raise ValueError("reference FASTA must identify accession NC_000017.10")
        sequence = "".join(line.strip() for line in handle)
    if sequence_hash(sequence) != expected_sequence_sha256:
        raise ValueError("reference sequence checksum does not match frozen preparation")
    if len(sequence) != 81195210:
        raise ValueError("NC_000017.10 must contain 81,195,210 bases")
    return sequence


def matched_contexts(request: ScoreRequest, sequence: str) -> tuple[str, str, ContextRecord]:
    start = request.pos - 1 - 4096
    end = start + 8192
    if start < 0 or end > len(sequence):
        raise ValueError("8192-base window is out of reference bounds; window will not be shifted")
    ref = sequence[start:end]
    if set(ref) - set("ACGT"):
        raise ValueError("context contains unsupported bases")
    if ref[4096] != request.ref:
        raise ValueError("genomic REF does not match sequence[position - 1]")
    alt = ref[:4096] + request.alt + ref[4097:]
    if sum(a != b for a, b in zip(ref, alt)) != 1:
        raise ValueError("REF and ALT contexts must differ at exactly one base")
    return ref, alt, ContextRecord(start=start + 1, end=end, ref_sha256=sequence_hash(ref), alt_sha256=sequence_hash(alt))


class ScoringService:
    def __init__(self, backend, sequence: str, reference_file_sha256: str, load_seconds: float):
        self.backend = backend
        self.sequence = sequence
        self.lock = threading.Lock()
        # This pair is independent of all pilot variants and has no biological labels.
        ref = "ACGT" * 2048
        alt = ref[:4096] + "C" + ref[4097:]
        started = time.perf_counter()
        ref_total, ref_seconds = self._score_one(ref)
        alt_total, alt_seconds = self._score_one(alt)
        warmup = WarmupRecord(ref_sha256=sequence_hash(ref), alt_sha256=sequence_hash(alt),
                              ref_total=ref_total, alt_total=alt_total, ref_seconds=ref_seconds,
                              alt_seconds=alt_seconds, backend_seconds=time.perf_counter() - started)
        self.metadata = ServiceMetadata(
            session_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc),
            reference_sequence_sha256=sequence_hash(sequence), reference_file_sha256=reference_file_sha256,
            precision=backend.freeze_precision(), dependencies=backend.dependencies, gpu=backend.gpu,
            load_seconds=load_seconds, warmup=warmup,
        )
        self.metadata_digest = self.metadata.sha256

    def _score_one(self, sequence: str) -> tuple[float, float]:
        self.backend.synchronize()
        started = time.perf_counter()
        try:
            score = float(self.backend.score(sequence))
        finally:
            self.backend.synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(score) or score > 0:
            raise ValueError("backend did not return a finite nonpositive log-likelihood")
        return score, elapsed

    def score(self, request: ScoreRequest) -> ScoreResponse:
        with self.lock:
            started = time.perf_counter()
            values = dict(variant_id=request.variant_id, request=request, session_id=self.metadata.session_id,
                          metadata_sha256=self.metadata_digest)
            stage = "metadata_drift"
            try:
                if self.backend.precision() != self.metadata.precision:
                    raise RuntimeError("backend precision metadata changed")
                stage = "context"
                ref, alt, context = matched_contexts(request, self.sequence)
                values["context"] = context
                stage = "inference"
                for name, sequence in (("ref", ref), ("alt", alt)):
                    score_started = time.perf_counter()
                    try:
                        values[f"{name}_total"], values[f"{name}_seconds"] = self._score_one(sequence)
                    finally:
                        values.setdefault(f"{name}_seconds", time.perf_counter() - score_started)
                stage = "metadata_drift"
                if self.backend.precision() != self.metadata.precision:
                    raise RuntimeError("backend precision metadata changed")
                values["delta"] = values["alt_total"] - values["ref_total"]
                values["status"] = "ok"
            except Exception as exc:
                values["status"] = "error"
                values["error"] = f"{type(exc).__name__}: {exc}"
                # Arc wraps inference exceptions, so inspect the observed settings too.
                drift = isinstance(exc, PrecisionChanged) or self.backend.precision() != self.metadata.precision
                values["error_code"] = "metadata_drift" if drift else stage
            return ScoreResponse(**values, backend_seconds=time.perf_counter() - started)


def create_app(service: ScoringService) -> FastAPI:
    app = FastAPI(title="Evo 2 frozen pilot scoring")

    @app.get("/metadata", response_model=ServiceMetadata)
    def metadata():
        if service.backend.precision() != service.metadata.precision:
            raise HTTPException(409, "backend precision metadata changed")
        return service.metadata

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest):
        return service.score(request)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize", help="explicitly download and verify pinned model weights")
    materialize.add_argument("--directory", type=Path, default=Path("data/cache/models"))
    serve = subparsers.add_parser("serve")
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--reference", type=Path, required=True)
    serve.add_argument("--reference-sequence-sha256", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        print(materialize_checkpoint(args.directory))
        return 0
    sequence = read_reference(args.reference, args.reference_sequence_sha256)
    started = time.perf_counter()
    backend = Evo2Backend(args.checkpoint)
    backend.synchronize()
    load_seconds = time.perf_counter() - started
    service = ScoringService(backend, sequence, sha256_file(args.reference), load_seconds)
    import uvicorn

    # An app object and one worker prevent re-import/model reload in worker processes.
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
