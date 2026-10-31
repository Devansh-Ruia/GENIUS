"""Label-free, finite, immutable wire contracts for the timing pilot."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from vus_panel.schemas import Variant, reject_answer_key

SOURCE_REVISION = "53f195997257c56c00e5ef8d33a54f5baad143a6"
CHECKPOINT_REVISION = "bda0089f92582d5baabf0f22d9fc85f3588f6b58"
CHECKPOINT_SHA256 = "c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4"
CONFIG_SHA256 = "9637f84ffd796f42cce3f926832b3999ff34a82686beba04505327162b288b1a"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Total = Annotated[float, Field(le=0, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Protocol(Contract):
    name: Literal["evo2-findlay-pilot-v1"] = "evo2-findlay-pilot-v1"
    context_length: Literal[8192] = 8192
    variant_index: Literal[4096] = 4096
    strand: Literal["genomic-forward"] = "genomic-forward"
    batch_size: Literal[1] = 1
    prepend_bos: Literal[False] = False
    reduce_method: Literal["sum"] = "sum"
    average_reverse_complement: Literal[False] = False
    next_token_terms: Literal[8191] = 8191
    log_base: Literal["e"] = "e"
    delta_direction: Literal["alt-minus-ref"] = "alt-minus-ref"
    reference_score_cache: Literal[False] = False


class ScoreRequest(Contract):
    chrom: Literal["17"]
    pos: int = Field(gt=0, strict=True)
    ref: str = Field(pattern=r"^[ACGT]$")
    alt: str = Field(pattern=r"^[ACGT]$")
    genome_build: Literal["GRCh37"]

    @model_validator(mode="after")
    def distinct(self) -> "ScoreRequest":
        if self.ref == self.alt:
            raise ValueError("ref and alt must differ")
        return self

    @property
    def variant_id(self) -> str:
        return f"{self.genome_build}-{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    @classmethod
    def from_variant(cls, variant: Variant) -> "ScoreRequest":
        reject_answer_key(variant, "scoring client")
        return cls(**variant.model_dump(include={"chrom", "pos", "ref", "alt", "genome_build"}))


class ContextRecord(Contract):
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    variant_index: Literal[4096] = 4096
    ref_sha256: Digest
    alt_sha256: Digest

    @model_validator(mode="after")
    def window(self) -> "ContextRecord":
        if self.end - self.start + 1 != 8192 or self.ref_sha256 == self.alt_sha256:
            raise ValueError("context must describe distinct 8192-base sequences")
        return self


class WarmupRecord(Contract):
    name: Literal["ACGT-repeat-index4096-A-to-C"] = "ACGT-repeat-index4096-A-to-C"
    ref_sha256: Digest
    alt_sha256: Digest
    ref_total: Total
    alt_total: Total
    ref_seconds: Seconds
    alt_seconds: Seconds
    backend_seconds: Seconds

    @model_validator(mode="after")
    def fixed_pair(self) -> "WarmupRecord":
        ref = "ACGT" * 2048
        alt = ref[:4096] + "C" + ref[4097:]
        if self.ref_sha256 != hashlib.sha256(ref.encode()).hexdigest():
            raise ValueError("warmup REF checksum does not match the fixed synthetic pair")
        if self.alt_sha256 != hashlib.sha256(alt.encode()).hexdigest():
            raise ValueError("warmup ALT checksum does not match the fixed synthetic pair")
        return self


class ServiceMetadata(Contract):
    session_id: str = Field(min_length=1)
    created_at: AwareDatetime
    protocol: Protocol = Field(default_factory=Protocol)
    model_name: Literal["evo2_7b"] = "evo2_7b"
    source_revision: Literal[SOURCE_REVISION] = SOURCE_REVISION
    checkpoint_revision: Literal[CHECKPOINT_REVISION] = CHECKPOINT_REVISION
    checkpoint_sha256: Literal[CHECKPOINT_SHA256] = CHECKPOINT_SHA256
    config_sha256: Literal[CONFIG_SHA256] = CONFIG_SHA256
    reference_accession: Literal["NC_000017.10"] = "NC_000017.10"
    genome_build: Literal["GRCh37"] = "GRCh37"
    reference_sequence_sha256: Digest
    reference_file_sha256: Digest
    layer_count: Literal[32] = 32
    precision: dict[str, Any]
    dependencies: dict[str, str]
    gpu: list[dict[str, Any]] = Field(min_length=1)
    load_seconds: Seconds
    warmup: WarmupRecord

    @model_validator(mode="after")
    def observed_precision(self) -> "ServiceMetadata":
        required = {"parameter_dtypes", "logit_dtype", "use_fp8_input_projections", "projection_mode",
                    "log_softmax_dtype", "reduction_dtype", "effective_config_sha256"}
        if not required <= self.precision.keys():
            raise ValueError("metadata is missing observed precision settings")
        if self.precision["reduction_dtype"] != "float32":
            raise ValueError("pinned scoring implementation reduces float32 values")
        if not self.precision["parameter_dtypes"] or not self.precision["logit_dtype"]:
            raise ValueError("precision must be observed during warmup")
        if self.precision["logit_dtype"] not in {"torch.float32", "torch.bfloat16", "torch.float16"}:
            raise ValueError("unsupported observed logit precision")
        if self.precision["log_softmax_dtype"] != self.precision["logit_dtype"]:
            raise ValueError("pinned scoring computes log_softmax in logit precision")
        fp8 = self.precision["use_fp8_input_projections"]
        if type(fp8) is not bool or self.precision["projection_mode"] != ("fp8" if fp8 else "bf16"):
            raise ValueError("inconsistent projection precision")
        # Also rejects nonfinite values nested in otherwise arbitrary metadata.
        digest_json(self.precision)
        digest_json(self.gpu)
        return self

    @property
    def sha256(self) -> str:
        return digest_json(self.model_dump(mode="json"))


class ScoreResponse(Contract):
    variant_id: str
    request: ScoreRequest
    session_id: str
    metadata_sha256: Digest
    protocol: Protocol = Field(default_factory=Protocol)
    status: Literal["ok", "error"]
    context: ContextRecord | None = None
    ref_total: Total | None = None
    alt_total: Total | None = None
    delta: float | None = Field(default=None, allow_inf_nan=False)
    ref_seconds: Seconds | None = None
    alt_seconds: Seconds | None = None
    backend_seconds: Seconds
    error: str | None = None
    error_code: Literal["context", "inference", "metadata_drift"] | None = None

    @model_validator(mode="after")
    def consistent(self) -> "ScoreResponse":
        if self.variant_id != self.request.variant_id:
            raise ValueError("response variant identity does not match request")
        if self.context and self.context.start + 4096 != self.request.pos:
            raise ValueError("response context is not centered at index 4096")
        if self.alt_total is not None and self.ref_total is None:
            raise ValueError("ALT cannot precede REF")
        if self.ref_total is not None and (self.context is None or self.ref_seconds is None):
            raise ValueError("REF score requires context and timing")
        if self.alt_total is not None and self.alt_seconds is None:
            raise ValueError("ALT score requires timing")
        if self.delta is not None:
            if self.ref_total is None or self.alt_total is None or self.delta != self.alt_total - self.ref_total:
                raise ValueError("delta must equal saved ALT minus REF totals")
        if self.status == "ok":
            if self.delta is None or self.error is not None or self.error_code is not None:
                raise ValueError("successful response requires both scores and no error")
        elif not self.error:
            raise ValueError("failed response requires an error")
        return self


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
