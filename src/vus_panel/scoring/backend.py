"""Pinned checkpoint materialization and GPU adapter, imported only by the service."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
from pathlib import Path
from urllib.request import urlopen

from vus_panel.scoring.contracts import CHECKPOINT_REVISION, CHECKPOINT_SHA256, CONFIG_SHA256, digest_json

CHECKPOINT_SIZE = 13766621200
CHECKPOINT_URL = f"https://huggingface.co/arcinstitute/evo2_7b/resolve/{CHECKPOINT_REVISION}/evo2_7b.pt"
SOURCE_HASHES = {
    "models.py": "0b757fcc42b37a52b0c3f508e876a52356331392ae8aa16a923576265043b78c",
    "scoring.py": "17e6985cef3a62c2a488133ab5a65050dcf0abeec2f55d1c0b068dfe37c44de6",
    "utils.py": "1f59e32e38cac3f20f8dc48a51041950184268eb4a0dbfa618316a9c5af3fc66",
    "configs/evo2-7b-1m.yml": CONFIG_SHA256,
}


class PrecisionChanged(RuntimeError):
    """The model no longer matches the precision frozen after warmup."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_checkpoint(path: Path) -> None:
    if path.stat().st_size != CHECKPOINT_SIZE or sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the pinned evo2_7b revision and SHA256")


def verify_source(package_dir: Path) -> None:
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(package_dir / relative) != expected:
            raise ValueError(f"installed Arc source differs from pinned revision: {relative}")


def materialize_checkpoint(directory: Path) -> Path:
    """Download only on explicit invocation; never use Evo2's unversioned cache."""
    target = directory / CHECKPOINT_REVISION / "evo2_7b.pt"
    if target.exists():
        verify_checkpoint(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.partial")
    # Refuse stale incomplete downloads instead of silently replacing them.
    with temporary.open("xb") as output:
        with urlopen(CHECKPOINT_URL, timeout=120) as response:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    verify_checkpoint(temporary)
    temporary.rename(target)
    manifest = {"url": CHECKPOINT_URL, "revision": CHECKPOINT_REVISION,
                "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


class Evo2Backend:
    """One model load, fixed scoring calls, and measured precision snapshots."""

    def __init__(self, checkpoint: Path):
        verify_checkpoint(checkpoint)
        # GPU dependencies intentionally remain optional in the CPU application.
        import torch
        from evo2 import Evo2

        package_dir = Path(inspect.getfile(Evo2)).parent
        verify_source(package_dir)
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required for the real Evo 2 pilot")
        self.torch = torch
        self.evo = Evo2("evo2_7b", local_path=str(checkpoint.resolve()), use_kernels=False)
        self.evo.model.eval()
        if len(self.evo.model.blocks) != 32:
            raise ValueError("full evo2_7b must have 32 layers")
        self._logit_dtype: str | None = None
        self._frozen_precision: dict | None = None
        self.evo.model.register_forward_hook(self._observe_logits)
        self.dependencies = {"python": platform.python_version()}
        for name in ("evo2", "vtx", "torch", "numpy", "transformer-engine", "flash-attn", "huggingface-hub",
                     "triton", "biopython", "safetensors", "pydantic", "fastapi", "uvicorn"):
            try:
                self.dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.dependencies[name] = "not installed"
        self.dependencies["cuda_runtime"] = str(torch.version.cuda)
        self.dependencies["cudnn"] = str(torch.backends.cudnn.version())
        self.gpu = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            self.gpu.append({"index": index, "name": props.name, "total_memory_bytes": props.total_memory,
                             "compute_capability": list(torch.cuda.get_device_capability(index))})

    def _observe_logits(self, module, inputs, output) -> None:
        logits = output[0] if isinstance(output, tuple) else output
        if tuple(logits.shape) != (1, 8192, 512):
            raise ValueError("model must emit one 8192-token, 512-vocabulary logit tensor")
        observed = str(logits.dtype)
        previous = self._logit_dtype
        self._logit_dtype = observed
        if previous is not None and observed != previous:
            raise PrecisionChanged("model logit dtype changed")

    def synchronize(self) -> None:
        for index in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(index)

    def precision(self) -> dict:
        cfg = json.loads(json.dumps(dict(self.evo.model.config), sort_keys=True, default=str, allow_nan=False))
        fp8 = bool(cfg.get("use_fp8_input_projections", False))
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in self.evo.model.parameters()})
        return {
            "parameter_dtypes": parameter_dtypes,
            "projection_parameter_dtypes": {name: str(parameter.dtype)
                                            for name, parameter in self.evo.model.named_parameters()
                                            if "proj" in name},
            "logit_dtype": self._logit_dtype,
            "use_fp8_input_projections": fp8,
            "projection_mode": "fp8" if fp8 else "bf16",
            "log_softmax_dtype": self._logit_dtype,
            "reduction_dtype": "float32",
            "effective_config_sha256": digest_json(cfg),
            "effective_config": cfg,
            "source_file_sha256": dict(SOURCE_HASHES),
            "tf32_matmul": self.torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": self.torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": self.torch.get_float32_matmul_precision(),
        }

    def freeze_precision(self) -> dict:
        observed = self.precision()
        if observed["logit_dtype"] is None:
            raise ValueError("warmup must observe logits before precision is frozen")
        self._frozen_precision = observed
        return observed

    def score(self, sequence: str) -> float:
        if len(sequence) != 8192 or set(sequence) - set("ACGT"):
            raise ValueError("backend only accepts 8192 ACGT bases")
        if self._frozen_precision is not None and self.precision() != self._frozen_precision:
            raise PrecisionChanged("model precision or effective configuration changed")
        values = self.evo.score_sequences(
            [sequence], batch_size=1, prepend_bos=False, reduce_method="sum", average_reverse_complement=False,
        )
        if len(values) != 1:
            raise ValueError("expected one sequence likelihood")
        if self._frozen_precision is not None and self.precision() != self._frozen_precision:
            raise PrecisionChanged("model precision or effective configuration changed")
        return float(values[0])
