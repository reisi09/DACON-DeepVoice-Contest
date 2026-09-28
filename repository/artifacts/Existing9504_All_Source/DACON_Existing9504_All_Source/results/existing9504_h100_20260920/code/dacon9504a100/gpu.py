"""Device identity and numeric policy shared by every GPU child stage (isolated venv)."""
from __future__ import annotations

from typing import Any

import torch

from .common import PipelineError


def configure_numerics() -> None:
    """Same policy as the original full_bank_* scripts: fp32 math, no TF32, no autotune."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def require_device(device: str, expected_gpu: str | None) -> dict[str, Any]:
    """Production passes device='cuda', expected_gpu='H100'. CPU is reachable only from tests."""
    if device == "cuda":
        if not torch.cuda.is_available():
            raise PipelineError("CUDA is unavailable; this stage requires the authorized GPU")
        name = torch.cuda.get_device_name(0)
        if expected_gpu and expected_gpu not in name:
            raise PipelineError(f"This stage requires {expected_gpu}; found {name}")
        properties = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "name": name, "total_memory": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(0)), "torch": str(torch.__version__),
                "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if expected_gpu:
        raise PipelineError(f"device={device!r} cannot satisfy the required {expected_gpu}")
    return {"device": device, "name": "cpu", "torch": str(torch.__version__)}


def memory(device: str) -> dict[str, int]:
    if device != "cuda":
        return {}
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(), "max_reserved": torch.cuda.max_memory_reserved(),
            "total": torch.cuda.get_device_properties(0).total_memory}


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def release(device: str) -> None:
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
