from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run(command: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def collect_metadata(upstream_root: str, seed: int, argv: list[str]) -> dict:
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        gpu = {
            "visible_device_index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "allocated_mb": torch.cuda.memory_allocated(index) / 2**20,
        }
    return {
        "seed": seed,
        "argv": argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
        "upstream_root": upstream_root,
        "upstream_commit": _run(["git", "rev-parse", "HEAD"], cwd=upstream_root),
        "upstream_status": _run(["git", "status", "--short"], cwd=upstream_root),
        "nvidia_smi": _run([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]),
    }


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
