"""Execution provenance, limited to scientific dependencies and non-secret device settings."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from ace_backend.capabilities import inference_checkpoint_paths


def checkpoint_metadata() -> list[dict[str, str]]:
    assets = []
    for path in inference_checkpoint_paths():
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        assets.append({"identifier": path.name, "sha256": digest})
    return assets


def execution_environment(device: str) -> dict[str, object]:
    import torch
    from openbabel import openbabel

    properties = torch.cuda.get_device_properties(device)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        drivers = sorted(set(result.stdout.splitlines()))
    except (OSError, subprocess.SubprocessError):
        drivers = []
    versions = {"python": platform.python_version(), "openbabel": openbabel.OBReleaseVersion()}
    for package in (
        "torch",
        "rdkit",
        "numpy",
        "scipy",
        "biopython",
        "openbabel-wheel",
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
    ):
        versions[package] = importlib.metadata.version(package)
    return {
        "dependencies": versions,
        "sampler_device": device,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "nvidia_driver_versions": drivers,
        "torch_cuda_version": torch.version.cuda,
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True, timeout=30
    ).stdout


def code_metadata() -> dict[str, object]:
    # The backend is installed alongside the parent ACE sources, not an independent model distribution.
    import inference.condition_sampling

    root = Path(inference.condition_sampling.__file__).resolve().parents[2]
    submodules = _git(root, "submodule", "status", "--recursive", "--", ".")
    repositories = [
        (".", root),
        *((line[1:].split()[1], root / line[1:].split()[1]) for line in submodules.splitlines() if line),
    ]
    versions = []
    for name, repository in repositories:
        untracked_sources = {}
        for filename in _git(repository, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
            path = repository / filename
            if filename and path.suffix in {".py", ".yaml", ".yml", ".toml", ".cu", ".cpp", ".h"}:
                if name != "." or filename.startswith(("src/", "backend/src/")):
                    untracked_sources[filename] = path.read_text(encoding="utf-8")
        versions.append(
            {
                "repository": name,
                "commit": _git(repository, "rev-parse", "HEAD").strip(),
                "dirty": bool(_git(repository, "status", "--porcelain", "--", ".")),
                "diff": _git(repository, "diff", "--binary", "HEAD", "--", "."),
                "untracked_sources": untracked_sources,
            }
        )
    return {"repositories": versions}
