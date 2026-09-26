"""Substitute expensive model boundaries inside the real subprocess worker.

Only tests select this command. Production has no client-selectable fake mode.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

import ace_backend.worker as worker
import inference.condition_sampling as sampling
import inference.sampling_runtime as runtime
from configs.config_sampler import _BaseSamplerConfig

root, device, mode = sys.argv[1:]


def fake_load(**kwargs: object) -> object:
    if mode == "load_failure":
        raise RuntimeError("private /server/checkpoints/broken.ckpt")
    return object()


def fake_sample(
    condition: sampling.SamplingCondition, loaded: object, sampler: _BaseSamplerConfig, save_dir: Path
) -> sampling.SamplingResult:
    if mode == "sampling_failure":
        raise RuntimeError("private /server/sampling/details")
    if mode == "oom":
        raise torch.cuda.OutOfMemoryError("private GPU details")
    samples = [Chem.Mol(condition.ref_ligand) for _ in range(sampler.batch_size)]
    if mode == "all_invalid":
        samples = [None] * len(samples)
    elif len(samples) > 1:
        samples[1] = None
    return sampling.SamplingResult(
        condition=condition, xyz_blocks=[], samples=samples, logweight_trajectory=torch.empty(0), choices=np.empty(0)
    )


def fake_assets() -> list[dict[str, str]]:
    return [{"identifier": "substituted-test-model", "sha256": "0" * 64}]


def fake_environment(device: str) -> dict[str, str]:
    return {"sampler_device": device, "validation": "substituted models; no CUDA inference"}


runtime.load_sampling_runtime = fake_load
sampling.sample_condition = fake_sample
worker.checkpoint_metadata = fake_assets
worker.execution_environment = fake_environment
if mode == "write_failure":
    original_describe = worker.describe_artifact

    def fail_sdf_write(job_id: str, root: Path, relative_path: str, role: str, media_type: str):
        if role == "ligand":
            raise OSError("private /server/output/full")
        return original_describe(job_id, root, relative_path, role, media_type)

    worker.describe_artifact = fail_sdf_write

worker.run_job(Path(root), device)
