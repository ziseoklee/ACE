"""Subprocess entry point. Never deserialize client-specified checkpoints or execute client commands."""

import ctypes
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ace_backend.inference_runtime import build_preset
from ace_backend.job_store import describe_artifact, utc_now, write_json
from ace_backend.jobs_schema import (
    INFERENCE_CONFIG_ADAPTER,
    JOB_ADAPTER,
    AvailableSample,
    Error,
    InferenceResult,
    InputRefs,
    InvalidSample,
    Manifest,
    Preparation,
    Sample,
    StoredArtifact,
    Summary,
    WarningMessage,
    WorkerFailure,
    WorkerSuccess,
)
from ace_backend.molecule_io import serialize_sdf
from ace_backend.provenance import checkpoint_metadata, code_metadata, execution_environment

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


def run_job(root: Path, device: str) -> None:
    phase = "loading_models"
    try:
        from omegaconf import OmegaConf
        from rdkit import Chem

        from inference.condition_sampling import SamplingCondition, sample_condition
        from inference.sampling_runtime import load_sampling_runtime

        config = INFERENCE_CONFIG_ADAPTER.validate_json((root / "request.json").read_bytes())
        preparation = Preparation.model_validate_json((root / "preparation.json").read_bytes())
        job = JOB_ADAPTER.validate_json((root / "job.json").read_bytes())
        sampler, moe = build_preset(config, device)
        resolved = {
            "preset": config.preset,
            "sampler": OmegaConf.to_container(OmegaConf.structured(sampler), resolve=True),
            "moe": OmegaConf.to_container(OmegaConf.structured(moe), resolve=True),
            "data": {
                "num_ligand_atoms": preparation.resolved_num_ligand_atoms,
                "protein_pocket_pdb_path": "prepared/pocket_pdb.pdb",
                "fragment_sdf_path": "prepared/fragment_sdf.sdf",
                "ligand_sdf_path": "prepared/reference_ligand_sdf.sdf",
            },
        }
        # Capture assets before model loading, so provenance describes what was loaded.
        checkpoints = checkpoint_metadata()
        environment = execution_environment(device)
        code = code_metadata()
        runtime = load_sampling_runtime(
            device=device,
            component_configs=list(moe.components.items()),
            global_scheduler_key=moe.global_scheduler_key,
            exponent_configs=moe.exponents,
            diffusion_scale=moe.diffusion_scale,
        )
        condition = SamplingCondition(
            protein_pocket_pdb_path=root / "prepared/pocket_pdb.pdb",
            fragment=Chem.SDMolSupplier(str(root / "prepared/fragment_sdf.sdf"), removeHs=False)[0],
            ref_ligand=Chem.SDMolSupplier(str(root / "prepared/reference_ligand_sdf.sdf"), removeHs=False)[0],
            num_ligand_atoms=preparation.resolved_num_ligand_atoms,
            condition_id=job.job_id,
        )
        phase = "sampling"
        write_json(root / "work/resolved_config.json", resolved)
        write_json(root / "work/phase.json", {"phase": phase})
        result = sample_condition(condition, runtime, sampler, save_dir=root / "work/diagnostics")
        if len(result.samples) != config.num_samples:
            raise ValueError("Sampler returned an unexpected particle count.")
        phase = "postprocessing"
        write_json(root / "work/phase.json", {"phase": phase})
        samples: list[Sample] = []
        files: list[StoredArtifact] = []
        for sample_id, molecule in enumerate(result.samples):
            sample, artifact = publish_sample(root, job.job_id, sample_id, molecule)
            samples.append(sample)
            if artifact is not None:
                files.append(artifact)
        phase = "writing_results"
        write_json(root / "work/phase.json", {"phase": phase})
        for sample_id, block in enumerate(result.xyz_blocks):
            relative = f"work/sample_{sample_id}.xyz"
            (root / relative).write_text(block, encoding="utf-8")
            files.append(describe_artifact(job.job_id, root, relative, "diagnostic", "text/plain"))
        manifest = Manifest.model_validate_json((root / "manifest.json").read_bytes())
        inputs = InputRefs(
            **{
                Path(file.relative_path).stem: file.artifact.reference()
                for file in manifest.files
                if file.artifact.role == "input_prepared"
            }
        )
        available = sum(sample.status == "available" for sample in samples)
        warnings = (
            ()
            if available
            else (WarningMessage(code="no_valid_samples", message="No generated samples passed validation."),)
        )
        response = InferenceResult(
            job_id=job.job_id,
            resolved_num_ligand_atoms=preparation.resolved_num_ligand_atoms,
            summary=Summary(requested=config.num_samples, available=available, invalid=config.num_samples - available),
            pocket_selection=preparation.pocket_selection,
            inputs=inputs,
            samples=tuple(samples),
            warnings=warnings,
        )
        write_json(root / "work/result.json", response)
        provenance = {
            "request": config.model_dump(mode="json"),
            "preparation": preparation.model_dump(mode="json"),
            "started_at": job.started_at,
            "finished_at": utc_now(),
            "code": code,
            "checkpoints": checkpoints,
            "environment": environment,
            "inputs": [file.artifact.model_dump(mode="json") for file in manifest.files],
            "summary": response.summary.model_dump(mode="json"),
        }
        write_json(root / "work/provenance.json", provenance)
        for name, role in (
            ("result.json", "result"),
            ("resolved_config.json", "resolved_config"),
            ("provenance.json", "provenance"),
        ):
            files.append(describe_artifact(job.job_id, root, f"work/{name}", role, "application/json"))
        for path in sorted((root / "work/diagnostics").glob("*.txt")):
            files.append(
                describe_artifact(job.job_id, root, path.relative_to(root).as_posix(), "diagnostic", "text/plain")
            )
        write_json(root / "work/outcome.json", WorkerSuccess(manifest=Manifest(files=tuple(files))))
    except Exception as error:
        logger.exception("Inference failed during %s", phase)
        # Avoid importing torch again if its initial import itself failed.
        torch_module = sys.modules.get("torch")
        if torch_module is not None and isinstance(error, torch_module.cuda.OutOfMemoryError):
            code, message = "gpu_out_of_memory", "The GPU ran out of memory."
        elif isinstance(error, OSError) and phase != "loading_models":
            code, message = "artifact_write_failed", "Required inference files could not be written."
        elif phase == "loading_models":
            code, message = "model_load_failed", "The inference models could not be loaded."
        elif phase == "writing_results":
            code, message = "artifact_write_failed", "Required inference results could not be stored."
        else:
            code, message = "inference_failed", "Inference could not complete."
        write_json(root / "work/outcome.json", WorkerFailure(error=Error(code=code, message=message)))


def publish_sample(
    root: Path, job_id: str, sample_id: int, molecule: "Mol | None"
) -> tuple[Sample, StoredArtifact | None]:
    import io

    import numpy as np
    from rdkit import Chem

    if molecule is None:
        return InvalidSample(
            sample_id=sample_id,
            error=Error(
                code="molecule_reconstruction_failed",
                message="The sampled coordinates could not be converted to a molecule.",
            ),
        ), None
    try:
        molecule = Chem.Mol(molecule)
        Chem.SanitizeMol(molecule)
        if molecule.GetNumAtoms() == 0 or not molecule.GetNumConformers():
            raise ValueError("Missing atoms or coordinates.")
        conformer = molecule.GetConformer()
        if not conformer.Is3D() or not np.isfinite(conformer.GetPositions()).all():
            raise ValueError("Invalid generated coordinates.")
        sdf = serialize_sdf(molecule)
        reread = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=False))
        if reread is None or reread.GetNumAtoms() != molecule.GetNumAtoms() or not reread.GetConformer().Is3D():
            raise ValueError("SDF serialization failed.")
        if not np.allclose(reread.GetConformer().GetPositions(), conformer.GetPositions(), rtol=0, atol=1e-12):
            raise ValueError("Serialization changed the generated coordinates.")
        smiles = Chem.MolToSmiles(reread, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError, StopIteration):
        return InvalidSample(
            sample_id=sample_id,
            error=Error(
                code="invalid_generated_molecule",
                message="The generated molecule did not pass chemical or coordinate validation.",
            ),
        ), None
    relative = f"work/sample_{sample_id}.sdf"
    # Storage errors here fail the job, never just the sample.
    (root / relative).write_bytes(sdf)
    artifact = describe_artifact(job_id, root, relative, "ligand", "chemical/x-mdl-sdfile")
    return AvailableSample(
        sample_id=sample_id,
        atom_count=reread.GetNumAtoms(),
        smiles=smiles,
        component_count=len(Chem.GetMolFrags(reread)),
        sdf=artifact.artifact.reference(),
    ), artifact


def main() -> None:
    root, device, parent_pid = sys.argv[1:]
    # Linux: stop a GPU worker if its API process dies even without a graceful shutdown.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not bind worker lifetime to the API process.")
    if os.getppid() != int(parent_pid):
        return
    logging.basicConfig(level=logging.INFO)
    job_root = Path(root)
    job = JOB_ADAPTER.validate_json((job_root / "job.json").read_bytes())
    if job.kind == "evaluation":
        from ace_backend.evaluation_worker import run_evaluation

        run_evaluation(job_root)
    else:
        run_job(job_root, device)


if __name__ == "__main__":
    main()
