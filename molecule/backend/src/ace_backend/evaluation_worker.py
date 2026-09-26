"""CPU evaluation subprocess: immutable inputs, independent reports, durable outputs."""

import hashlib
import importlib.metadata
import logging
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ace_backend.evaluation_runtime import evaluate_molecule, has_evaluation_values
from ace_backend.evaluation_schema import EvaluationConfig, EvaluationPreparation, EvaluationResult, EvaluationSample
from ace_backend.job_store import describe_artifact, utc_now, write_json
from ace_backend.jobs_schema import JOB_ADAPTER, Error, Manifest, WarningMessage, WorkerFailure, WorkerSuccess
from ace_backend.provenance import code_metadata

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


def _load_molecule(root: Path, relative: str) -> "Mol":
    from rdkit import Chem

    molecule = Chem.MolFromMolBlock(
        (root / relative).read_text(encoding="utf-8").split("$$$$")[0],
        sanitize=False,
        removeHs=False,
        strictParsing=True,
    )
    if molecule is None:
        raise ValueError("A persisted evaluation input is no longer parseable.")
    return molecule


def _environment(docking: bool) -> dict[str, object]:
    versions = {"python": platform.python_version()}
    for name in ("rdkit", "numpy"):
        versions[name] = importlib.metadata.version(name)
    tools: dict[str, object] = {}
    if docking:
        from evaluation.backends.qvina import PATH_QVINA2

        versions["biopython"] = importlib.metadata.version("biopython")
        for name, command in (("openbabel", ("obabel", "-V")), ("quickvina", (str(PATH_QVINA2), "--version"))):
            try:
                completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
                versions[name] = completed.stdout.strip().splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                versions[name] = "unavailable"
        if PATH_QVINA2.is_file():
            with PATH_QVINA2.open("rb") as stream:
                tools["quickvina"] = {
                    "identifier": PATH_QVINA2.name,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
    return {"device": "cpu", "dependencies": versions, "tools": tools}


def run_evaluation(root: Path) -> None:
    phase = "evaluating"
    try:
        from rdkit import Chem

        config = EvaluationConfig.model_validate_json((root / "request.json").read_bytes())
        preparation = EvaluationPreparation.model_validate_json((root / "preparation.json").read_bytes())
        job = JOB_ADAPTER.validate_json((root / "job.json").read_bytes())
        manifest = Manifest.model_validate_json((root / "manifest.json").read_bytes())
        fragment = (
            _load_molecule(root, preparation.fragment_sdf)
            if "scaffold_preservation" in config.metrics and preparation.fragment_sdf
            else None
        )
        reference = (
            _load_molecule(root, preparation.reference_ligand_sdf)
            if config.docking and preparation.reference_ligand_sdf
            else None
        )
        pocket = root / preparation.pocket_pdb if config.docking and preparation.pocket_pdb else None
        docking_details = None
        if config.docking is not None:
            from evaluation.backends.qvina import box_from_mol

            assert reference is not None, "Validated docking reference is required."
            Chem.SanitizeMol(reference)
            center, size = box_from_mol(reference, pad=config.docking.padding_angstrom)
            docking_details = {
                **config.docking.model_dump(mode="json"),
                # Match the three-decimal arguments sent by the existing QuickVina backend.
                "box_center_angstrom": [float(f"{value:.3f}") for value in center],
                "box_size_angstrom": [float(f"{value:.3f}") for value in size],
                "receptor_preparation": "Open Babel CLI, rigid receptor (-xr), protonation at pH 7.4 (-p 7.4)",
                "receptor_ph": 7.4,
                "external_command_timeout_seconds": 60,
                "mode": "redocking",
                "pocket_selection": "entire supplied pocket; no inference residue reselection",
            }
        resolved = {
            "config": config.model_dump(mode="json"),
            "inputs": preparation.model_dump(mode="json"),
            "docking": docking_details,
        }
        environment = _environment(config.docking is not None)
        code = code_metadata()
        write_json(root / "work/resolved_config.json", resolved)
        write_json(root / "work/phase.json", {"phase": phase})
        samples = tuple(
            EvaluationSample(
                sample_id=sample.sample_id,
                metrics=evaluate_molecule(
                    _load_molecule(root, sample.ligand_sdf),
                    config,
                    fragment=fragment,
                    pocket=pocket,
                    reference=reference,
                ),
            )
            for sample in preparation.samples
        )
        warnings = (
            ()
            if any(has_evaluation_values(sample.metrics) for sample in samples)
            else (WarningMessage(code="no_evaluation_values", message="Every requested metric was failed or skipped."),)
        )
        response = EvaluationResult(
            job_id=job.job_id, source=config.source, metrics=config.metrics, samples=samples, warnings=warnings
        )
        phase = "writing_results"
        write_json(root / "work/phase.json", {"phase": phase})
        write_json(root / "work/result.json", response)
        write_json(
            root / "work/provenance.json",
            {
                "request": config.model_dump(mode="json"),
                "preparation": preparation.model_dump(mode="json"),
                "started_at": job.started_at,
                "finished_at": utc_now(),
                "code": code,
                "environment": environment,
                "inputs": [file.artifact.model_dump(mode="json") for file in manifest.files],
                "docking": docking_details,
                "results": response.model_dump(mode="json"),
            },
        )
        files = tuple(
            describe_artifact(job.job_id, root, f"work/{name}.json", name, "application/json")
            for name in ("result", "resolved_config", "provenance")
        )
        write_json(root / "work/outcome.json", WorkerSuccess(manifest=Manifest(files=files)))
    except Exception as error:
        logger.exception("Evaluation failed during %s", phase)
        code = (
            "artifact_write_failed" if isinstance(error, OSError) or phase == "writing_results" else "evaluation_failed"
        )
        write_json(
            root / "work/outcome.json",
            WorkerFailure(
                error=Error(
                    code=code,
                    message="Evaluation results could not be stored."
                    if code == "artifact_write_failed"
                    else "The evaluation could not complete.",
                )
            ),
        )
