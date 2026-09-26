"""Validate evaluation uploads or snapshot a completed inference job before acceptance."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Request
from pydantic import ValidationError

from ace_backend.errors import APIError, invalid_input
from ace_backend.evaluation_schema import (
    EvaluationConfig,
    EvaluationInput,
    EvaluationPreparation,
    InferenceJobSource,
    SourceArtifact,
)
from ace_backend.inputs import FILE_FIELDS, InputFile, Upload, parse_pocket, read_multipart, unique_json_object
from ace_backend.jobs_schema import Error, ErrorDetail, InferenceResult
from ace_backend.molecule_io import serialize_sdf
from ace_backend.schemas import OperationalLimits

if TYPE_CHECKING:
    from rdkit.Chem import Mol

    from ace_backend.job_store import JobStore

EVALUATION_FILE_FIELDS = {"ligand_sdf": ".sdf", **FILE_FIELDS}


@dataclass(frozen=True)
class EvaluationSubmission:
    config: EvaluationConfig
    config_json: str
    uploads: tuple[Upload, ...]


@dataclass(frozen=True)
class PreparedEvaluation:
    config_json: str
    files: tuple[InputFile, ...]
    preparation: EvaluationPreparation


def parse_evaluation_config(raw: str, limits: OperationalLimits) -> EvaluationConfig:
    try:
        json.loads(raw, object_pairs_hook=unique_json_object)
    except (ValueError, RecursionError) as error:
        raise invalid_input("config", "invalid_json", "Config must contain valid JSON.", 400) from error
    try:
        # JSON arrays become immutable tuples without accepting coerced scalar values.
        config = EvaluationConfig.model_validate_json(raw)
    except ValidationError as error:
        details = tuple(
            ErrorDetail(
                field=".".join(("config", *(str(part) for part in entry["loc"]))),
                code=entry["type"],
                message=entry["msg"],
            )
            for entry in error.errors(include_input=False, include_context=False, include_url=False)
        )
        raise APIError(
            422, Error(code="validation_error", message="Invalid evaluation config.", details=details)
        ) from error
    if isinstance(config.source, InferenceJobSource) and len(config.source.sample_ids) > limits.max_num_samples:
        raise invalid_input("config.source.sample_ids", "limit_exceeded", "Too many selected samples.")
    return config


async def read_evaluation(request: Request, limits: OperationalLimits) -> EvaluationSubmission:
    form = await read_multipart(request, limits, EVALUATION_FILE_FIELDS)
    config = parse_evaluation_config(form.config_json, limits)
    required: set[str] = set()
    if config.source.type == "upload":
        required.add("ligand_sdf")
        if "scaffold_preservation" in config.metrics:
            required.add("fragment_sdf")
        if "docking" in config.metrics:
            required.update(("pocket_pdb", "reference_ligand_sdf"))
    actual = {upload.field for upload in form.uploads}
    for field in sorted(actual - required):
        raise invalid_input(field, "unexpected_field", "This file is not used by the requested evaluation.")
    for field in sorted(required - actual):
        raise invalid_input(field, "missing_field", "Required evaluation file is missing.")
    return EvaluationSubmission(config, form.config_json, form.uploads)


def parse_evaluation_molecule(
    content: bytes, field: str, limits: OperationalLimits, *, require_3d: bool
) -> "Mol | None":
    """Return a normalized copy, or None for a parseable unsanitizable ligand."""
    import numpy as np
    from rdkit import Chem

    try:
        text = content.decode("utf-8")
        records = text.split("$$$$")
        if len(records) > 2 or (len(records) == 2 and records[1].strip()):
            raise invalid_input(field, "multiple_records", "Expected exactly one SDF molecule record.")
        molecule = Chem.MolFromMolBlock(records[0], sanitize=False, removeHs=False, strictParsing=True)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise invalid_input(field, "invalid_structure", "The SDF must contain a nonempty molecule.")
        if field == "ligand_sdf" and molecule.GetNumAtoms() > limits.max_num_ligand_atoms:
            raise invalid_input(field, "limit_exceeded", "The evaluation molecule has too many atoms.")
        if require_3d and (
            molecule.GetNumConformers() != 1
            or not molecule.GetConformer().Is3D()
            or (len(text.splitlines()) > 1 and text.splitlines()[1][20:22] == "2D")
        ):
            raise invalid_input(field, "coordinates_required", "A finite 3D conformer is required for docking.")
        if molecule.GetNumConformers() and not np.isfinite(molecule.GetConformer().GetPositions()).all():
            raise invalid_input(field, "coordinates_required", "Finite coordinates are required.")
        sanitized = Chem.Mol(molecule)
        try:
            Chem.SanitizeMol(sanitized)
        except (ValueError, RuntimeError):
            if field != "ligand_sdf":
                raise invalid_input(field, "invalid_structure", "The molecule could not be sanitized.") from None
            # An unsanitizable ligand is an evaluation outcome, not a parse failure.
            return None
        else:
            if field != "ligand_sdf" and len(Chem.GetMolFrags(sanitized)) != 1:
                raise invalid_input(field, "invalid_structure", "The input molecule must have one connected component.")
            if field == "fragment_sdf":
                sanitized = Chem.RemoveHs(sanitized)
        if sanitized.GetNumAtoms() > limits.max_num_ligand_atoms:
            raise invalid_input(field, "limit_exceeded", "The evaluation molecule has too many atoms.")
        return sanitized
    except (UnicodeError, ValueError, RuntimeError) as error:
        raise invalid_input(field, "invalid_structure", "The molecule could not be parsed.") from error


def prepare_evaluation(
    submission: EvaluationSubmission, limits: OperationalLimits, store: "JobStore"
) -> PreparedEvaluation:
    from rdkit import rdBase

    if isinstance(submission.config.source, InferenceJobSource):
        return _snapshot_inference(submission, limits, store)
    docking = "docking" in submission.config.metrics
    files: list[InputFile] = []
    paths: dict[str, str] = {}
    for upload in submission.uploads:
        normalized = upload.content
        if upload.field == "pocket_pdb":
            from Bio.PDB.Polypeptide import is_aa

            model, _ = parse_pocket(upload.content, limits)
            if not any(is_aa(residue, standard=True) for residue in model.get_residues()):
                raise invalid_input(
                    upload.field, "empty_pocket", "The receptor must contain standard amino acid residues."
                )
        else:
            molecule = parse_evaluation_molecule(
                upload.content, upload.field, limits, require_3d=docking and upload.field != "fragment_sdf"
            )
            if molecule is not None:
                normalized = serialize_sdf(molecule)
        extension = EVALUATION_FILE_FIELDS[upload.field]
        for folder in ("original", "prepared"):
            relative = f"{folder}/{upload.field}{extension}"
            files.append(
                InputFile(
                    relative,
                    upload.content if folder == "original" else normalized,
                    f"input_{folder}",
                    "chemical/x-pdb" if extension == ".pdb" else "chemical/x-mdl-sdfile",
                )
            )
        paths[upload.field] = f"prepared/{upload.field}{extension}"
    return PreparedEvaluation(
        submission.config_json,
        tuple(files),
        EvaluationPreparation(
            samples=(EvaluationInput(sample_id=0, ligand_sdf=paths["ligand_sdf"]),),
            fragment_sdf=paths.get("fragment_sdf"),
            pocket_pdb=paths.get("pocket_pdb"),
            reference_ligand_sdf=paths.get("reference_ligand_sdf"),
            rdkit_version=rdBase.rdkitVersion,
        ),
    )


def _snapshot_inference(
    submission: EvaluationSubmission, limits: OperationalLimits, store: "JobStore"
) -> PreparedEvaluation:
    from rdkit import rdBase

    source = submission.config.source
    assert isinstance(source, InferenceJobSource)
    job = store.get(source.job_id)
    if job.kind != "inference":
        raise invalid_input("config.source.job_id", "invalid_job_kind", "Expected an inference job.")
    if job.status != "succeeded":
        raise APIError(
            409, Error(code="source_job_not_succeeded", message="The source inference job has not succeeded.")
        )
    result = store.result(job.job_id)
    assert isinstance(result, InferenceResult)
    available = {sample.sample_id: sample for sample in result.samples if sample.status == "available"}
    if any(sample_id not in available for sample_id in source.sample_ids):
        raise invalid_input(
            "config.source.sample_ids", "invalid_sample", "Select only available samples from the source job."
        )
    selected = {available[sample_id].sdf.artifact_id for sample_id in source.sample_ids}
    files: list[InputFile] = []
    copied: dict[str, str] = {}
    origins: list[SourceArtifact] = []
    required_inputs: dict[str, str] = {}
    if "scaffold_preservation" in submission.config.metrics:
        required_inputs[result.inputs.fragment_sdf.artifact_id] = "fragment_sdf"
    if "docking" in submission.config.metrics:
        required_inputs[result.inputs.reference_ligand_sdf.artifact_id] = "reference_ligand_sdf"
        required_inputs[result.inputs.pocket_pdb.artifact_id] = "pocket_pdb"
    for entry in store.manifest(job.job_id).files:
        artifact = entry.artifact
        if artifact.artifact_id not in selected and artifact.role not in {
            "input_original",
            "input_prepared",
            "request_config",
            "resolved_config",
            "provenance",
        }:
            continue
        path, _ = store.artifact(job.job_id, artifact.artifact_id)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise OSError("Source artifact changed after publication.")
        relative = f"source/{artifact.artifact_id}/{Path(entry.relative_path).name}"
        files.append(InputFile(relative, content, artifact.role, artifact.media_type))
        copied[artifact.artifact_id] = relative
        origins.append(SourceArtifact(artifact_id=artifact.artifact_id, relative_path=relative, sha256=artifact.sha256))
        if artifact.artifact_id in selected:
            parse_evaluation_molecule(content, "ligand_sdf", limits, require_3d="docking" in submission.config.metrics)
        elif artifact.artifact_id in required_inputs:
            field = required_inputs[artifact.artifact_id]
            if field == "pocket_pdb":
                parse_pocket(content, limits)
            else:
                parse_evaluation_molecule(content, field, limits, require_3d=field == "reference_ligand_sdf")
    preparation = EvaluationPreparation(
        samples=tuple(
            EvaluationInput(sample_id=sample_id, ligand_sdf=copied[available[sample_id].sdf.artifact_id])
            for sample_id in sorted(source.sample_ids)
        ),
        fragment_sdf=copied[result.inputs.fragment_sdf.artifact_id],
        pocket_pdb=copied[result.inputs.pocket_pdb.artifact_id],
        reference_ligand_sdf=copied[result.inputs.reference_ligand_sdf.artifact_id],
        rdkit_version=rdBase.rdkitVersion,
        source_artifacts=tuple(origins),
    )
    return PreparedEvaluation(submission.config_json, tuple(files), preparation)
