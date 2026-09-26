"""Bounded multipart parsing and scientific input preparation, before acceptance."""

import io
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

from fastapi import Request
from pydantic import ValidationError
from python_multipart.multipart import parse_options_header
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ace_backend.errors import APIError, invalid_input
from ace_backend.jobs_schema import Error, ErrorDetail, InferenceConfig, PocketSelection, Preparation, ResidueId
from ace_backend.molecule_io import serialize_sdf
from ace_backend.schemas import OperationalLimits

if TYPE_CHECKING:
    from rdkit.Chem import Mol

FILE_FIELDS = {"pocket_pdb": ".pdb", "fragment_sdf": ".sdf", "reference_ligand_sdf": ".sdf"}


@dataclass(frozen=True)
class Upload:
    field: str
    content: bytes


@dataclass(frozen=True)
class Submission:
    config: InferenceConfig
    config_json: str
    uploads: tuple[Upload, ...]


@dataclass(frozen=True)
class PreparedSubmission:
    submission: Submission
    prepared: tuple[Upload, ...]
    preparation: Preparation


class InferenceMultipartParser(MultiPartParser):
    """Add byte limits, strict UTF-8 fields, and an end-boundary check to Starlette."""

    def __init__(self, request: Request, limits: OperationalLimits) -> None:
        super().__init__(request.headers, _bounded_stream(request, limits.max_request_bytes), max_files=4, max_fields=2)
        self.limits = limits
        self.complete = False
        self.part_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        field = self._current_part.field_name
        if field in FILE_FIELDS and self._current_part.file is None:
            raise invalid_input(field, "invalid_type", "Expected a file upload.")
        if field == "config" and self._current_part.file is not None:
            raise invalid_input(field, "invalid_type", "Config must be a regular JSON form field, not a file.")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.part_bytes += end - start
        limit = self.limits.max_file_bytes if self._current_part.file else self.limits.max_config_bytes
        if self.part_bytes > limit:
            raise invalid_input(self._current_part.field_name, "limit_exceeded", "The form part is too large.", 413)
        super().on_part_data(data, start, end)

    def on_part_end(self) -> None:
        if self._current_part.file is None:
            try:
                value = self._current_part.data.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise invalid_input("config", "invalid_json", "Config must contain UTF-8 JSON.", 400) from error
            self.items.append((self._current_part.field_name, value))
        else:
            super().on_part_end()

    def on_end(self) -> None:
        self.complete = True


async def _bounded_stream(request: Request, limit: int) -> AsyncGenerator[bytes, None]:
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise invalid_input("request", "limit_exceeded", "The whole request is too large.", 413)
        yield chunk


async def read_submission(request: Request, limits: OperationalLimits) -> Submission:
    media_type, _ = parse_options_header(request.headers.get("content-type", ""))
    if media_type.lower() != b"multipart/form-data":
        raise invalid_input("request", "unsupported_media_type", "Expected multipart/form-data.", 415)
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
            if size < 0:
                raise ValueError
        except ValueError as error:
            raise invalid_input("request", "invalid_length", "Invalid Content-Length.", 400) from error
        if size > limits.max_request_bytes:
            raise invalid_input("request", "limit_exceeded", "The whole request is too large.", 413)

    parser = InferenceMultipartParser(request, limits)
    try:
        form = await parser.parse()
    except MultiPartException as error:
        excessive_parts = error.message.startswith("Too many")
        raise invalid_input(
            "request",
            "invalid_parts" if excessive_parts else "invalid_multipart",
            "Unexpected form parts." if excessive_parts else "Malformed multipart data.",
            422 if excessive_parts else 400,
        ) from error
    except BaseException:
        # Older supported Starlette releases only close uploads for their own
        # parser errors, whereas byte-limit/UTF-8 errors above are API errors.
        for file in parser._files_to_close_on_error:
            file.close()
        raise
    try:
        if not parser.complete:
            raise invalid_input("request", "invalid_multipart", "The multipart body is incomplete.", 400)
        _validate_parts(form)
        raw_config = form["config"]
        if not isinstance(raw_config, str):
            raise invalid_input("config", "invalid_type", "Config must be a regular JSON form field, not a file.")
        config = parse_config(raw_config, limits)
        uploads: list[Upload] = []
        for field, extension in FILE_FIELDS.items():
            upload = form[field]
            if not isinstance(upload, UploadFile):
                raise invalid_input(field, "invalid_type", "Expected a file upload.")
            if PureWindowsPath(upload.filename or "").suffix.lower() != extension:
                raise invalid_input(field, "unsupported_extension", f"Expected a {extension} file.", 415)
            uploads.append(Upload(field, await upload.read()))
        return Submission(config, raw_config, tuple(uploads))
    finally:
        await form.close()


def _validate_parts(form: FormData) -> None:
    required = {*FILE_FIELDS, "config"}
    seen: set[str] = set()
    for field, _ in form.multi_items():
        if field not in required:
            raise invalid_input(field, "unknown_field", "Unknown form field.")
        if field in seen:
            raise invalid_input(field, "duplicate_field", "Each form field must occur exactly once.")
        seen.add(field)
    for field in sorted(required - seen):
        raise invalid_input(field, "missing_field", "Required form field is missing.")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise invalid_input("config", "duplicate_field", "Duplicate JSON fields are not allowed.")
        result[key] = value
    return result


def parse_config(raw: str, limits: OperationalLimits) -> InferenceConfig:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (ValueError, RecursionError) as error:
        raise invalid_input("config", "invalid_json", "Config must contain valid JSON.", 400) from error
    try:
        config = InferenceConfig.model_validate(value)
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
            422, Error(code="validation_error", message="Invalid inference config.", details=details)
        ) from error
    for field, limit in (
        ("num_samples", limits.max_num_samples),
        ("num_sampling_steps", limits.max_num_sampling_steps),
        ("num_ligand_atoms", limits.max_num_ligand_atoms),
    ):
        value = getattr(config, field)
        if value is not None and value > limit:
            raise invalid_input(f"config.{field}", "limit_exceeded", f"Expected no more than {limit}.")
    return config


def prepare_submission(submission: Submission, limits: OperationalLimits) -> PreparedSubmission:
    # Keep scientific imports off the API import/startup path when inference is unavailable.
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionException
    from diffsbdd.utils import get_pocket_from_ligand
    from rdkit import rdBase

    contents = {upload.field: upload.content for upload in submission.uploads}
    fragment = _prepare_molecule(contents["fragment_sdf"], "fragment_sdf", limits)
    reference = _prepare_molecule(contents["reference_ligand_sdf"], "reference_ligand_sdf", limits)
    count = submission.config.num_ligand_atoms or reference.GetNumAtoms()
    if count < fragment.GetNumAtoms():
        raise invalid_input(
            "config.num_ligand_atoms",
            "atom_count_too_small",
            "Ligand must have at least as many atoms as the fragment.",
        )
    try:
        pdb_text = contents["pocket_pdb"].decode("utf-8")
        if sum(line.startswith("MODEL ") for line in pdb_text.splitlines()) > 1:
            raise invalid_input("pocket_pdb", "multiple_models", "Expected a single protein model.")
        atom_lines = [line for line in pdb_text.splitlines() if line[:6] in ("ATOM  ", "HETATM")]
        if len(atom_lines) > limits.max_pocket_atoms:
            raise invalid_input("pocket_pdb", "limit_exceeded", "The input PDB has too many atoms.")
        # Validate all records, including alternate locations the parser might not select.
        if not atom_lines or not all(
            np.isfinite([float(line[start : start + 8]) for start in (30, 38, 46)]).all() for line in atom_lines
        ):
            raise invalid_input("pocket_pdb", "coordinates_required", "Finite protein coordinates are required.")
        structure = PDBParser(PERMISSIVE=False, QUIET=True).get_structure("pocket", io.StringIO(pdb_text))
        models = list(structure.get_models())
        if len(models) != 1:
            raise invalid_input("pocket_pdb", "multiple_models", "Expected a single protein model.")
    except (UnicodeError, ValueError, PDBConstructionException, IndexError) as error:
        raise invalid_input("pocket_pdb", "invalid_structure", "The PDB could not be parsed.") from error
    residues = get_pocket_from_ligand(models[0], reference, dist_cutoff=8.0)
    if not residues:
        raise invalid_input(
            "pocket_pdb", "empty_pocket", "No standard amino acid residues are within 8 angstrom of the reference."
        )
    selection = PocketSelection(
        residues=tuple(
            ResidueId(
                chain_id=residue.get_parent().id.strip(),
                residue_number=residue.id[1],
                insertion_code=residue.id[2].strip(),
            )
            for residue in residues
        )
    )
    prepared = (
        Upload("pocket_pdb", contents["pocket_pdb"]),
        Upload("fragment_sdf", serialize_sdf(fragment)),
        Upload("reference_ligand_sdf", serialize_sdf(reference)),
    )
    return PreparedSubmission(
        submission,
        prepared,
        Preparation(
            resolved_num_ligand_atoms=count,
            fragment_atom_count=fragment.GetNumAtoms(),
            reference_atom_count=reference.GetNumAtoms(),
            pocket_atom_count=len(atom_lines),
            pocket_selection=selection,
            rdkit_version=rdBase.rdkitVersion,
        ),
    )


def _prepare_molecule(content: bytes, field: str, limits: OperationalLimits) -> "Mol":
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
        Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise invalid_input(field, "invalid_structure", "The input molecule must have one connected component.")
        if (
            molecule.GetNumConformers() != 1
            or not molecule.GetConformer().Is3D()
            or not np.isfinite(molecule.GetConformer().GetPositions()).all()
            or (len(text.splitlines()) > 1 and text.splitlines()[1][20:22] == "2D")
        ):
            raise invalid_input(field, "coordinates_required", "A finite 3D conformer is required.")
        molecule = Chem.RemoveHs(molecule)
        if field == "fragment_sdf" and any(
            atom.GetSymbol() not in {"H", "C", "N", "O", "F"} or atom.HasQuery() for atom in molecule.GetAtoms()
        ):
            raise invalid_input(
                field,
                "unsupported_fragment_atom",
                "Fragment atoms must belong to H, C, N, O, F and cannot be query atoms.",
            )
        if molecule.GetNumAtoms() > limits.max_num_ligand_atoms:
            raise invalid_input(field, "limit_exceeded", "The normalized molecule has too many atoms.")
        return molecule
    except (UnicodeError, ValueError, RuntimeError) as error:
        raise invalid_input(field, "invalid_structure", "The molecule could not be parsed or sanitized.") from error
