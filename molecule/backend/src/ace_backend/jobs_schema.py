"""Strict request, job-state, and inference-result types for API v1."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class ACEParameters(FrozenModel):
    omega: float = Field(ge=0, le=10)
    diffusion_scale: float = Field(gt=0, le=10)
    b1: float = Field(ge=0, le=100)
    b2: float = Field(ge=0, le=10)


class InferenceConfig(FrozenModel):
    preset: Literal["ace_scaffold_v1"]
    num_samples: int = Field(ge=1, le=16)
    seed: int = Field(ge=0, le=4294967295)
    num_sampling_steps: int = Field(ge=10, le=2000)
    num_ligand_atoms: Annotated[int, Field(ge=1, le=128)] | None
    ace: ACEParameters


CONFIG_EXAMPLE = {
    "preset": "ace_scaffold_v1",
    "num_samples": 2,
    "seed": 42,
    "num_sampling_steps": 500,
    "num_ligand_atoms": None,
    "ace": {"omega": 1.4, "diffusion_scale": 2.0, "b1": 30.0, "b2": 0.336},
}


class ErrorDetail(FrozenModel):
    field: str
    code: str
    message: str


class Error(FrozenModel):
    code: str
    message: str
    details: tuple[ErrorDetail, ...] = ()


class ErrorResponse(FrozenModel):
    request_id: str
    error: Error


class JobLinks(FrozenModel):
    self: str
    result: str
    artifacts: str


class JobBase(FrozenModel):
    job_id: str
    kind: Literal["inference"] = "inference"
    created_at: str
    updated_at: str
    links: JobLinks


class QueuedJob(JobBase):
    status: Literal["queued"] = "queued"


InferencePhase = Literal["loading_models", "sampling", "postprocessing", "writing_results"]


class RunningJob(JobBase):
    status: Literal["running"] = "running"
    started_at: str
    phase: InferencePhase
    # The existing sampler does not expose a progress callback.
    progress: None = None


class SucceededJob(JobBase):
    status: Literal["succeeded"] = "succeeded"
    started_at: str
    finished_at: str


class FailedJob(JobBase):
    status: Literal["failed"] = "failed"
    started_at: str | None
    finished_at: str
    error: Error


Job = Annotated[QueuedJob | RunningJob | SucceededJob | FailedJob, Field(discriminator="status")]
JOB_ADAPTER: TypeAdapter[Job] = TypeAdapter(Job)


class ArtifactRef(FrozenModel):
    artifact_id: str
    url: str


class Artifact(ArtifactRef):
    role: Literal[
        "input_original",
        "input_prepared",
        "ligand",
        "preview",
        "request_config",
        "resolved_config",
        "provenance",
        "result",
        "diagnostic",
    ]
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def reference(self) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.artifact_id, url=self.url)


class StoredArtifact(FrozenModel):
    artifact: Artifact
    relative_path: str


class Manifest(FrozenModel):
    files: tuple[StoredArtifact, ...]


class ArtifactList(FrozenModel):
    job_id: str
    artifacts: tuple[Artifact, ...]


class ResidueId(FrozenModel):
    chain_id: str
    residue_number: int
    insertion_code: str


class PocketSelection(FrozenModel):
    cutoff_angstrom: Literal[8.0] = 8.0
    residues: tuple[ResidueId, ...]


class InputRefs(FrozenModel):
    pocket_pdb: ArtifactRef
    fragment_sdf: ArtifactRef
    reference_ligand_sdf: ArtifactRef


class Preparation(FrozenModel):
    resolved_num_ligand_atoms: int
    fragment_atom_count: int
    reference_atom_count: int
    pocket_atom_count: int
    pocket_selection: PocketSelection
    normalization_policy: str = "RDKit sanitization followed by default RemoveHs; no alignment or coordinate generation"
    rdkit_version: str


class AvailableSample(FrozenModel):
    sample_id: int = Field(ge=0)
    status: Literal["available"] = "available"
    atom_count: int = Field(gt=0)
    smiles: str
    component_count: int = Field(gt=0)
    sdf: ArtifactRef
    preview_png: ArtifactRef | None = None


class InvalidSample(FrozenModel):
    sample_id: int = Field(ge=0)
    status: Literal["invalid"] = "invalid"
    error: Error


Sample = Annotated[AvailableSample | InvalidSample, Field(discriminator="status")]


class Summary(FrozenModel):
    requested: int
    available: int
    invalid: int


class WarningMessage(FrozenModel):
    code: str
    message: str


class InferenceResult(FrozenModel):
    job_id: str
    kind: Literal["inference"] = "inference"
    resolved_num_ligand_atoms: int
    summary: Summary
    pocket_selection: PocketSelection
    inputs: InputRefs
    samples: tuple[Sample, ...]
    warnings: tuple[WarningMessage, ...]


class WorkerSuccess(FrozenModel):
    status: Literal["succeeded"] = "succeeded"
    manifest: Manifest


class WorkerFailure(FrozenModel):
    status: Literal["failed"] = "failed"
    error: Error


WORKER_OUTCOME_ADAPTER = TypeAdapter(Annotated[WorkerSuccess | WorkerFailure, Field(discriminator="status")])
