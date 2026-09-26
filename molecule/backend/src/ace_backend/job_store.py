"""Persistent jobs and immutable published artifacts for one API serving process."""

import fcntl
import hashlib
import json
import os
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID, uuid4

from pydantic import BaseModel

from ace_backend.errors import APIError
from ace_backend.inputs import FILE_FIELDS, PreparedSubmission
from ace_backend.jobs_schema import (
    JOB_ADAPTER,
    Artifact,
    Error,
    FailedJob,
    InferenceResult,
    Job,
    JobLinks,
    Manifest,
    QueuedJob,
    RunningJob,
    StoredArtifact,
    SucceededJob,
)
from ace_backend.settings import Settings


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: BaseModel | dict[str, object]) -> None:
    text = (
        value.model_dump_json(indent=2)
        if isinstance(value, BaseModel)
        else json.dumps(value, indent=2, allow_nan=False)
    )
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(text + "\n")
            file.flush()
            os.fsync(file.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def describe_artifact(job_id: str, root: Path, relative_path: str, role: str, media_type: str) -> StoredArtifact:
    path = root / relative_path
    artifact_id = str(uuid4())
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    artifact = Artifact(
        artifact_id=artifact_id,
        url=f"/api/v1/jobs/{job_id}/artifacts/{artifact_id}",
        role=role,
        filename=path.name,
        media_type=media_type,
        size_bytes=path.stat().st_size,
        sha256=digest,
    )
    return StoredArtifact(artifact=artifact, relative_path=relative_path)


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.storage_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._owner = (self.root / ".owner.lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise RuntimeError("Job storage is already in use. Run one API worker per storage directory.") from None
        self._lock = threading.RLock()
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        try:
            for directory in self.root.iterdir():
                if directory.name.startswith(".submission-"):
                    shutil.rmtree(directory)
                elif directory.is_dir() and (directory / "job.json").exists():
                    job = JOB_ADAPTER.validate_json((directory / "job.json").read_bytes())
                    if directory.name != job.job_id:
                        raise ValueError("Stored job identifier does not match its directory.")
                    self._jobs[job.job_id] = job
            for job in tuple(self._jobs.values()):
                if isinstance(job, QueuedJob | RunningJob):
                    self.fail(
                        job.job_id,
                        Error(code="service_restarted", message="The service restarted before this job completed."),
                    )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._owner.close()

    def _save(self, job: Job) -> None:
        write_json(self.root / job.job_id / "job.json", job)
        self._jobs[job.job_id] = job

    def submit(self, prepared: PreparedSubmission) -> QueuedJob:
        with self._lock:
            active = sum(isinstance(job, QueuedJob | RunningJob) for job in self._jobs.values())
            # One slot is reserved for dispatch, including when max_pending_jobs is zero.
            if active >= 1 + self._settings.limits.max_pending_jobs:
                raise APIError(429, Error(code="queue_full", message="The job queue is full. Try again later."))
            job_id = str(uuid4())
            timestamp = utc_now()
            url = f"/api/v1/jobs/{job_id}"
            job = QueuedJob(
                job_id=job_id,
                created_at=timestamp,
                updated_at=timestamp,
                links=JobLinks(self=url, result=f"{url}/result", artifacts=f"{url}/artifacts"),
            )
            with TemporaryDirectory(prefix=".submission-", dir=self.root) as staging:
                root = Path(staging)
                (root / "work").mkdir()
                files: list[StoredArtifact] = []
                for folder, uploads in (("original", prepared.submission.uploads), ("prepared", prepared.prepared)):
                    (root / folder).mkdir()
                    for upload in uploads:
                        extension = FILE_FIELDS[upload.field]
                        relative = f"{folder}/{upload.field}{extension}"
                        (root / relative).write_bytes(upload.content)
                        files.append(
                            describe_artifact(
                                job_id,
                                root,
                                relative,
                                f"input_{folder}",
                                "chemical/x-pdb" if extension == ".pdb" else "chemical/x-mdl-sdfile",
                            )
                        )
                (root / "request.json").write_text(prepared.submission.config_json, encoding="utf-8")
                files.append(describe_artifact(job_id, root, "request.json", "request_config", "application/json"))
                write_json(root / "preparation.json", prepared.preparation)
                write_json(root / "limits.json", self._settings.limits)
                write_json(root / "manifest.json", Manifest(files=tuple(files)))
                write_json(root / "job.json", job)
                root.rename(self.root / job_id)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                canonical = str(UUID(job_id))
                if not (self.root / canonical / "job.json").is_file():
                    raise KeyError(canonical)
                return self._jobs[canonical]
            except (ValueError, KeyError) as error:
                raise APIError(404, Error(code="job_not_found", message="Job not found.")) from error

    def claim(self) -> RunningJob | None:
        with self._lock:
            if any(isinstance(job, RunningJob) for job in self._jobs.values()):
                return None
            queued = sorted(
                (job for job in self._jobs.values() if isinstance(job, QueuedJob)), key=lambda job: job.created_at
            )
            if not queued:
                return None
            job = queued[0]
            timestamp = utc_now()
            running = RunningJob(
                **job.model_dump(exclude={"status", "updated_at"}),
                updated_at=timestamp,
                started_at=timestamp,
                phase="loading_models",
            )
            self._save(running)
            return running

    def phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            job = self.get(job_id)
            if not isinstance(job, RunningJob):
                raise ValueError("Only a running job can change phase.")
            if job.phase != phase:
                self._save(
                    RunningJob(**job.model_dump(exclude={"phase", "updated_at"}), phase=phase, updated_at=utc_now())
                )

    def fail(self, job_id: str, error: Error) -> None:
        with self._lock:
            job = self.get(job_id)
            if not isinstance(job, QueuedJob | RunningJob):
                raise ValueError("A final job cannot change state.")
            timestamp = utc_now()
            self._save(
                FailedJob(
                    **job.model_dump(exclude={"status", "updated_at", "started_at", "phase", "progress"}),
                    updated_at=timestamp,
                    finished_at=timestamp,
                    started_at=job.started_at if isinstance(job, RunningJob) else None,
                    error=error,
                )
            )

    def succeed(self, job_id: str, manifest: Manifest) -> None:
        with self._lock:
            job = self.get(job_id)
            if not isinstance(job, RunningJob):
                raise ValueError("Only a running job can succeed.")
            root = self.root / job_id
            result = InferenceResult.model_validate_json((root / "work/result.json").read_bytes())
            published = self.manifest(job_id).files + manifest.files
            references = {file.artifact.artifact_id for file in published}
            for reference in [
                *result.inputs.model_dump().values(),
                *(sample.sdf.model_dump() for sample in result.samples if sample.status == "available"),
            ]:
                if reference["artifact_id"] not in references:
                    raise ValueError("Result refers to an unpublished artifact.")
            for file in published:
                path = root / file.relative_path
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if path.stat().st_size != file.artifact.size_bytes or digest != file.artifact.sha256:
                    raise OSError("Artifact contents changed before publication.")
            write_json(root / "manifest.json", Manifest(files=published))
            timestamp = utc_now()
            self._save(
                SucceededJob(
                    **job.model_dump(exclude={"status", "updated_at", "phase", "progress"}),
                    updated_at=timestamp,
                    finished_at=timestamp,
                )
            )

    def manifest(self, job_id: str) -> Manifest:
        job = self.get(job_id)
        return Manifest.model_validate_json((self.root / job.job_id / "manifest.json").read_bytes())

    def result(self, job_id: str) -> InferenceResult:
        job = self.get(job_id)
        if isinstance(job, FailedJob):
            raise APIError(409, Error(code="job_failed", message="This job failed; no completed result is available."))
        if not isinstance(job, SucceededJob):
            raise APIError(409, Error(code="result_not_ready", message="This job has not completed."))
        return InferenceResult.model_validate_json((self.root / job.job_id / "work/result.json").read_bytes())

    def artifact(self, job_id: str, artifact_id: str) -> tuple[Path, Artifact]:
        job = self.get(job_id)
        for file in self.manifest(job.job_id).files:
            if file.artifact.artifact_id == artifact_id:
                path = self.root / job.job_id / file.relative_path
                if path.is_file():
                    return path, file.artifact
                break
        raise APIError(404, Error(code="artifact_not_found", message="Artifact not found for this job."))
