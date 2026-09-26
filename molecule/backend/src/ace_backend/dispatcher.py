"""One durable queue consumer; each job runs in a fresh, terminable process."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

from ace_backend.evaluation_schema import EvaluationConfig
from ace_backend.job_store import JobStore
from ace_backend.jobs_schema import INFERENCE_CONFIG_ADAPTER, WORKER_OUTCOME_ADAPTER, Error, WorkerFailure
from ace_backend.settings import Settings

logger = logging.getLogger(__name__)


def worker_command(root: Path, device: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "ace_backend.worker", str(root), device, str(os.getpid()))


class Dispatcher:
    def __init__(self, store: JobStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.wake = asyncio.Event()

    async def run(self) -> None:
        while True:
            self.wake.clear()
            job = self.store.claim()
            if job is None:
                # Also discover a persisted submission if its HTTP task disconnected
                # just before it could notify this consumer.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                continue
            try:
                await self._execute(job.job_id)
            except TimeoutError:
                self.store.fail(
                    job.job_id, Error(code="job_timeout", message="The job exceeded its execution time limit.")
                )
            except Exception:
                logger.exception("Worker failed for job %s", job.job_id)
                self.store.fail(
                    job.job_id,
                    Error(code=f"{job.kind}_failed", message="The worker could not complete the job."),
                )

    async def _execute(self, job_id: str) -> None:
        root = self.store.root / job_id
        job = self.store.get(job_id)
        raw = (root / "request.json").read_bytes()
        environment = dict(os.environ)
        if job.kind == "inference":
            config = INFERENCE_CONFIG_ADAPTER.validate_json(raw)
            environment.update(PYTHONHASHSEED=str(config.seed), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        else:
            evaluation = EvaluationConfig.model_validate_json(raw)
            environment["PYTHONHASHSEED"] = str(evaluation.docking.seed if evaluation.docking else 0)
        # Keep native-library errors and private paths in an unpublished operator log.
        with (root / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *worker_command(root, self.settings.device),
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            completion = asyncio.create_task(process.wait())
            try:
                async with asyncio.timeout(self.settings.limits.job_timeout_seconds):
                    while not completion.done():
                        try:
                            await asyncio.wait_for(asyncio.shield(completion), timeout=0.1)
                        except TimeoutError:
                            pass
                        phase_path = root / "work/phase.json"
                        if phase_path.exists():
                            phase = json.loads(phase_path.read_text())["phase"]
                            self.store.phase(job_id, phase)
            finally:
                if process.returncode is None:
                    # Stop model workers and any external docking processes before the next job.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                await completion
        if process.returncode != 0:
            self.store.fail(job_id, Error(code=f"{job.kind}_failed", message="The worker exited unexpectedly."))
            return
        outcome = WORKER_OUTCOME_ADAPTER.validate_json((root / "work/outcome.json").read_bytes())
        if isinstance(outcome, WorkerFailure):
            self.store.fail(job_id, outcome.error)
        else:
            try:
                publication = asyncio.create_task(asyncio.to_thread(self.store.succeed, job_id, outcome.manifest))
                try:
                    await asyncio.shield(publication)
                except asyncio.CancelledError:
                    # Finish the atomic publication before releasing the storage lock.
                    await publication
                    raise
            except Exception:
                logger.exception("Could not publish artifacts for %s", job_id)
                self.store.fail(
                    job_id,
                    Error(code="artifact_write_failed", message="Required results could not be stored."),
                )
