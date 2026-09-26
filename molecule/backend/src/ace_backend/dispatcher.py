"""One durable queue consumer; each inference runs in a fresh, terminable process."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

from ace_backend.job_store import JobStore
from ace_backend.jobs_schema import WORKER_OUTCOME_ADAPTER, Error, InferenceConfig, WorkerFailure
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
                logger.exception("Inference worker failed for job %s", job.job_id)
                self.store.fail(
                    job.job_id,
                    Error(code="inference_failed", message="The inference worker could not complete the job."),
                )

    async def _execute(self, job_id: str) -> None:
        root = self.store.root / job_id
        config = InferenceConfig.model_validate_json((root / "request.json").read_bytes())
        environment = {**os.environ, "PYTHONHASHSEED": str(config.seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
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
                    # Kill the process group before another job can claim the GPU.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                await completion
        if process.returncode != 0:
            self.store.fail(job_id, Error(code="inference_failed", message="The inference worker exited unexpectedly."))
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
                logger.exception("Could not publish inference artifacts for %s", job_id)
                self.store.fail(
                    job_id,
                    Error(code="artifact_write_failed", message="Required inference results could not be stored."),
                )
