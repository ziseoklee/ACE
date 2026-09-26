"""FastAPI application factory for the independent ACE backend."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic.json_schema import JsonSchemaValue
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import RequestResponseEndpoint

from ace_backend.capabilities import detect_capabilities
from ace_backend.dispatcher import Dispatcher
from ace_backend.errors import APIError
from ace_backend.evaluation_inputs import EVALUATION_FILE_FIELDS, prepare_evaluation, read_evaluation
from ace_backend.evaluation_schema import EVALUATION_CONFIG_EXAMPLE, EvaluationConfig, JobResult
from ace_backend.inputs import prepare_submission, read_submission
from ace_backend.job_store import JobStore
from ace_backend.jobs_schema import (
    CONFIG_EXAMPLE,
    INFERENCE_CONFIG_ADAPTER,
    ArtifactList,
    Error,
    ErrorDetail,
    ErrorResponse,
    Job,
    QueuedJob,
)
from ace_backend.schemas import Capabilities
from ace_backend.settings import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, object]]:
        # Probe once in the serving process. No model is loaded here or by GET.
        store = JobStore(settings)
        try:
            readiness = detect_capabilities(settings)
            dispatcher = Dispatcher(store, settings)
            consumer = asyncio.create_task(dispatcher.run())
            try:
                yield {"capabilities": readiness, "jobs": store, "dispatcher": dispatcher}
            finally:
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer
        finally:
            store.close()

    app = FastAPI(title="ACE backend", version="1.0.0", lifespan=lifespan, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["Location", "Retry-After", "X-Request-ID", "Content-Disposition"],
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "http_error")
        return _error_response(request, error.status_code, code, str(error.detail), error.headers)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        logger.error("Unhandled API error", exc_info=error)
        return _error_response(request, 500, "internal_error", "An unexpected server error occurred.")

    @app.exception_handler(APIError)
    async def api_error(request: Request, error: APIError) -> JSONResponse:
        headers = {"Retry-After": "2"} if error.status_code == 429 else None
        return _error_response(
            request, error.status_code, error.error.code, error.error.message, headers, error.error.details
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        details = tuple(
            ErrorDetail(
                field=".".join(str(item) for item in entry["loc"]), code=entry["type"], message="Invalid request value."
            )
            for entry in error.errors()
        )
        return _error_response(request, 422, "validation_error", "Invalid request.", details=details)

    @app.get("/api/v1/capabilities", response_model=Capabilities, tags=["capabilities"])
    async def capabilities(request: Request, response: Response) -> Capabilities:
        """Return runtime prerequisites and limits, independently for each feature."""
        response.headers["Cache-Control"] = "no-store"
        return cast(Capabilities, request.state.capabilities)

    @app.post(
        "/api/v1/inference/jobs",
        response_model=QueuedJob,
        status_code=202,
        tags=["inference"],
        responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 429, 503, 500)},
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["pocket_pdb", "fragment_sdf", "reference_ligand_sdf", "config"],
                            "properties": {
                                "pocket_pdb": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "One protein model (.pdb).",
                                },
                                "fragment_sdf": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "One connected, sanitizable 3D scaffold (.sdf).",
                                },
                                "reference_ligand_sdf": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "One connected, sanitizable 3D reference (.sdf).",
                                },
                                "config": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "contentSchema": _inline_config_schema(INFERENCE_CONFIG_ADAPTER.json_schema()),
                                    "description": "Required UTF-8 InferenceConfig JSON as a regular form field, not a file. Select nr_scaffold_v1 or fkc_scaffold_v1 with moe parameters, or ace_scaffold_v1 with ace parameters. All fields for the selected preset are required; only num_ligand_atoms may be null. num_samples is one particle batch.",
                                    "example": json.dumps(CONFIG_EXAMPLE),
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def submit_inference(request: Request, response: Response) -> QueuedJob:
        """Validate and persist inputs, then queue NR, FKC, or ACE inference independently of the HTTP connection."""
        readiness = cast(Capabilities, request.state.capabilities)
        if not readiness.inference.available:
            raise APIError(503, Error(code="inference_unavailable", message="Inference prerequisites are unavailable."))
        submission = await read_submission(request, settings.limits)
        prepared = await run_in_threadpool(prepare_submission, submission, settings.limits)
        store = cast(JobStore, request.state.jobs)
        job = await run_in_threadpool(store.submit, prepared)
        cast(Dispatcher, request.state.dispatcher).wake.set()
        response.headers.update({"Location": job.links.self, "Retry-After": "2", "Cache-Control": "no-store"})
        return job

    @app.post(
        "/api/v1/evaluation/jobs",
        response_model=QueuedJob,
        status_code=202,
        tags=["evaluation"],
        responses={code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 415, 422, 429, 503, 500)},
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["config"],
                            "properties": {
                                **{field: {"type": "string", "format": "binary"} for field in EVALUATION_FILE_FIELDS},
                                "config": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "contentSchema": _inline_config_schema(EvaluationConfig.model_json_schema()),
                                    "description": "EvaluationConfig JSON as a regular form field. Inference-job sources forbid files. Upload sources require ligand_sdf, plus fragment_sdf for scaffold preservation and pocket_pdb/reference_ligand_sdf for docking. All config fields are required; docking must be null unless requested.",
                                    "example": json.dumps(EVALUATION_CONFIG_EXAMPLE),
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def submit_evaluation(request: Request, response: Response) -> QueuedJob:
        """Preserve evaluation inputs and queue CPU metrics/docking independently of inference availability."""
        submission = await read_evaluation(request, settings.limits)
        readiness = cast(Capabilities, request.state.capabilities)
        if any(not getattr(readiness.evaluation, metric).available for metric in submission.config.metrics):
            raise APIError(
                503, Error(code="evaluation_unavailable", message="A requested evaluation prerequisite is unavailable.")
            )
        store = cast(JobStore, request.state.jobs)
        prepared = await run_in_threadpool(prepare_evaluation, submission, settings.limits, store)
        job = await run_in_threadpool(store.submit_evaluation, prepared)
        cast(Dispatcher, request.state.dispatcher).wake.set()
        response.headers.update({"Location": job.links.self, "Retry-After": "2", "Cache-Control": "no-store"})
        return job

    job_errors = {code: {"model": ErrorResponse} for code in (404, 422, 500)}

    @app.get("/api/v1/jobs/{job_id}", response_model=Job, tags=["jobs"], responses=job_errors)
    def get_job(job_id: str, request: Request, response: Response) -> Job:
        job = cast(JobStore, request.state.jobs).get(job_id)
        response.headers["Cache-Control"] = "no-store"
        if job.status in {"queued", "running"}:
            response.headers["Retry-After"] = "2"
        return job

    @app.get(
        "/api/v1/jobs/{job_id}/result",
        response_model=JobResult,
        tags=["jobs"],
        responses={**job_errors, 409: {"model": ErrorResponse}},
    )
    def get_result(job_id: str, request: Request, response: Response) -> JobResult:
        response.headers["Cache-Control"] = "no-store"
        return cast(JobStore, request.state.jobs).result(job_id)

    @app.get("/api/v1/jobs/{job_id}/artifacts", response_model=ArtifactList, tags=["jobs"], responses=job_errors)
    def get_artifacts(job_id: str, request: Request, response: Response) -> ArtifactList:
        store = cast(JobStore, request.state.jobs)
        job = store.get(job_id)
        response.headers["Cache-Control"] = "no-store"
        return ArtifactList(
            job_id=job.job_id, artifacts=tuple(file.artifact for file in store.manifest(job.job_id).files)
        )

    @app.get(
        "/api/v1/jobs/{job_id}/artifacts/{artifact_id}",
        tags=["jobs"],
        response_class=FileResponse,
        responses=job_errors,
    )
    def get_artifact(job_id: str, artifact_id: str, request: Request, download: bool = False) -> FileResponse:
        path, artifact = cast(JobStore, request.state.jobs).artifact(job_id, artifact_id)
        return FileResponse(
            path,
            media_type=artifact.media_type,
            filename=artifact.filename,
            content_disposition_type="attachment" if download else "inline",
            headers={"Cache-Control": "no-store"},
        )

    return app


def _inline_config_schema(schema: JsonSchemaValue) -> object:
    definitions = schema.pop("$defs", {})

    def inline(value: object) -> object:
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].rsplit("/", 1)[1]])
            # Inline oneOf branches retain their literal discriminator constraints.
            return {key: inline(item) for key, item in value.items() if key != "discriminator"}
        if isinstance(value, list):
            return [inline(item) for item in value]
        return value

    return inline(schema)


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
    details: tuple[ErrorDetail, ...] = (),
) -> JSONResponse:
    request_id = request.state.request_id
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            request_id=request_id, error=Error(code=code, message=message, details=details)
        ).model_dump(mode="json"),
        headers={**(headers or {}), "X-Request-ID": request_id, "Cache-Control": "no-store"},
    )
