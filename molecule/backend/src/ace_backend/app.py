"""FastAPI application factory for the independent ACE backend."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.middleware.base import RequestResponseEndpoint

from ace_backend.capabilities import detect_capabilities
from ace_backend.schemas import Capabilities
from ace_backend.settings import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, Capabilities]]:
        # Probe once in the serving process. No model is loaded here or by GET.
        yield {"capabilities": detect_capabilities(settings)}

    app = FastAPI(title="ACE backend", version="1.0.0", lifespan=lifespan, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET"],
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

    @app.get("/api/v1/capabilities", response_model=Capabilities, tags=["capabilities"])
    async def capabilities(request: Request, response: Response) -> Capabilities:
        """Return runtime prerequisites and limits, independently for each feature."""
        response.headers["Cache-Control"] = "no-store"
        return cast(Capabilities, request.state.capabilities)

    return app


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = request.state.request_id
    return JSONResponse(
        status_code=status_code,
        content={"request_id": request_id, "error": {"code": code, "message": message, "details": []}},
        headers={**(headers or {}), "X-Request-ID": request_id},
    )
