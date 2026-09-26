"""Public errors intentionally omit underlying exceptions and filesystem paths."""

from ace_backend.jobs_schema import Error, ErrorDetail


class APIError(Exception):
    def __init__(self, status_code: int, error: Error) -> None:
        super().__init__(error.message)
        self.status_code = status_code
        self.error = error


def invalid_input(field: str, code: str, message: str, status_code: int = 422) -> APIError:
    error_code = {400: "malformed_request", 413: "payload_too_large", 415: "unsupported_media_type"}.get(
        status_code, "validation_error"
    )
    return APIError(
        status_code,
        Error(code=error_code, message=message, details=(ErrorDetail(field=field, code=code, message=message),)),
    )
