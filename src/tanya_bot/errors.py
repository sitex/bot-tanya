"""Safe errors shared by the provider and HTTP boundary."""

from typing import override

from tanya_bot.schemas import ErrorDetail, ErrorResponse


class ServiceError(Exception):
    """Safe HTTP error; exception traceback attributes must remain mutable."""

    status: int
    code: str
    message: str

    def __init__(self, status: int, code: str, message: str) -> None:
        """Keep only sanitized HTTP fields in the exception."""
        self.status = status
        self.code = code
        self.message = message
        super().__init__(code)

    @override
    def __str__(self) -> str:
        return self.code

    def envelope(self) -> ErrorResponse:
        """Serialize the stable public error contract."""
        return ErrorResponse(detail=ErrorDetail(code=self.code, message=self.message))
