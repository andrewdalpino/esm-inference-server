from time import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from starlette.middleware.base import BaseHTTPMiddleware


class ExceptionHandler(BaseHTTPMiddleware):
    """Handle exceptions emitted from the domain model."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException as e:
            return JSONResponse(
                content={"message": e.detail}, status_code=e.status_code
            )
        except Exception as e:
            return JSONResponse(
                content={"message": "Something went wrong."}, status_code=500
            )


class TokenAuthentication(BaseHTTPMiddleware):
    """Check for the API token in the request headers."""

    def __init__(self, app, api_token: str):
        super().__init__(app)

        self.api_token = api_token


    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("Authorization")

        if not token or token != f"Bearer {self.api_token}":
            return JSONResponse(content={"message": "Unauthorized."}, status_code=401)

        return await call_next(request)


class ResponseTime(BaseHTTPMiddleware):
    """Measure the time taken to process a request."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time()

        response = await call_next(request)

        duration = time() - start_time

        response.headers["X-Response-Time"] = str(duration)

        return response
