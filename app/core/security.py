"""Single-operator access gate, applied before request bodies are parsed."""

import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class AccessControlMiddleware:
    def __init__(self, app: ASGIApp, access_key: str) -> None:
        self.app = app
        self.access_key = access_key.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        is_api = path == "/api" or path.startswith("/api/")
        if is_api and self.access_key:
            authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
            scheme, _, candidate = authorization.partition(b" ")
            if scheme.lower() != b"bearer" or not secrets.compare_digest(
                candidate, self.access_key
            ):
                response = JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "unauthorized",
                            "message": "A valid workspace access key is required.",
                            "details": {},
                        }
                    },
                    headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
                )
                await response(scope, receive, send)
                return

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-frame-options", b"DENY"),
                    ]
                )
                if is_api:
                    headers.append((b"cache-control", b"no-store"))
                if path == "/" or path.startswith("/assets/"):
                    headers.append(
                        (
                            b"content-security-policy",
                            (
                                "default-src 'self'; script-src 'self'; style-src 'self'; "
                                "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                            ).encode("ascii"),
                        )
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, secure_send)
