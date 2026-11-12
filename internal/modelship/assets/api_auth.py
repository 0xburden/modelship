"""Native-only access boundary. No key values in argv, metadata, URLs or errors."""
import hashlib
import hmac
import os
import re
import stat


class Access:
    def __init__(self, listen, digest):
        self.listen = listen
        self.digest = digest


def load_access(environ=None):
    env = os.environ if environ is None else environ
    listen = env.get("MODEL_LISTEN", "127.0.0.1")
    path = env.get("MODEL_API_KEY_FILE", "")
    if listen not in ("127.0.0.1", "0.0.0.0"):
        raise RuntimeError("invalid native listener")
    if listen != "127.0.0.1" and not path:
        raise RuntimeError("non-loopback listener requires API authentication")
    digest = None
    if path:
        message = "API key file must be an owned regular 0600 file containing 32-128 base64url characters"
        if not os.path.isabs(path):
            raise RuntimeError(message)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid() or not 32 <= info.st_size <= 130:
                    raise ValueError()
                key = stream.read(131).removesuffix(b"\n").removesuffix(b"\r")
                if not re.fullmatch(rb"[A-Za-z0-9_-]{32,128}", key):
                    raise ValueError()
                digest = hashlib.sha256(key).digest()
        except (OSError, ValueError):
            raise RuntimeError(message) from None
    return Access(listen, digest)


class BearerAuth:
    """Check every HTTP path before routing/body parsing; no query/cookie keys."""
    def __init__(self, app, access):
        self.app = app
        self.digest = access.digest

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http" or self.digest is None:
            await self.app(scope, receive, send)
            return
        headers = [value for name, value in scope.get("headers", []) if name.lower() == b"authorization"]
        valid = False
        if len(headers) == 1 and len(headers[0]) <= 512:
            scheme, separator, key = headers[0].partition(b" ")
            if scheme.lower() == b"bearer" and separator and re.fullmatch(rb"[A-Za-z0-9_-]{32,128}", key):
                valid = hmac.compare_digest(hashlib.sha256(key).digest(), self.digest)
        if not valid:
            body = b'{"detail":"Authentication required"}'
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="modelship"'), (b"cache-control", b"no-store"),
            ]})
            await send({"type": "http.response.body", "body": body})
            return
        async def private_send(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [(name, value) for name, value in message.get("headers", []) if name.lower() != b"cache-control"] + [(b"cache-control", b"no-store")]}
            await send(message)
        await self.app(scope, receive, private_send)
