"""Shared private phase budget; an ambiguous reservation is never retried."""
import asyncio
import time
from urllib.parse import urlsplit

import httpx

from .contracts import digest, strict_json


def validate_authority(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or not parsed.hostname or
            not parsed.hostname.endswith(".svc.spcs.internal") or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            parsed.path or parsed.port != 8091):
        raise ValueError("Exact private phase authority origin on port 8091 required")
    return url


class LeaseStream(httpx.AsyncByteStream):
    def __init__(self, stream, owner, lease, expires):
        self.stream, self.owner, self.lease, self.expires = stream, owner, lease, expires
        self.closed = False

    async def __aiter__(self):
        async with asyncio.timeout_at(self.expires):
            async for chunk in self.stream:
                yield chunk

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        try:
            await self.stream.aclose()
        finally:
            await self.owner.release(self.lease)


class PhaseTransport(httpx.AsyncBaseTransport):
    def __init__(self, upstream, authority, campaign, execution, *, inner=None, control=None):
        self.upstream = upstream
        self.authority = validate_authority(authority)
        if not campaign or not execution:
            raise ValueError("Frozen phase and execution identities required")
        self.campaign, self.execution = campaign, execution
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        self.control = httpx.AsyncClient(transport=control, timeout=5, trust_env=False, follow_redirects=False)

    async def release(self, lease):
        # Failure intentionally leaves the durable slot occupied.
        response = await self.control.post(self.authority + "/release", json={
            "campaign": self.campaign, "lease": lease})
        response.raise_for_status()

    async def handle_async_request(self, request):
        from .app import SUFFIX
        protocol = next((key for key, suffix in SUFFIX.items()
                         if str(request.url) == self.upstream + suffix), None)
        if request.method != "POST" or protocol is None:
            raise ValueError("Phase transport destination mismatch")
        body = strict_json(request.content)
        field = {"chat": "max_completion_tokens", "responses": "max_output_tokens", "messages": "max_tokens"}[protocol]
        if protocol == "chat" and "max_tokens" in body:
            if field in body:
                raise ValueError("Ambiguous output token limits")
            field = "max_tokens"
        tokens = body.get(field)
        if type(tokens) is not int or not 0 < tokens <= 2048:
            raise ValueError("Explicit bounded output tokens required")
        started = time.monotonic()
        reservation = await self.control.post(self.authority + "/reserve", json={
            "campaign": self.campaign, "execution": self.execution, "protocol": protocol,
            "tokens": tokens, "request_sha256": digest({"protocol": protocol, "body": body})})
        reservation.raise_for_status()
        record = strict_json(reservation.content)
        if (set(record) != {"lease", "remaining_seconds"} or not isinstance(record["lease"], str) or
                not record["lease"] or type(record["remaining_seconds"]) not in (int, float) or
                not 0 < record["remaining_seconds"] <= 7200):
            raise ValueError("Malformed phase lease; no upstream attempt permitted")
        # Conservatively subtract the full control roundtrip, without clock sync.
        expires = started + record["remaining_seconds"]
        try:
            async with asyncio.timeout_at(expires):
                response = await self.inner.handle_async_request(request)
            return httpx.Response(response.status_code, headers=response.headers, extensions=response.extensions,
                                  stream=LeaseStream(response.stream, self, record["lease"], expires))
        except BaseException:
            await self.release(record["lease"])
            raise

    async def aclose(self):
        try:
            await self.inner.aclose()
        finally:
            await self.control.aclose()