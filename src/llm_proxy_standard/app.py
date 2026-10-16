"""Private data-plane only. No publishing/signing route and no inference retries."""
import asyncio
import base64
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import uuid

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from starlette.requests import ClientDisconnect
from .cache import SnapshotCache, Unavailable
from .contracts import canonical, digest, strict_json
from .protocol import SSEParser, Stream, map_model, validate_document, validate_capabilities
from .registry import Trust
from .replay_cache import ReplayCache
from .rules import RequestPolicyError, prepare
from .stream_order import OrderedItems

SUFFIX = {"chat": "/chat/completions", "responses": "/responses", "messages": "/messages"}


@dataclass(frozen=True)
class Settings:
    upstream: str
    token_path: Path
    concurrency: int = 8
    queue_size: int = 8
    queue_wait: float = 0.1
    deadline: float = 180
    max_body: int = 1024 * 1024
    max_response: int = 16 * 1024 * 1024
    send_timeout: float = 10

    def __post_init__(self):
        parsed = urlsplit(self.upstream)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or
                parsed.query or parsed.fragment or parsed.path != "/api/v2/cortex/v1" or
                parsed.port not in (None, 443)):
            raise ValueError("Explicit Cortex HTTPS destination required")
        if (self.concurrency < 1 or self.queue_size < 0 or min(self.queue_wait, self.deadline,
                self.max_body, self.max_response, self.send_timeout) <= 0):
            raise ValueError("Invalid resource limits")


class Admission:
    def __init__(self, settings):
        self.slots = asyncio.Semaphore(settings.concurrency)
        self.waiting = 0
        self.settings = settings
        self.draining = False

    async def acquire(self):
        if self.draining or self.slots.locked() and self.waiting >= self.settings.queue_size:
            raise Unavailable("Admission limit")
        self.waiting += 1
        try:
            await asyncio.wait_for(self.slots.acquire(), self.settings.queue_wait)
        finally:
            self.waiting -= 1


async def close_response(response):
    try:
        with anyio.move_on_after(2, shield=True):
            await response.aclose()
    except Exception:
        pass


async def until_disconnect(request, operation):
    """Observe ASGI disconnect while waiting for upstream headers/body."""
    async def disconnected():
        while True:
            event = await request.receive()
            if event["type"] == "http.disconnect":
                return
            await asyncio.sleep(0.01)
    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        work.cancel()
        with suppress(asyncio.CancelledError):
            await work
        raise ClientDisconnect()
    finally:
        watcher.cancel()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


class BoundedStreamingResponse(StreamingResponse):
    def __init__(self, *args, cleanup, expires, send_timeout, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup, self.expires, self.send_timeout = cleanup, expires, send_timeout

    async def __call__(self, scope, receive, send):
        async def bounded_send(message):
            timeout = min(self.send_timeout, self.expires - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Client deadline")
            with anyio.fail_after(timeout):
                await send(message)
        try:
            with anyio.fail_after(max(0.001, self.expires - time.monotonic())):
                await super().__call__(scope, receive, bounded_send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await self.cleanup()


def error(protocol, status, message, request_id, *, code=None, field=None):
    kind = "invalid_request_error" if status < 500 else "api_error"
    body = {"error": {"type": kind, "message": message}, "request_id": request_id}
    if code:
        body["error"]["code"] = code
    if field:
        body["error"]["param"] = field
    if protocol == "messages":
        body["type"] = "error"
    return JSONResponse(body, status_code=status, headers={"x-request-id": request_id})


def create_app(settings, cache, *, transport=None):
    replica = os.environ.get("HOSTNAME", "local")
    release = os.environ.get("PROXY_ARTIFACT", "unconfigured")
    replay_cache = ReplayCache()
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False,
                timeout=httpx.Timeout(60, connect=10, write=30, pool=1),
                limits=httpx.Limits(max_connections=settings.concurrency)) as client:
            app.state.client = client
            app.state.admission = Admission(settings)
            app.state.replay_cache = replay_cache
            await cache.refresh()
            refresh = asyncio.create_task(cache.run())
            try:
                yield
            finally:
                app.state.admission.draining = True
                refresh.cancel()
                with suppress(asyncio.CancelledError):
                    await refresh

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/readyz")
    async def ready():
        try:
            token_ok = bool(settings.token_path.read_text().strip())
        except OSError:
            token_ok = False
        good = cache.available() and token_ok and not app.state.admission.draining
        return JSONResponse({"ready": good, "revision": cache.revision, "refresh_ok": cache.last_refresh_ok,
                             "replica": replica, "release": release}, status_code=200 if good else 503)

    def descriptor(profile, protocol):
        status = cache.release_status(profile.id) if hasattr(cache, "release_status") else "candidate"
        metadata = {"release_status": status, "expires_at": profile.expires_at}
        if getattr(profile, "schema_version", 1) == 2:
            metadata.update(profile_revision=profile.revision, capabilities=profile.capabilities,
                            response_stream=profile.response_stream,
                            constraints={path: value.document() for path, value in profile.constraints.items()})
        if protocol == "messages":
            return {"id": profile.id, "type": "model", "display_name": profile.id, "created_at": profile.created_at,
                    "metadata": metadata}
        return {"id": profile.id, "object": "model", "created": 0, "owned_by": "cortex", "metadata": metadata}

    def register(protocol, prefix):
        async def models(request: Request):
            request_id = str(uuid.uuid4())
            try:
                rows = [descriptor(profile, protocol) for profile in cache.profiles(protocol)]
                if protocol != "messages":
                    return JSONResponse({"object": "list", "data": rows}, headers={"Cache-Control": "no-store"})
                params = request.query_params
                if not set(params) <= {"limit", "after_id", "before_id"} or params.get("after_id") and params.get("before_id"):
                    raise ValueError("Invalid pagination")
                limit = int(params.get("limit", "20"))
                if not 1 <= limit <= 1000:
                    raise ValueError("Invalid page limit")
                ids = [row["id"] for row in rows]
                if params.get("after_id"):
                    rows = rows[ids.index(params["after_id"]) + 1:]
                if params.get("before_id"):
                    rows = rows[:ids.index(params["before_id"])]
                more, rows = len(rows) > limit, rows[:limit]
                return JSONResponse({"data": rows, "has_more": more, "first_id": rows[0]["id"] if rows else None,
                    "last_id": rows[-1]["id"] if rows else None}, headers={"Cache-Control": "no-store"})
            except Unavailable:
                return error(protocol, 503, "Registry unavailable", request_id)
            except ValueError:
                return error(protocol, 400, "Invalid pagination", request_id)

        async def detail(identifier: str):
            try:
                return JSONResponse(descriptor(cache.resolve(protocol, identifier), protocol), headers={"Cache-Control": "no-store"})
            except Unavailable:
                return error(protocol, 503, "Registry unavailable", str(uuid.uuid4()))
            except KeyError:
                return error(protocol, 404, "Model unavailable", str(uuid.uuid4()))

        async def invoke(request: Request):
            request_id = str(uuid.uuid4())
            expires = time.monotonic() + settings.deadline
            response, acquired = None, False
            admission = app.state.admission
            try:
                async with asyncio.timeout_at(expires):
                    # Admit before buffering a body: bound simultaneous slow uploads too.
                    try:
                        await admission.acquire()
                    except (Unavailable, TimeoutError):
                        return error(protocol, 429, "Bounded admission limit reached", request_id)
                    acquired = True
                    raw = bytearray()
                    async for chunk in request.stream():
                        if len(raw) + len(chunk) > settings.max_body:
                            return error(protocol, 413, "Request too large", request_id)
                        raw.extend(chunk)
                    body = strict_json(raw)
                    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
                        raise ValueError("Model required")
                    profile = cache.resolve(protocol, body["model"])
                    revision = cache.revision
                    release_status = cache.release_status(profile.id) if hasattr(cache, "release_status") else "candidate"
                    rehydrated_changes = []
                    if getattr(profile, "normalize_client_options", False):
                        body, rehydrated_changes = replay_cache.rehydrate(body, protocol)
                    effective, changes = prepare(body, profile)
                    changes = rehydrated_changes + changes
                    current = cache.resolve(protocol, profile.id)
                    if current != profile:
                        raise Unavailable("Profile changed while waiting")
                    token = settings.token_path.read_text().strip()
                    if not token:
                        raise OSError("Missing token")
                    headers = {"Authorization": "Bearer " + token, "X-Snowflake-Authorization-Token-Type": "OAUTH",
                               "Content-Type": "application/json", "Accept": "text/event-stream" if effective.get("stream") else "application/json"}
                    if protocol == "messages":
                        headers["anthropic-version"] = "2023-06-01"
                    upstream = app.state.client.build_request("POST", settings.upstream + SUFFIX[protocol], json=effective, headers=headers)
                    response = await until_disconnect(request, app.state.client.send(upstream, stream=True))
                    if not 200 <= response.status_code < 300:
                        status = response.status_code if response.status_code >= 400 else 502
                        return error(protocol, status, f"Upstream HTTP {response.status_code}; no retry or fallback", request_id)
                    metadata = {"x-request-id": request_id, "x-proxy-revision": revision,
                                "x-proxy-profile": digest(profile.document()), "Cache-Control": "no-store",
                                "x-proxy-replica": replica, "x-proxy-release": release}
                    metadata["x-proxy-release-status"] = release_status
                    if changes:
                        metadata["x-proxy-adjustments"] = base64.urlsafe_b64encode(canonical(changes)).decode()
                    ordered = getattr(profile, "response_stream", "native") == "ordered-items"
                    if ordered and effective.get("stream"):
                        metadata["x-proxy-stream-mode"] = "ordered-items"
                    if not effective.get("stream"):
                        async def read_output():
                            output = bytearray()
                            async for chunk in response.aiter_bytes():
                                if len(output) + len(chunk) > settings.max_response:
                                    raise RuntimeError("Response too large")
                                output.extend(chunk)
                            return output
                        output = await until_disconnect(request, read_output())
                        document = strict_json(output)
                        validate_document(document, protocol)
                        validate_capabilities(document, protocol, profile)
                        replay_cache.record_document(protocol, document)
                        return JSONResponse(map_model(document, protocol, profile), headers=metadata)
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise RuntimeError("Expected SSE")
                    stream_response = response
                    cleaned = False

                    async def cleanup():
                        nonlocal cleaned
                        if not cleaned:
                            cleaned = True
                            try:
                                await close_response(stream_response)
                            finally:
                                admission.slots.release()

                    async def stream():
                        parser, validator = SSEParser(), Stream(protocol, profile)
                        ordering = OrderedItems(settings.max_response) if ordered else None
                        total, terminal = 0, bytearray()
                        try:
                            async with asyncio.timeout_at(expires):
                                async for chunk in stream_response.aiter_bytes():
                                    total += len(chunk)
                                    if total > settings.max_response:
                                        raise RuntimeError("Stream too large")
                                    for event in parser.feed(chunk):
                                        mapped = validator.event(event)
                                        for frame in ordering.feed(mapped) if ordering else [mapped]:
                                            if validator.terminal:
                                                terminal.extend(frame)
                                            else:
                                                yield frame
                                parser.finish()
                                validator.finish()
                                replay_cache.record_stream(protocol, validator)
                                if terminal:
                                    yield bytes(terminal)
                        finally:
                            # On protocol/transport failure close incompletely, never invent DONE.
                            await cleanup()

                    result = BoundedStreamingResponse(stream(), media_type="text/event-stream", headers=metadata,
                        cleanup=cleanup, expires=expires, send_timeout=settings.send_timeout)
                    response, acquired = None, False
                    return result
            except ClientDisconnect:
                return Response(status_code=499)
            except KeyError:
                return error(protocol, 404, "Model unavailable", request_id)
            except Unavailable:
                return error(protocol, 503, "Registry unavailable or profile changed", request_id)
            except (TimeoutError, httpx.TimeoutException):
                return error(protocol, 504, "Request deadline exceeded", request_id)
            except RequestPolicyError as exc:
                return error(protocol, 400, "Request option is not supported by the selected profile",
                             request_id, code=exc.code, field=exc.field)
            except ValueError:
                return error(protocol, 400 if response is None else 502, "Invalid request or protocol response", request_id)
            except (OSError, RuntimeError, httpx.HTTPError, TypeError):
                return error(protocol, 502, "Upstream transport or protocol failure", request_id)
            finally:
                if response is not None:
                    await close_response(response)
                if acquired:
                    admission.slots.release()

        app.add_api_route(prefix + "/models", models, methods=["GET"])
        app.add_api_route(prefix + "/models/{identifier}", detail, methods=["GET"])
        app.add_api_route(prefix + SUFFIX[protocol], invoke, methods=["POST"])

    for protocol in SUFFIX:
        register(protocol, "/" + protocol + "/v1")
    return app


def environment():
    host = os.environ["SNOWFLAKE_HOST"]
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
        raise ValueError("Invalid platform host")
    limits = {name: int(os.environ[env]) for name, env in {
        "concurrency": "PROXY_CONCURRENCY", "queue_size": "PROXY_QUEUE_SIZE",
        "max_body": "PROXY_MAX_BODY", "max_response": "PROXY_MAX_RESPONSE"}.items() if env in os.environ}
    limits.update({name: float(os.environ[env]) for name, env in {
        "queue_wait": "PROXY_QUEUE_WAIT", "deadline": "PROXY_DEADLINE",
        "send_timeout": "PROXY_SEND_TIMEOUT"}.items() if env in os.environ})
    settings = Settings("https://" + host + "/api/v2/cortex/v1", Path("/snowflake/session/token"), **limits)
    keys = strict_json(Path(os.environ["PROXY_TRUST_FILE"]).read_bytes())
    experimental = []
    historical_artifacts = []
    if os.environ.get("PROXY_HISTORICAL_ARTIFACTS_FILE"):
        historical_artifacts = strict_json(Path(os.environ["PROXY_HISTORICAL_ARTIFACTS_FILE"]).read_bytes())
        if not isinstance(historical_artifacts, list):
            raise ValueError("Historical artifacts must be an explicit list")
    if os.environ.get("PROXY_EXPERIMENTAL_PINS_FILE"):
        experimental = strict_json(Path(os.environ["PROXY_EXPERIMENTAL_PINS_FILE"]).read_bytes())
        if not isinstance(experimental, list):
            raise ValueError("Experimental pins must be a list of exact entry digests")
    admin_approvers = [v.strip() for v in os.environ.get("PROXY_ADMIN_APPROVERS", "").split(",") if v.strip()]
    serve_historical = os.environ.get("PROXY_SERVE_HISTORICAL_ARTIFACTS", "").lower() in {"1", "true"}
    trust = Trust({key: (value["identity"], base64.b64decode(value["public_key"], validate=True)) for key, value in keys.items()},
                  os.environ["PROXY_ACCOUNT"], os.environ["PROXY_ARTIFACT"], settings.upstream,
                  experimental_entries=experimental, historical_artifacts=historical_artifacts,
                  admin_approvers=admin_approvers, serve_historical_artifacts=serve_historical)
    url = os.environ["PROXY_REGISTRY_URL"]
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or not parsed.hostname or not parsed.hostname.endswith(".svc.spcs.internal") or
            parsed.username or parsed.password or parsed.path != "/snapshot" or parsed.query or parsed.fragment):
        raise ValueError("Exact private registry snapshot URL required")
    async def fetch():
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=4) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 4 * 1024 * 1024:
                        raise ValueError("Registry too large")
                    raw.extend(chunk)
                return strict_json(raw)
    phase_url = os.environ.get("PROXY_PHASE_BUDGET_URL")
    transport = None
    if phase_url:
        from .phase_budget import PhaseTransport
        transport = PhaseTransport(settings.upstream, phase_url, os.environ["PROXY_PHASE_CAMPAIGN"],
                                   os.environ["PROXY_ARTIFACT"] + ":" + os.environ.get("HOSTNAME", "unknown"))
    elif "PROXY_PHASE_CAMPAIGN" in os.environ:
        raise ValueError("Phase campaign configured without its authority")
    return create_app(settings, SnapshotCache(fetch, trust), transport=transport)