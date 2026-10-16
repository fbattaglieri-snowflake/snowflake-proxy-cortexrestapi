import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import httpx
from starlette.requests import Request, ClientDisconnect
from llm_proxy_standard.app import Settings, create_app, Admission
from llm_proxy_standard.cache import SnapshotCache, Unavailable
from tests.support import Authority, profile, UPSTREAM
from tests.test_app import payload, sse, ByteStream


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self):
        self.reads, self.closed = 0, False
        self.blocked = asyncio.Event()

    async def __aiter__(self):
        self.reads += 1
        yield b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        self.reads += 1
        self.blocked.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.token = Path(self.directory.name) / "token"
        self.token.write_text("synthetic")
        self.authority = Authority()
        self.document = self.authority.snapshot(profile(capabilities=["text", "streaming"])).document()
        async def fetch():
            return self.document
        self.cache = SnapshotCache(fetch, self.authority.trust)

    async def asyncTearDown(self):
        self.directory.cleanup()

    def request(self, stream=True):
        async def receive():
            return {"type": "http.request", "body": json.dumps({**payload("chat"), "stream": stream}).encode(), "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/chat/v1/chat/completions", "headers": []}, receive)

    def endpoint(self, app):
        return next(route.endpoint for route in app.routes if route.path == "/chat/v1/chat/completions")

    async def test_incremental_pull_backpressure_cancel_releases_resources(self):
        upstream = ControlledStream()
        app = create_app(Settings(UPSTREAM, self.token, concurrency=1), self.cache,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream)))
        async with app.router.lifespan_context(app):
            response = await self.endpoint(app)(self.request())
            self.assertIn(b"partial", await anext(response.body_iterator))
            await asyncio.sleep(0.02)
            self.assertEqual(upstream.reads, 1)  # No eager drain while consumer pauses.
            task = asyncio.create_task(anext(response.body_iterator))
            await upstream.blocked.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(upstream.closed)
            self.assertEqual(app.state.admission.slots._value, 1)

    async def test_deadline_stream_no_success(self):
        upstream = ControlledStream()
        app = create_app(Settings(UPSTREAM, self.token, concurrency=1, deadline=0.04), self.cache,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream)))
        async with app.router.lifespan_context(app):
            response = await self.endpoint(app)(self.request())
            self.assertNotIn(b"[DONE]", await anext(response.body_iterator))
            with self.assertRaises(TimeoutError):
                await anext(response.body_iterator)
            self.assertTrue(upstream.closed)
            self.assertEqual(app.state.admission.slots._value, 1)

    async def test_cancel_before_headers_and_nonstream_deadline(self):
        entered = asyncio.Event()
        async def handler(_):
            entered.set()
            await asyncio.Event().wait()
        app = create_app(Settings(UPSTREAM, self.token, concurrency=1, deadline=0.06), self.cache, transport=httpx.MockTransport(handler))
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(self.endpoint(app)(self.request(False)))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(app.state.admission.slots._value, 1)
            response = await self.endpoint(app)(self.request(False))
            self.assertEqual(response.status_code, 504)
            self.assertEqual(app.state.admission.slots._value, 1)

    async def test_queue_count_wait_and_draining(self):
        admission = Admission(Settings(UPSTREAM, self.token, concurrency=1, queue_size=1, queue_wait=0.04))
        await admission.acquire()
        waiting = asyncio.create_task(admission.acquire())
        await asyncio.sleep(0.005)
        with self.assertRaises(Unavailable):
            await admission.acquire()
        with self.assertRaises(TimeoutError):
            await waiting
        self.assertEqual(admission.waiting, 0)
        admission.slots.release()
        admission.draining = True
        with self.assertRaises(Unavailable):
            await admission.acquire()

    async def test_asgi_disconnect_before_headers_cancels_upstream(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def handler(_):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        first = True
        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": json.dumps(payload("chat")).encode(), "more_body": False}
            await entered.wait()
            return {"type": "http.disconnect"}
        request = Request({"type": "http", "method": "POST", "path": "/chat/v1/chat/completions", "headers": []}, receive)
        app = create_app(Settings(UPSTREAM, self.token, concurrency=1), self.cache, transport=httpx.MockTransport(handler))
        async with app.router.lifespan_context(app):
            response = await self.endpoint(app)(request)
            self.assertEqual(response.status_code, 499)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(app.state.admission.slots._value, 1)

    async def test_send_timeout_even_before_stream_iteration_cleans_up(self):
        upstream = ControlledStream()
        app = create_app(Settings(UPSTREAM, self.token, concurrency=1, send_timeout=0.03), self.cache,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream)))
        async with app.router.lifespan_context(app):
            response = await self.endpoint(app)(self.request())
            async def send(_):
                await asyncio.Event().wait()
            async def receive():
                await asyncio.Event().wait()
            # Starlette maps send-side OSError (including TimeoutError) to disconnect.
            with self.assertRaises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            self.assertTrue(upstream.closed)
            self.assertEqual(app.state.admission.slots._value, 1)

    async def test_registry_outage_does_not_cancel_inflight(self):
        upstream = ByteStream(sse("chat"))
        app = create_app(Settings(UPSTREAM, self.token), self.cache,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream)))
        async with app.router.lifespan_context(app):
            response = await self.endpoint(app)(self.request())
            self.cache.updated -= 301
            raw = b"".join([chunk async for chunk in response.body_iterator])
            self.assertIn(b"[DONE]", raw)
            self.assertEqual((await self.endpoint(app)(self.request())).status_code, 503)