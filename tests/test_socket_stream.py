"""Real loopback HTTP: prove first SSE frame arrives before upstream completion."""
import asyncio
from pathlib import Path
import socket
import tempfile
import unittest
import httpx
import uvicorn
from llm_proxy_standard.app import Settings, create_app
from llm_proxy_standard.cache import SnapshotCache
from tests.support import Authority, profile, UPSTREAM
from tests.test_app import payload


class SocketStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_loopback_incremental_and_disconnect(self):
        closed = asyncio.Event()
        released = asyncio.Event()

        class Upstream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'
                await released.wait()
                yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

            async def aclose(self):
                closed.set()

        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token"
            token.write_text("synthetic")
            authority = Authority()
            snapshot = authority.snapshot(profile(capabilities=["text", "streaming"])).document()
            async def fetch():
                return snapshot
            cache = SnapshotCache(fetch, authority.trust)
            app = create_app(Settings(UPSTREAM, token, concurrency=1, deadline=3), cache,
                transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Upstream())))
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False, timeout_graceful_shutdown=1))
            runner = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                async with asyncio.timeout(5):
                    while not server.started:
                        if runner.done():
                            await runner
                        await asyncio.sleep(0.01)
                    async with httpx.AsyncClient(trust_env=False) as client:
                        async with client.stream("POST", f"http://127.0.0.1:{sock.getsockname()[1]}/chat/v1/chat/completions",
                                json={**payload("chat"), "stream": True}) as response:
                            first = await anext(response.aiter_lines())
                            self.assertIn("first", first)
                            self.assertFalse(released.is_set())
                        await asyncio.wait_for(closed.wait(), 2)
                    self.assertEqual(app.state.admission.slots._value, 1)
            finally:
                released.set()
                server.should_exit = True
                await asyncio.wait_for(runner, 5)
                sock.close()