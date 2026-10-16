import asyncio
import os
import time
import unittest
from unittest.mock import patch
import httpx
from llm_proxy_standard.contracts import Snapshot
from llm_proxy_standard.registry_service import environment


class Reader:
    def read(self):
        return Snapshot()


class SlowPublisher:
    def __init__(self, *args):
        self.registry = Reader()

    def verify_floor(self):
        time.sleep(0.15)

    def scan(self):
        time.sleep(0.15)
        return []

    def close(self):
        pass


class ResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_io_does_not_block_ready_or_snapshot(self):
        env = {"PROXY_REGISTRY_DB": "/unused", "PROXY_PUBLICATION_INBOX": "/unused",
               "PROXY_CHECKPOINTS": "/unused", "PROXY_TRUST_FILE": "/unused"}
        with patch.dict(os.environ, env, clear=True), patch("llm_proxy_standard.registry_service.load_trust"), \
                patch("llm_proxy_standard.registry_service.Registry"), \
                patch("llm_proxy_standard.registry_service.Publisher", SlowPublisher):
            app = environment()
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0.02)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture") as client:
                    start = time.monotonic()
                    self.assertEqual((await client.get("/readyz")).status_code, 200)
                    self.assertEqual((await client.get("/snapshot")).json(), Snapshot().document())
                    self.assertLess(time.monotonic() - start, 0.1)
                await asyncio.sleep(0.5)