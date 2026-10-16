import unittest
from datetime import timedelta
from llm_proxy_standard.cache import SnapshotCache, Unavailable
from llm_proxy_standard.contracts import utcnow
from tests.support import Authority, profile


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.authority = Authority()
        self.document = self.authority.snapshot().document()
        self.tick, self.now, self.failed = 0, utcnow(), False
        async def fetch():
            if self.failed:
                raise OSError("offline")
            return self.document
        self.cache = SnapshotCache(fetch, self.authority.trust, monotonic=lambda: self.tick, now=lambda: self.now)
        self.assertTrue(await self.cache.refresh())

    async def test_ttl_failure_does_not_renew(self):
        self.tick, self.failed = 299, True
        self.assertFalse(await self.cache.refresh())
        self.assertEqual(len(self.cache.profiles("chat")), 1)
        self.tick = 300
        with self.assertRaises(Unavailable):
            self.cache.profiles("chat")

    async def test_revocation_and_generation_regression(self):
        self.document["generation"] = 2
        self.document["revoked"] = ["fixture-chat"]
        self.assertTrue(await self.cache.refresh())
        self.assertEqual(self.cache.profiles("chat"), [])
        self.document["generation"] = 3
        self.document["revoked"] = []
        self.assertFalse(await self.cache.refresh())
        self.assertEqual(self.cache.profiles("chat"), [])

    async def test_mutation_and_expiry(self):
        resolved = self.cache.resolve("chat", "fixture-chat")
        resolved.capabilities.append("tools")
        self.assertEqual(self.cache.resolve("chat", "fixture-chat").capabilities, ["text"])
        self.now += timedelta(days=2)
        self.assertEqual(self.cache.profiles("chat"), [])

    async def test_two_replicas_converge_without_extending_outage(self):
        async def fetch():
            return self.document
        other = SnapshotCache(fetch, self.authority.trust)
        await other.refresh()
        self.assertEqual(other.revision, self.cache.revision)
        self.document["generation"] = 2
        self.document["revoked"] = ["fixture-chat"]
        await other.refresh()
        self.assertNotEqual(other.revision, self.cache.revision)
        self.assertEqual(other.profiles("chat"), [])
        await self.cache.refresh()
        self.assertEqual(other.revision, self.cache.revision)