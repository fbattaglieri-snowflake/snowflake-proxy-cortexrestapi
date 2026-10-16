import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from llm_proxy_standard.contracts import utcnow
from llm_proxy_standard.registry import Registry, Trust
from tests.support import Authority, profile, ACCOUNT, ARTIFACT, UPSTREAM


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.authority = Authority()
        self.registry = Registry(Path(self.directory.name) / "registry.db", self.authority.trust)

    def tearDown(self):
        self.directory.cleanup()

    def test_signed_publish_idempotent_restart_and_revocation(self):
        entry = self.authority.entry()
        revision = self.registry.publish(entry)
        self.assertEqual(revision, self.registry.publish(entry))
        restarted = Registry(self.registry.path, self.authority.trust)
        self.assertEqual(restarted.read().generation, 1)
        restarted.revoke(entry.proposal.profile.id, revision, operator="test-human")
        with self.assertRaises(ValueError):
            self.registry.publish(entry)
        self.assertEqual(restarted.read().generation, 2)

    def test_missing_tampered_expired_wrong_target_approval(self):
        entry = self.authority.entry()
        for field, value in (("account", "other"), ("proposal_sha256", "f" * 64), ("approver", "agent")):
            data = entry.document()
            data["approval"][field] = value
            with self.subTest(field=field), self.assertRaises(Exception):
                self.registry.publish(data)
        with self.assertRaises(ValueError):
            self.registry.publish(entry, now=utcnow() + timedelta(days=2))
        with self.assertRaises(Exception):
            self.registry.publish({"proposal": entry.proposal.document()})

    def test_production_rejects_synthetic_authority(self):
        trust = Trust(self.authority.trust.keys, ACCOUNT, ARTIFACT, UPSTREAM)
        with self.assertRaises(ValueError):
            trust.verify(self.authority.entry())

    def test_cas_conflict_and_missing_coverage(self):
        self.registry.publish(self.authority.entry())
        with self.assertRaises(ValueError):
            self.registry.publish(self.authority.entry(profile("messages")))
        entry = self.authority.entry(profile("messages"), self.registry.read())
        data = entry.document()
        data["proposal"]["evidence"][0]["coverage"] = []
        with self.assertRaises(ValueError):
            self.registry.publish(data)

    def test_protected_fields_rejected(self):
        rule = dict(id="r1", path="messages", operation="set", when="missing", value="bad", reason="test", effects="test")
        with self.assertRaises(ValueError):
            profile(rules=[rule])

    def test_sqlite_permissions_private(self):
        self.assertEqual(self.registry.path.stat().st_mode & 0o777, 0o600)