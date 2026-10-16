import base64
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing

from llm_proxy_standard.contracts import Approval, Entry, Evidence, Proposal, canonical, digest, required_coverage
from llm_proxy_standard.evidence import import_campaign
from llm_proxy_standard.publisher import Publisher, resolve_inside
from llm_proxy_standard.registry import Registry
from tests.support import Authority, profile, UPSTREAM


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inbox, self.checkpoints = self.root / "inbox", self.root / "protected"
        self.inbox.mkdir()
        self.checkpoints.mkdir()
        self.authority = Authority()
        self.registry = Registry(self.root / "registry.db", self.authority.trust)

    def tearDown(self):
        self.temp.cleanup()

    def package(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cortex-model-compatibility-matrix"))
        campaign = importlib.import_module("model_probe.campaign")
        class Transport:
            base = "http://127.0.0.1:8080/chat/v1"
            auth_type = "NONE"
            destination_policy = {"expected_base_url": base, "local_proxy": True}
            def redact(self, value):
                return value
            def send(self, protocol, body):
                return {"choices": [{"message": {"content": "BLUE7" if len(body["messages"]) > 1 else "OK"},
                                     "finish_reason": "stop"}]}, {"http_status": 200}
        value = profile()
        campaign.Campaign(Transport(), value.id, "chat", self.inbox / "campaign", scope="proxy", capabilities=["text"]).run()
        evidence = import_campaign(self.inbox / "campaign", value, target=Transport.base, expires_at=value.expires_at, synthetic=True)
        coverage = {"profile_sha256": digest(value.document()), "target": Transport.base, "expires_at": value.expires_at,
                    "synthetic": True, "producer": "synthetic-tests-only", "results": [
                        {"run": run, "case": case, "status": "PASS"} for run in (1, 2, 3) for case in sorted(required_coverage(value))]}
        (self.inbox / "coverage").write_bytes(canonical(coverage))
        supplemental = Evidence(kind="reviewed-coverage", sha256=digest(coverage), profile_sha256=digest(value.document()),
                                target=Transport.base, expires_at=value.expires_at, coverage=sorted(required_coverage(value)), synthetic=True)
        proposal = Proposal(profile=value, upstream=UPSTREAM, expected_revision=digest(self.registry.read().document()),
                            evidence=[evidence, supplemental], synthetic=True)
        approval = self.authority.entry(value).approval.document()
        approval.pop("signature")
        approval["proposal_sha256"] = digest(proposal.document())
        approval["signature"] = base64.b64encode(self.authority.key.sign(canonical(approval))).decode()
        entry = Entry(proposal=proposal, approval=Approval(**approval))
        bundle = [{"kind": "kit-proxy", "path": "campaign", "target": Transport.base, "expires_at": value.expires_at},
                  {"kind": "reviewed-coverage", "path": "coverage"}]
        raw = canonical({"entry": entry.document(), "bundle": bundle})
        path = self.inbox / (hashlib.sha256(raw).hexdigest() + ".json")
        path.write_bytes(raw)
        return path, entry

    def test_verified_publication_idempotency_audit_and_single_writer(self):
        path, entry = self.package()
        publisher = Publisher(self.registry, self.inbox, self.checkpoints)
        try:
            with self.assertRaises(BlockingIOError):
                Publisher(self.registry, self.inbox, self.checkpoints)
            revision = publisher.consume(path)
            self.assertEqual(publisher.consume(path), revision)
            self.assertEqual(self.registry.read().generation, 1)
            with self.registry.connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM audit").fetchone()[0], 1)
            revoked = self.registry.revoke(entry.proposal.profile.id, revision, operator="test-human")
            publisher.checkpoint(self.registry.read())
            with self.assertRaises(ValueError):
                publisher.consume(path)
        finally:
            publisher.close()
        restarted = Publisher(self.registry, self.inbox, self.checkpoints)
        restarted.close()
        self.assertEqual(digest(self.registry.read().document()), revoked)

    def test_restore_old_database_cannot_resurrect_revoked_profile(self):
        path, entry = self.package()
        publisher = Publisher(self.registry, self.inbox, self.checkpoints)
        publisher.consume(path)
        backup = self.root / "old.sqlite"
        with self.registry.connect() as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        revision = digest(self.registry.read().document())
        self.registry.revoke(entry.proposal.profile.id, revision, operator="test-human")
        publisher.checkpoint(self.registry.read())
        publisher.close()
        self.registry.path.write_bytes(backup.read_bytes())
        with self.assertRaisesRegex(ValueError, "behind protected checkpoint"):
            Publisher(Registry(self.registry.path, self.authority.trust), self.inbox, self.checkpoints)

    def test_reject_modified_evidence_missing_signatures_traversal_and_symlinks(self):
        path, _ = self.package()
        publisher = Publisher(self.registry, self.inbox, self.checkpoints)
        try:
            (self.inbox / "coverage").write_text("{}")
            with self.assertRaises(ValueError):
                publisher.consume(path)
            self.assertEqual(self.registry.read().generation, 0)
            for relative in ("../registry.db", str(self.registry.path)):
                with self.assertRaises(ValueError):
                    resolve_inside(self.inbox, relative)
            (self.inbox / "link").symlink_to(self.registry.path)
            with self.assertRaises(ValueError):
                resolve_inside(self.inbox, "link")
            package = json.loads(path.read_bytes())
            package["entry"].pop("approval")
            raw = canonical(package)
            missing = self.inbox / (hashlib.sha256(raw).hexdigest() + ".json")
            missing.write_bytes(raw)
            with self.assertRaises(ValueError):
                publisher.consume(missing)
        finally:
            publisher.close()