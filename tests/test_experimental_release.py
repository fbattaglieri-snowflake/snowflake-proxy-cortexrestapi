import asyncio
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from llm_proxy_standard.admin import verify_bundle
from llm_proxy_standard.cache import SnapshotCache
from llm_proxy_standard.contracts import Entry, Evidence, ExperimentalApproval, Proposal, Snapshot, canonical, digest, utcnow
from llm_proxy_standard.publisher import Publisher
from llm_proxy_standard.registry import Registry, Trust
from tests.support import ACCOUNT, ARTIFACT, UPSTREAM, Authority, profile


def experimental(value=None):
    value = value or profile()
    report = {"schema_version": 1, "profile_sha256": digest(value.document()), "target": "http://private/chat/v1",
              "expires_at": value.expires_at, "limitations": ["Not qualified"], "sources": [{"description": "synthetic fixture only"}]}
    evidence = Evidence(kind="experimental-record", sha256=digest(report), profile_sha256=digest(value.document()),
                        target=report["target"], expires_at=value.expires_at, coverage=[])
    proposal = Proposal(profile=value, upstream=UPSTREAM, expected_revision=digest(Snapshot().document()), evidence=[evidence])
    approval = ExperimentalApproval(method="administrative-experimental-v1", approver="fixture-admin", account=ACCOUNT,
        decision="approve-experimental", issued_at=utcnow().isoformat(), expires_at=value.expires_at,
        proposal_sha256=digest(proposal.document()), reason="Explicit test fixture authorization", limitations=report["limitations"])
    return Entry(proposal=proposal, approval=approval), report


def trust_for(entry):
    return Trust({}, ACCOUNT, ARTIFACT, UPSTREAM, experimental_entries=[digest(entry.document())])


class ExperimentalTests(unittest.TestCase):
    def test_no_pin_no_admission_and_exact_pin_binding(self):
        entry, _ = experimental()
        with self.assertRaisesRegex(ValueError, "not pinned"):
            Trust({}, ACCOUNT, ARTIFACT, UPSTREAM).verify(entry)
        trust_for(entry).verify(Entry.model_validate(entry.document()))
        changed = entry.model_copy(update={"approval": entry.approval.model_copy(update={"approver": "other-admin"})})
        with self.assertRaises(ValueError):
            trust_for(entry).verify(changed)
        with self.assertRaises(ValueError):
            Trust({}, "other-account", ARTIFACT, UPSTREAM, experimental_entries=[digest(entry.document())]).verify(entry)

    def test_expiry_and_no_qualified_coverage(self):
        entry, _ = experimental()
        with self.assertRaises(ValueError):
            trust_for(entry).verify(entry, now=utcnow() + timedelta(days=8))
        long, _ = experimental(profile(expires_at=(utcnow() + timedelta(days=9)).isoformat()))
        with self.assertRaisesRegex(ValueError, "seven days"):
            trust_for(long).verify(long)
        altered = entry.proposal.evidence[0].model_copy(update={"coverage": ["kit:text"]})
        proposal = entry.proposal.model_copy(update={"evidence": [altered]})
        changed = entry.model_copy(update={"proposal": proposal, "approval": entry.approval.model_copy(update={"proposal_sha256": digest(proposal.document())})})
        with self.assertRaisesRegex(ValueError, "qualified coverage"):
            trust_for(changed).verify(changed)
        signed = Authority().entry()
        with self.assertRaises((ValueError, KeyError)):
            trust_for(entry).verify(signed)

    def test_publisher_restart_revocation_cache_and_tampered_evidence(self):
        entry, report = experimental()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inbox, checkpoints = root / "inbox", root / "checkpoints"
            inbox.mkdir()
            checkpoints.mkdir()
            record = inbox / "record.data"
            record.write_bytes(canonical(report))
            bundle = [{"kind": "experimental-record", "path": "record.data"}]
            package = {"entry": entry.document(), "bundle": bundle}
            path = inbox / (digest(package) + ".json")
            path.write_bytes(canonical(package))
            registry = Registry(root / "registry.sqlite", trust_for(entry))
            publisher = Publisher(registry, inbox, checkpoints)
            try:
                revision = publisher.consume(path)
                self.assertEqual(publisher.consume(path), revision)
                async def fetch():
                    return registry.read().document()
                cache = SnapshotCache(fetch, trust_for(entry))
                self.assertTrue(asyncio.run(cache.refresh()))
                self.assertEqual(cache.release_status(entry.proposal.profile.id), "experimental")
                registry.revoke(entry.proposal.profile.id, revision, operator="fixture-admin")
                publisher.checkpoint(registry.read())
                self.assertTrue(asyncio.run(cache.refresh()))
                self.assertEqual(cache.profiles("chat"), [])
                with self.assertRaises(ValueError):
                    publisher.consume(path)
                record.write_text("{}")
                with self.assertRaises(ValueError):
                    verify_bundle(entry, [{"kind": "experimental-record", "path": str(record)}])
            finally:
                publisher.close()
            restarted = Publisher(registry, inbox, checkpoints)
            restarted.close()