"""Fresh approvals for every activation, including rollback; offline only."""
import base64
from datetime import timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing

from llm_proxy_standard.contracts import (Approval, Entry, RevisionProposal, RevisionSnapshot,
    canonical, digest, parse_snapshot, utcnow)
from llm_proxy_standard.cache import SnapshotCache
from llm_proxy_standard.registry import Registry
from llm_proxy_standard.publisher import Publisher
from llm_proxy_standard.registry_server import read_snapshot
from tests.support import Authority, profile


def revision_entry(authority, state, value, change="compatible"):
    template = authority.entry(value, state)
    data = template.proposal.document()
    data.update(schema_version=2, previous_entry=digest(state.entries[value.id].document()), change=change)
    proposal = RevisionProposal.model_validate(data)
    approval = template.approval.document()
    approval.pop("signature")
    approval["proposal_sha256"] = digest(proposal.document())
    approval["signature"] = base64.b64encode(authority.key.sign(canonical(approval))).decode()
    return Entry(proposal=proposal, approval=Approval(**approval))


class RevisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.authority = Authority()
        self.registry = Registry(self.root / "registry.sqlite", self.authority.trust)
        self.original = profile("messages", options={"max_tokens": [2048]})
        self.registry.publish(self.authority.entry(self.original))
        (self.root / "inbox").mkdir()
        (self.root / "checkpoints").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def wider(self):
        return self.original.model_copy(update={"revision": "v2", "options": {"max_tokens": [2048, 4096]}})

    async def test_update_cache_restart_rollback_and_revocation(self):
        async def fetch():
            return self.registry.read().document()
        cache = SnapshotCache(fetch, self.authority.trust)
        self.assertTrue(await cache.refresh())
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        update = revision_entry(self.authority, self.registry.read(), self.wider())
        initial_entry = self.registry.read().entries[self.original.id]
        new_revision = self.registry.publish(update)
        self.assertEqual(new_revision, self.registry.publish(update))
        self.assertEqual(new_revision, self.registry.publish(initial_entry))
        self.assertEqual(self.registry.read().entries[self.original.id], update)
        publisher.checkpoint(self.registry.read())
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.resolve("messages", self.original.id).revision, "v2")
        self.assertEqual(len(cache.profiles("messages")), 1)
        state = read_snapshot(self.registry.path)
        self.assertIsInstance(state, RevisionSnapshot)
        self.assertEqual(len(state.history), 2)
        publisher.close()
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        rollback = revision_entry(self.authority, state, self.original, "rollback")
        self.registry.publish(rollback)
        publisher.checkpoint(self.registry.read())
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.resolve("messages", self.original.id), self.original)
        self.assertEqual(len(self.registry.read().history), 3)
        publisher.close()
        self.registry.revoke(self.original.id, digest(self.registry.read().document()), operator="test-human")
        with self.assertRaises(ValueError):
            self.registry.publish(revision_entry(self.authority, self.registry.read(), self.wider()))
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.profiles("messages"), [])

    async def test_incompatible_expired_tampered_and_stale_are_rejected(self):
        state = self.registry.read()
        for change in ({"model": "other"}, {"options": {"max_tokens": [4096]}}, {"revision": "v1"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.registry.publish(revision_entry(self.authority, state, self.wider().model_copy(update=change)))
        update = revision_entry(self.authority, state, self.wider())
        corrupted = update.document()
        corrupted["proposal"]["change"] = "rollback"
        with self.assertRaises(ValueError):
            self.registry.publish(corrupted)
        with self.assertRaises(ValueError):
            self.registry.publish(update, now=utcnow() + timedelta(days=3))
        self.registry.publish(update)
        stale = revision_entry(self.authority, state, self.wider().model_copy(update={"revision": "v3"}))
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.registry.publish(stale)
        data = self.registry.read().document()
        data["entries"][self.original.id] = state.entries[self.original.id].document()
        with self.assertRaises(ValueError):
            parse_snapshot(data)

    async def test_old_database_and_trimmed_chain_rejected(self):
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        backup = self.root / "backup.sqlite"
        with self.registry.connect() as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        self.registry.publish(revision_entry(self.authority, self.registry.read(), self.wider()))
        publisher.checkpoint(self.registry.read())
        publisher.close()
        data = self.registry.read().document()
        data["activations"] = []
        with self.assertRaises(ValueError):
            parse_snapshot(data)
        self.registry.path.write_bytes(backup.read_bytes())
        with self.assertRaisesRegex(ValueError, "behind protected checkpoint"):
            Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")

    async def test_new_root_after_migration_and_no_forged_rollback(self):
        self.registry.publish(revision_entry(self.authority, self.registry.read(), self.wider()))
        another = profile("chat")
        self.registry.publish(self.authority.entry(another, self.registry.read()))
        self.assertEqual(len(self.registry.read().entries), 2)
        never_active = self.original.model_copy(update={"revision": "never-active"})
        with self.assertRaises(ValueError):
            self.registry.publish(revision_entry(self.authority, self.registry.read(), never_active, "rollback"))

    async def test_stream_mode_change_requires_explicit_new_id(self):
        from llm_proxy_standard.contracts import ProfileV2, compatible_profiles
        previous = profile("responses", capabilities=["text", "streaming"])
        current = ProfileV2.model_validate({**previous.document(), "schema_version": 2,
                                           "revision": "v2", "response_stream": "ordered-items"})
        self.assertFalse(compatible_profiles(previous, current))

    async def test_administrative_pins_required_for_each_revision(self):
        from tests.test_experimental_release import experimental
        from llm_proxy_standard.registry import Trust
        from tests.support import ACCOUNT, ARTIFACT, UPSTREAM
        original, _ = experimental(profile("chat", options={"max_tokens": [2048]}))
        trust = Trust({}, ACCOUNT, ARTIFACT, UPSTREAM, experimental_entries=[digest(original.document())])
        registry = Registry(self.root / "experimental.sqlite", trust)
        registry.publish(original)
        updated, _ = experimental(original.proposal.profile.model_copy(update={"revision": "v2", "options": {"max_tokens": [2048, 4096]}}))
        proposal = RevisionProposal.model_validate({**updated.proposal.document(), "schema_version": 2,
            "previous_entry": digest(original.document()), "expected_revision": digest(registry.read().document()), "change": "compatible"})
        updated = Entry(proposal=proposal, approval=updated.approval.model_copy(update={"proposal_sha256": digest(proposal.document())}))
        with self.assertRaisesRegex(ValueError, "not pinned"):
            registry.publish(updated)
        trust.experimental_entries = frozenset([digest(original.document()), digest(updated.document())])
        registry.publish(updated)
        async def fetch():
            return registry.read().document()
        cache = SnapshotCache(fetch, trust)
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.release_status(original.proposal.profile.id), "experimental")
        self.assertEqual(cache.resolve("chat", original.proposal.profile.id).revision, "v2")

    async def test_artifact_upgrade_is_explicit_and_history_cannot_serve(self):
        from llm_proxy_standard.registry import Trust
        from tests.support import ACCOUNT, ARTIFACT, UPSTREAM
        next_artifact = "c" * 64
        trust = Trust(self.authority.trust.keys, ACCOUNT, next_artifact, UPSTREAM,
                      synthetic=True, historical_artifacts=[ARTIFACT])
        current = self.registry.read()
        old_entry = current.entries[self.original.id]
        trust.verify(old_entry, historical=True)
        with self.assertRaises(ValueError):
            trust.verify(old_entry)
        async def fetch():
            return self.registry.read().document()
        cache = SnapshotCache(fetch, trust)
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.profiles("messages"), [])
        self.registry.trust = trust
        candidate = self.original.model_copy(update={"artifact": next_artifact, "revision": "new-artifact"})
        wrong = candidate.model_copy(update={"options": {"max_tokens": [128000]}})
        with self.assertRaises(ValueError):
            self.registry.publish(revision_entry(self.authority, current, wrong, "artifact"))
        self.registry.publish(revision_entry(self.authority, current, candidate, "artifact"))
        self.assertTrue(await cache.refresh())
        self.assertEqual(cache.resolve("messages", self.original.id).artifact, next_artifact)
        self.assertEqual(len(self.registry.read().history), 2)
        # A rollback to an old artifact cannot be activated while serving the new one.
        with self.assertRaises(ValueError):
            self.registry.publish(revision_entry(self.authority, self.registry.read(), self.original, "rollback"))

    async def test_v1_cannot_hide_revision_without_ancestry(self):
        updated = revision_entry(self.authority, self.registry.read(), self.wider())
        forged = {"schema_version": 1, "generation": 2,
                  "entries": {self.original.id: updated.document()}, "revoked": []}
        with self.assertRaises(ValueError):
            parse_snapshot(forged)

    async def test_uncheckpointed_state_must_be_verified_before_checkpoint(self):
        # Simulate a corrupt DB ahead of the retained floor, not a valid crash.
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        publisher.close()
        updated = revision_entry(self.authority, self.registry.read(), self.wider())
        self.registry.publish(updated)
        state = self.registry.read().document()
        old_digest = digest(updated.document())
        corrupted = updated.document()
        corrupted["approval"]["signature"] = base64.b64encode(b"invalid-signature").decode()
        new_digest = digest(corrupted)
        state["entries"][self.original.id] = corrupted
        del state["history"][old_digest]
        state["history"][new_digest] = corrupted
        state["activations"] = [new_digest]
        with self.registry.connect() as connection:
            connection.execute("UPDATE state SET body=? WHERE id=1", (canonical(state).decode(),))
        before = set((self.root / "checkpoints").iterdir())
        with self.assertRaises(Exception):
            Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        self.assertEqual(set((self.root / "checkpoints").iterdir()), before)

    async def test_artifact_cannot_reuse_earlier_revision_label(self):
        from llm_proxy_standard.registry import Trust
        from tests.support import ACCOUNT, ARTIFACT, UPSTREAM
        self.registry.publish(revision_entry(self.authority, self.registry.read(), self.wider()))
        self.registry.trust = Trust(self.authority.trust.keys, ACCOUNT, "c" * 64, UPSTREAM,
                                    synthetic=True, historical_artifacts=[ARTIFACT])
        candidate = self.wider().model_copy(update={"artifact": "c" * 64, "revision": "v1"})
        with self.assertRaisesRegex(ValueError, "revision identity reused"):
            self.registry.publish(revision_entry(self.authority, self.registry.read(), candidate, "artifact"))

    async def test_coordinated_artifact_rollback_restart_and_revocation(self):
        from llm_proxy_standard.registry import Trust
        from tests.support import ACCOUNT, ARTIFACT, UPSTREAM
        newer = "c" * 64
        new_trust = Trust(self.authority.trust.keys, ACCOUNT, newer, UPSTREAM,
                          synthetic=True, historical_artifacts=[ARTIFACT])
        old_trust = Trust(self.authority.trust.keys, ACCOUNT, ARTIFACT, UPSTREAM,
                          synthetic=True, historical_artifacts=[newer])
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        publisher.close()
        self.registry.trust = new_trust
        candidate = self.original.model_copy(update={"artifact": newer, "revision": "artifact-v2"})
        self.registry.publish(revision_entry(self.authority, self.registry.read(), candidate, "artifact"))
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        publisher.close()
        async def fetch():
            return self.registry.read().document()
        old_cache, new_cache = SnapshotCache(fetch, old_trust), SnapshotCache(fetch, new_trust)
        self.assertTrue(await old_cache.refresh())
        self.assertTrue(await new_cache.refresh())
        self.assertEqual(old_cache.profiles("messages"), [])
        self.assertEqual(len(new_cache.profiles("messages")), 1)
        self.registry.trust = old_trust
        rollback = revision_entry(self.authority, self.registry.read(), self.original, "rollback")
        self.registry.publish(rollback)
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        publisher.close()
        self.assertTrue(await old_cache.refresh())
        self.assertTrue(await new_cache.refresh())
        self.assertEqual(old_cache.resolve("messages", self.original.id), self.original)
        self.assertEqual(new_cache.profiles("messages"), [])
        self.registry.revoke(self.original.id, digest(self.registry.read().document()), operator="test-human")
        publisher = Publisher(self.registry, self.root / "inbox", self.root / "checkpoints")
        publisher.close()
        restarted = SnapshotCache(fetch, old_trust)
        self.assertTrue(await restarted.refresh())
        self.assertEqual(restarted.profiles("messages"), [])