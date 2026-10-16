"""Single-writer registry. Private signing keys are NEVER loaded by this code."""
import base64
from datetime import timedelta
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .contracts import Entry, ExperimentalApproval, Snapshot, canonical, digest, strict_json, required_coverage, timestamp, utcnow
from .contracts import RevisionProposal, RevisionSnapshot, parse_snapshot, snapshot_history, validate_revision


class Trust:
    def __init__(self, keys, account, artifact, upstream, *, synthetic=False, experimental_entries=None,
                 historical_artifacts=None, admin_approvers=None, serve_historical_artifacts=False):
        self.keys = keys  # id -> (human identity, raw Ed25519 public key)
        self.account, self.artifact, self.upstream = account, artifact, upstream
        self.synthetic = synthetic
        self.serve_historical_artifacts = bool(serve_historical_artifacts)
        self.admin_approvers = frozenset(admin_approvers or ())
        if len(self.admin_approvers) > 16 or any(not isinstance(v, str) or not v for v in self.admin_approvers):
            raise ValueError("Bounded admin approvers required")
        self.historical_artifacts = frozenset(historical_artifacts or ())
        if len(self.historical_artifacts) > 32 or any(
                not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value) for value in self.historical_artifacts):
            raise ValueError("Exact bounded historical artifact digests required")
        self.experimental_entries = frozenset(experimental_entries or ())
        if len(self.experimental_entries) > 100 or any(
                not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value) for value in self.experimental_entries):
            raise ValueError("Explicit bounded experimental entry digests required")

    def verify(self, entry, now=None, *, historical=False):
        now = now or utcnow()
        proposal, approval = entry.proposal, entry.approval
        profile = proposal.profile
        if proposal.synthetic != approval.synthetic or (proposal.synthetic and not self.synthetic):
            raise ValueError("Synthetic release is not allowed")
        if any(item.synthetic != proposal.synthetic for item in proposal.evidence):
            raise ValueError("Mixed synthetic evidence")
        allowed_artifacts = {self.artifact} | (self.historical_artifacts if (historical or self.serve_historical_artifacts) else set())
        if (profile.account != self.account or approval.account != self.account or
                profile.artifact not in allowed_artifacts or proposal.upstream != self.upstream):
            raise ValueError("Release target mismatch")
        if approval.proposal_sha256 != digest(proposal.document()):
            raise ValueError("Approval does not bind this proposal")
        if isinstance(approval, ExperimentalApproval):
            self.verify_experimental(entry, now, historical=historical)
            return
        if any(item.kind == "experimental-record" for item in proposal.evidence):
            raise ValueError("Experimental evidence cannot qualify a signed release")
        identity, raw_key = self.keys[approval.key_id]
        if identity != approval.approver:
            raise ValueError("Approver identity mismatch")
        signed = approval.document()
        signature = base64.b64decode(signed.pop("signature"), validate=True)
        Ed25519PublicKey.from_public_bytes(raw_key).verify(signature, canonical(signed))
        issued, expires = timestamp(approval.issued_at), timestamp(approval.expires_at)
        if issued >= expires:
            raise ValueError("Invalid approval interval")
        check_time = issued if historical else now
        if not historical and not issued <= now < expires:
            raise ValueError("Approval expired or future dated")
        if not profile.valid(check_time):
            raise ValueError("Profile invalid at approval/use time")
        if not proposal.evidence or any(timestamp(item.expires_at) <= check_time or
                item.profile_sha256 != digest(profile.document()) for item in proposal.evidence):
            raise ValueError("Missing, mismatched or expired evidence")
        if not any(item.kind == "kit-proxy" for item in proposal.evidence):
            raise ValueError("Kit proxy campaign required")
        coverage = set().union(*(set(item.coverage) for item in proposal.evidence))
        if not required_coverage(profile) <= coverage:
            raise ValueError("Required coverage missing")
        kit_coverage = set().union(*(set(item.coverage) for item in proposal.evidence if item.kind == "kit-proxy"))
        if not {"kit:" + cap for cap in profile.capabilities if cap != "reasoning"} <= kit_coverage:
            raise ValueError("Kit capability coverage missing")

    def verify_experimental(self, entry, now, *, historical=False):
        proposal, approval = entry.proposal, entry.approval
        is_standard = approval.method == "administrative-standard-v1"
        pinned = digest(entry.document()) in self.experimental_entries or (is_standard and approval.approver in self.admin_approvers)
        if proposal.synthetic or not pinned:
            raise ValueError("Experimental entry not pinned by deployment authority")
        issued, expires = timestamp(approval.issued_at), timestamp(approval.expires_at)
        max_days = 365 if is_standard else 7
        if not timedelta(0) < expires - issued <= timedelta(days=max_days):
            raise ValueError("Experimental approval must expire within seven days" if not is_standard else "Standard approval must expire within 365 days")
        check_time = issued if historical else now
        if not issued <= check_time < expires or not proposal.profile.valid(check_time):
            raise ValueError("Experimental approval/profile expired or future dated")
        if timestamp(proposal.profile.expires_at) > expires:
            raise ValueError("Profile outlives experimental approval")
        if len(proposal.evidence) != 1:
            raise ValueError("One immutable experimental evidence record required")
        evidence = proposal.evidence[0]
        if (evidence.kind != "experimental-record" or evidence.coverage or evidence.synthetic or
                evidence.profile_sha256 != digest(proposal.profile.document()) or
                timestamp(evidence.expires_at) != expires):
            raise ValueError("Experimental evidence cannot claim qualified coverage")


class Registry:
    def __init__(self, path, trust):
        self.path, self.trust = Path(path), trust
        if not self.path.parent.is_dir() or self.path.is_symlink():
            raise ValueError("Dedicated existing registry directory required")
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, action TEXT, identity TEXT, revision TEXT, at TEXT)")
            connection.execute("INSERT OR IGNORE INTO state VALUES (1, ?)", (canonical(Snapshot().document()).decode(),))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def read(self):
        with self.connect() as connection:
            return parse_snapshot(strict_json(connection.execute("SELECT body FROM state WHERE id=1").fetchone()[0]))

    def publish(self, entry, now=None):
        entry = Entry.model_validate(entry) if isinstance(entry, dict) else entry
        self.trust.verify(entry, now)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = parse_snapshot(strict_json(connection.execute("SELECT body FROM state WHERE id=1").fetchone()[0]))
            identifier = entry.proposal.profile.id
            if identifier in state.revoked:
                raise ValueError("Revoked IDs cannot be republished")
            prior = state.entries.get(identifier)
            if prior == entry:
                return digest(state.document())
            if isinstance(state, RevisionSnapshot) and state.history.get(digest(entry.document())) == entry:
                # Inbox rescans must not reactivate an old approved revision.
                return digest(state.document())
            revision_proposal = isinstance(entry.proposal, RevisionProposal)
            if prior is not None and not revision_proposal:
                raise ValueError("Changed profile requires a new ID")
            if digest(state.document()) != entry.proposal.expected_revision:
                raise ValueError("Registry revision conflict")
            data = state.document()
            if revision_proposal:
                if prior is None or digest(prior.document()) != entry.proposal.previous_entry:
                    raise ValueError("Active profile revision conflict")
                history = snapshot_history(state)
                validate_revision(prior, entry, history)
                data.update(schema_version=2, history={key: value.document() for key, value in history.items()},
                            activations=list(state.activations) if isinstance(state, RevisionSnapshot) else [])
                data["history"][digest(entry.document())] = entry.document()
                data["activations"].append(digest(entry.document()))
            elif isinstance(state, RevisionSnapshot):
                data["history"][digest(entry.document())] = entry.document()
            data["entries"][identifier] = entry.document()
            data["generation"] += 1
            return self._save(connection, data, "publish", entry.approval.approver)

    def revoke(self, identifier, expected_revision, *, operator):
        if not operator:
            raise ValueError("Local operator required")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            data = strict_json(connection.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
            if identifier in data["revoked"]:
                return digest(data)
            if digest(data) != expected_revision or identifier not in data["entries"]:
                raise ValueError("Unknown ID or revision conflict")
            data["revoked"].append(identifier)
            data["generation"] += 1
            return self._save(connection, data, "revoke", operator)

    def _save(self, connection, data, action, identity):
        parse_snapshot(data)
        raw = canonical(data)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Registry snapshot size limit")
        revision = digest(data)
        connection.execute("UPDATE state SET body=? WHERE id=1", (raw.decode(),))
        connection.execute("INSERT INTO audit(action,identity,revision,at) VALUES (?,?,?,?)",
                           (action, identity, revision, utcnow().isoformat()))
        return revision