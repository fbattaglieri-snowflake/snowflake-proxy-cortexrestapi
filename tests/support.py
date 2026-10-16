"""Synthetic authority exists exclusively in tests."""
import base64
from datetime import timedelta
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from llm_proxy_standard.contracts import (Profile, Proposal, Approval, Entry, Evidence,
                                        Snapshot, canonical, digest, required_coverage, utcnow)
from llm_proxy_standard.registry import Trust

ACCOUNT, ARTIFACT, UPSTREAM = "fixture-account", "a" * 64, "https://cortex.invalid/api/v2/cortex/v1"


def profile(protocol="chat", **kwargs):
    now = utcnow()
    data = dict(id="fixture-" + protocol, model="fixture-upstream", protocol=protocol,
                account=ACCOUNT, artifact=ARTIFACT, revision="v1", capabilities=["text"],
                created_at=(now - timedelta(hours=1)).isoformat(), expires_at=(now + timedelta(days=1)).isoformat())
    data.update(kwargs)
    return Profile(**data)


class Authority:
    def __init__(self):
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.trust = Trust({"test-key": ("test-human", self.public)}, ACCOUNT, ARTIFACT, UPSTREAM, synthetic=True)

    def entry(self, value=None, state=None):
        value = value or profile()
        proposal = Proposal(profile=value, upstream=UPSTREAM,
            expected_revision=digest((state or Snapshot()).document()), synthetic=True,
            evidence=[Evidence(kind="kit-proxy", sha256="b" * 64, profile_sha256=digest(value.document()),
                target="http://proxy.invalid/chat/v1", expires_at=value.expires_at,
                coverage=sorted(required_coverage(value) | {"kit:" + cap for cap in value.capabilities if cap != "reasoning"}), synthetic=True)])
        now = utcnow()
        approval = dict(key_id="test-key", approver="test-human", proposal_sha256=digest(proposal.document()),
            account=ACCOUNT, decision="approve", issued_at=(now - timedelta(minutes=1)).isoformat(),
            expires_at=value.expires_at, synthetic=True)
        approval["signature"] = base64.b64encode(self.key.sign(canonical(approval))).decode()
        return Entry(proposal=proposal, approval=Approval(**approval))

    def snapshot(self, *profiles):
        entries = [self.entry(value) for value in (profiles or (profile(),))]
        return Snapshot(generation=1, entries={entry.proposal.profile.id: entry for entry in entries})