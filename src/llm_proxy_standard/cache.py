"""Async single-flight refresh with bounded stale reads and sticky revocation."""
import asyncio
import time
from .contracts import ExperimentalApproval, digest, timestamp, utcnow, parse_snapshot, snapshot_history, validate_successor


class Unavailable(ValueError):
    pass


class SnapshotCache:
    def __init__(self, fetch, trust, *, interval=30, ttl=300, monotonic=time.monotonic, now=utcnow):
        if not 0 < interval <= 30 or not 0 < ttl <= 300:
            raise ValueError("Refresh/TTL exceeds policy")
        self.fetch, self.trust = fetch, trust
        self.interval, self.ttl, self.monotonic, self.now = interval, ttl, monotonic, now
        self.snapshot, self.updated, self.revision = None, None, None
        self.revoked = set()
        self.lock = asyncio.Lock()
        self.last_refresh_ok = False

    async def refresh(self):
        async with self.lock:
            try:
                async with asyncio.timeout(5):
                    data = await self.fetch()
                candidate = parse_snapshot(data)
                revision = digest(candidate.document())
                for entry in snapshot_history(candidate).values():
                    self.trust.verify(entry, historical=True)
                if self.snapshot:
                    validate_successor(self.snapshot, candidate)
                self.snapshot = candidate.model_copy(deep=True)
                self.revision = revision
                self.revoked.update(candidate.revoked)
                self.updated = self.monotonic()
                self.last_refresh_ok = True
                return True
            except Exception:
                self.last_refresh_ok = False
                return False

    def available(self):
        return self.updated is not None and 0 <= self.monotonic() - self.updated < self.ttl

    def profiles(self, protocol):
        if not self.available():
            raise Unavailable("Registry snapshot unavailable or expired")
        now = self.now()
        result = []
        allowed_artifacts = {self.trust.artifact} | (self.trust.historical_artifacts if getattr(self.trust, "serve_historical_artifacts", False) else set())
        for identifier, entry in self.snapshot.entries.items():
            value = entry.proposal.profile
            if (identifier not in self.revoked and value.artifact in allowed_artifacts and value.protocol == protocol and value.valid(now) and
                    timestamp(entry.approval.issued_at) <= now < timestamp(entry.approval.expires_at) and
                    all(timestamp(item.expires_at) > now for item in entry.proposal.evidence)):
                result.append(value.model_copy(deep=True))
        return sorted(result, key=lambda value: value.id)

    def resolve(self, protocol, identifier):
        for value in self.profiles(protocol):
            if value.id == identifier:
                return value
        raise KeyError("Unknown, expired or revoked model")

    def release_status(self, identifier):
        entry = self.snapshot.entries[identifier]
        return "experimental" if isinstance(entry.approval, ExperimentalApproval) else "qualified"

    async def run(self):
        while True:
            await asyncio.sleep(self.interval)
            await self.refresh()