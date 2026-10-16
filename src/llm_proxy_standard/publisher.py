"""Single local writer consuming signed, content-addressed stage bundles.

Stage paths are inputs, never executable commands. No HTTP administration route.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import time

from .admin import load_trust, verify_bundle
from .contracts import Entry, Snapshot, canonical, digest, strict_json
from .contracts import parse_snapshot, snapshot_history, validate_successor
from .registry import Registry


def bounded_file(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular non-symlink artifact required")
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Artifact exceeds size limit")
    return value


def resolve_inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts):
        raise ValueError("Bundle paths must be relative")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink in evidence path")
    current.resolve(strict=True).relative_to(root.resolve(strict=True))
    return current


class Publisher:
    def __init__(self, registry, inbox, checkpoints):
        self.registry, self.inbox, self.checkpoints = registry, Path(inbox), Path(checkpoints)
        if any(not path.is_dir() or path.is_symlink() for path in (self.inbox, self.checkpoints)):
            raise ValueError("Existing dedicated inbox and protected checkpoint directories required")
        lock_path = registry.path.with_suffix(".writer.lock")
        self.lock = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock)
            raise
        try:
            self.verify_floor()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def verify_floor(self):
        state = self.registry.read()
        candidates = sorted(self.checkpoints.glob("checkpoint-*.json"))
        if len(candidates) > 10000:
            raise ValueError("Checkpoint inventory exceeds bound")
        previous = Snapshot()
        for path in candidates:
            document = strict_json(bounded_file(path, 4 * 1024 * 1024))
            snapshot = parse_snapshot(document)
            if path.name != f"checkpoint-{snapshot.generation:020d}-{digest(document)}.json":
                raise ValueError("Checkpoint filename/content mismatch")
            try:
                validate_successor(previous, snapshot)
            except ValueError:
                raise ValueError("Checkpoint regression")
            for entry in snapshot_history(snapshot).values():
                self.registry.trust.verify(entry, historical=True)
            previous = snapshot
        try:
            validate_successor(previous, state)
        except ValueError:
            raise ValueError("Registry behind protected checkpoint; explicit recovery required")
        # A crash can leave valid committed entries ahead of the last checkpoint.
        # Verify them before creating a durable trusted floor from that state.
        for entry in snapshot_history(state).values():
            self.registry.trust.verify(entry, historical=True)
        self.checkpoint(state)

    def checkpoint(self, state):
        raw = canonical(state.document())
        path = self.checkpoints / f"checkpoint-{state.generation:020d}-{digest(state.document())}.json"
        if path.exists():
            if bounded_file(path, 4 * 1024 * 1024) != raw:
                raise ValueError("Immutable checkpoint conflict")
            return
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())

    def consume(self, path):
        raw = bounded_file(path, 4 * 1024 * 1024)
        identifier = hashlib.sha256(raw).hexdigest()
        if path.name != identifier + ".json":
            raise ValueError("Publication request is not content addressed")
        package = strict_json(raw)
        if set(package) != {"entry", "bundle"} or not isinstance(package["bundle"], list):
            raise ValueError("Invalid publication package")
        entry = Entry.model_validate(package["entry"])
        self.registry.trust.verify(entry)
        bundle = []
        for source in package["bundle"]:
            resolved = resolve_inside(self.inbox, source["path"])
            # Imported directories are fully bounded before verification.
            if resolved.is_dir():
                children = list(resolved.iterdir())
                if len(children) > 128:
                    raise ValueError("Evidence directory exceeds file bound")
                for child in children:
                    bounded_file(child, 8 * 1024 * 1024)
            else:
                bounded_file(resolved, 4 * 1024 * 1024)
            bundle.append({**source, "path": str(resolved)})
        verify_bundle(entry, bundle)
        revision = self.registry.publish(entry)
        self.checkpoint(self.registry.read())
        return revision

    def scan(self):
        paths = sorted(self.inbox.glob("*.json"))
        if len(paths) > 256:
            raise ValueError("Inbox exceeds scan bound")
        outcomes = {}
        remaining = list(paths)
        while remaining:
            progressed = False
            next_remaining = []
            for path in remaining:
                try:
                    revision = self.consume(path)
                    outcomes[path] = {"package": path.stem, "status": "accepted", "revision": revision}
                    progressed = True
                except Exception as error:
                    # Error class only: never log evidence or arbitrary untrusted paths.
                    outcomes[path] = {"package": hashlib.sha256(path.name.encode()).hexdigest(),
                                      "status": "rejected", "error_type": type(error).__name__}
                    next_remaining.append(path)
            if not progressed:
                break
            remaining = next_remaining
        return [outcomes[path] for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--trust", type=Path, required=True)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    publisher = Publisher(Registry(args.database, load_trust(args.trust)), args.inbox, args.checkpoints)
    try:
        while True:
            print(canonical({"publication_results": publisher.scan()}).decode(), flush=True)
            if args.once:
                break
            time.sleep(30)
    finally:
        publisher.close()


if __name__ == "__main__":
    main()