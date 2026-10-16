"""Local operator CLI. No signing command, no remote administrative HTTP API."""
import argparse
import base64
from pathlib import Path

from .contracts import Entry, Evidence, ExperimentalApproval, canonical, digest, required_coverage, strict_json
from .evidence import import_campaign
from .registry import Registry, Trust


def verify_bundle(entry, bundle):
    """Recompute evidence identities before mutation; never trust coverage labels alone."""
    verified = []
    for source in bundle:
        if source["kind"] == "experimental-record":
            if not isinstance(entry.approval, ExperimentalApproval):
                raise ValueError("Experimental record requires experimental approval")
            raw = Path(source["path"]).read_bytes()
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Experimental record too large")
            report = strict_json(raw)
            if (set(report) != {"schema_version", "profile_sha256", "target", "expires_at", "limitations", "sources"} or
                    report["schema_version"] != 1 or not isinstance(report["sources"], list) or not report["sources"] or
                    report["limitations"] != entry.approval.limitations):
                raise ValueError("Invalid experimental record")
            verified.append(Evidence(kind="experimental-record", sha256=digest(report),
                profile_sha256=report["profile_sha256"], target=report["target"], expires_at=report["expires_at"], coverage=[]))
        elif source["kind"] == "kit-proxy":
            verified.append(import_campaign(source["path"], entry.proposal.profile,
                target=source["target"], expires_at=source["expires_at"], synthetic=entry.proposal.synthetic))
        elif source["kind"] == "reviewed-coverage":
            raw = Path(source["path"]).read_bytes()
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Coverage evidence too large")
            report = strict_json(raw)
            required = required_coverage(entry.proposal.profile)
            if set(report) != {"profile_sha256", "target", "expires_at", "synthetic", "results", "producer"}:
                raise ValueError("Invalid coverage report")
            results = report["results"]
            expected = {(run, case) for run in (1, 2, 3) for case in required}
            actual = [(row["run"], row["case"]) for row in results]
            if (set(actual) != expected or len(actual) != len(expected) or any(row["status"] != "PASS" for row in results) or
                    report["synthetic"] != entry.proposal.synthetic or report["profile_sha256"] != digest(entry.proposal.profile.document()) or
                    not isinstance(report["producer"], str) or not report["producer"]):
                raise ValueError("Coverage incomplete or mismatched")
            verified.append(Evidence(kind="reviewed-coverage", sha256=digest(report),
                profile_sha256=report["profile_sha256"], target=report["target"], expires_at=report["expires_at"],
                coverage=sorted(required), synthetic=report["synthetic"]))
        else:
            raise ValueError("Unknown evidence kind")
    if sorted(canonical(value.document()) for value in verified) != sorted(canonical(value.document()) for value in entry.proposal.evidence):
        raise ValueError("Evidence bytes do not match signed proposal")


def load_trust(path):
    config = strict_json(Path(path).read_bytes())
    return Trust({key: (value["identity"], base64.b64decode(value["public_key"], validate=True))
                  for key, value in config["keys"].items()}, config["account"], config["artifact"], config["upstream"],
                 experimental_entries=config.get("experimental_entries", []),
                 historical_artifacts=config.get("historical_artifacts", []),
                 admin_approvers=config.get("admin_approvers", []),
                 serve_historical_artifacts=config.get("serve_historical_artifacts", False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--trust", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("snapshot")
    publish = commands.add_parser("publish")
    publish.add_argument("--entry", type=Path, required=True)
    publish.add_argument("--bundle", type=Path, required=True)
    revoke = commands.add_parser("revoke")
    revoke.add_argument("--id", required=True)
    revoke.add_argument("--expected-revision", required=True)
    revoke.add_argument("--operator", required=True)
    args = parser.parse_args()
    registry = Registry(args.database, load_trust(args.trust))
    if args.command == "publish":
        entry = Entry.model_validate(strict_json(args.entry.read_bytes()))
        registry.trust.verify(entry)
        verify_bundle(entry, strict_json(args.bundle.read_bytes()))
        print(registry.publish(entry))
    elif args.command == "revoke":
        print(registry.revoke(args.id, args.expected_revision, operator=args.operator))
    else:
        print(canonical(registry.read().document()).decode())


if __name__ == "__main__":
    main()