"""Zero-touch model promotion CLI for Snowflake Cortex REST API Proxy & Registry.

Usage:
  python scripts/promote_model.py \
    --model <model-id> \
    --protocol <chat|responses|messages> \
    --account <ACCOUNT_LOCATOR> \
    --upstream https://<org>-<account>.snowflakecomputing.com/api/v2/cortex/v1 \
    --artifact <64-hex-artifact-digest> \
    --expected-revision <64-hex-registry-revision> \
    --approver <APPROVER_IDENTITY> \
    --output-dir ./out-package
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_proxy_standard.contracts import (
    Entry, Evidence, ExperimentalApproval, OptionConstraint, ProfileV2,
    Proposal, RevisionProposal, canonical, digest, utcnow,
)

DEFAULT_CAPS = ["text", "tools", "parallel_tools", "streaming", "reasoning"]


def build_universal_profile(model_id, protocol, artifact, *, account="EXAMPLE_ACCOUNT",
                            revision="universal-v2", reasoning_none=False,
                            capabilities=None, now=None, days=365):
    now = now or utcnow()
    expires = (now + timedelta(days=days)).isoformat()
    caps = list(capabilities or DEFAULT_CAPS)
    if protocol == "chat":
        constraints = {
            "max_completion_tokens": OptionConstraint(kind="integer", minimum=1, maximum=128000),
            "max_tokens": OptionConstraint(kind="integer", minimum=1, maximum=128000),
            "temperature": OptionConstraint(kind="number", minimum=0.0, maximum=2.0),
            "top_p": OptionConstraint(kind="number", minimum=0.0, maximum=1.0),
            "parallel_tool_calls": OptionConstraint(kind="enum", values=[True, False]),
            "stream_options.include_usage": OptionConstraint(kind="enum", values=[True, False]),
        }
        if reasoning_none:
            constraints["reasoning_effort"] = OptionConstraint(kind="enum", values=["none"])
        stream_mode = "native"
    elif protocol == "responses":
        constraints = {
            "max_output_tokens": OptionConstraint(kind="integer", minimum=1, maximum=128000),
            "store": OptionConstraint(kind="enum", values=[False]),
            "include": OptionConstraint(kind="enum", values=[["reasoning.encrypted_content"], []]),
            "prompt_cache_key": OptionConstraint(kind="string", max_length=128),
            "parallel_tool_calls": OptionConstraint(kind="enum", values=[True, False]),
            "reasoning.effort": OptionConstraint(kind="enum", values=["low", "medium", "high"]),
            "reasoning.summary": OptionConstraint(kind="enum", values=["auto", "concise", "detailed"]),
        }
        stream_mode = "ordered-items"
    elif protocol == "messages":
        constraints = {
            "max_tokens": OptionConstraint(kind="integer", minimum=1, maximum=128000),
            "temperature": OptionConstraint(kind="number", minimum=0.0, maximum=1.0),
            "top_p": OptionConstraint(kind="number", minimum=0.0, maximum=1.0),
            "thinking.type": OptionConstraint(kind="enum", values=["enabled", "disabled"]),
            "thinking.budget_tokens": OptionConstraint(kind="integer", minimum=1024, maximum=64000),
            "output_config.effort": OptionConstraint(kind="enum", values=["low", "medium", "high"]),
            "tool_choice.type": OptionConstraint(kind="enum", values=["auto", "none"]),
            "tool_choice.disable_parallel_tool_use": OptionConstraint(kind="enum", values=[True, False]),
        }
        stream_mode = "native"
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return ProfileV2(
        schema_version=2,
        id=model_id,
        model=model_id,
        protocol=protocol,
        account=account,
        artifact=artifact,
        revision=revision,
        capabilities=caps,
        response_stream=stream_mode,
        normalize_client_options=True,
        constraints=constraints,
        created_at=now.isoformat(),
        expires_at=expires,
    )


def build_package_for_profile(profile, expected_revision, *,
                              upstream="https://example-account.snowflakecomputing.com/api/v2/cortex/v1",
                              approver="operator-admin",
                              target_origin="http://llm-proxy-standard.namespace.svc.spcs.internal:8080",
                              previous_entry=None, change="compatible", sources=None):
    target = f"{target_origin.rstrip('/')}/{profile.protocol}/v1"
    limitations = [
        "Universal ProfileV2 release with client option normalization and 365-day validity",
        "Automated replay cache enabled for Responses encrypted_content and Messages thinking signatures",
    ]
    sources = sources or [{"scope": "qualified model promotion"}]
    report = {
        "schema_version": 1,
        "profile_sha256": digest(profile.document()),
        "target": target,
        "expires_at": profile.expires_at,
        "limitations": limitations,
        "sources": sources,
    }
    evidence = Evidence(
        kind="experimental-record",
        sha256=digest(report),
        profile_sha256=digest(profile.document()),
        target=target,
        expires_at=profile.expires_at,
        coverage=[],
    )
    if previous_entry is not None:
        proposal = RevisionProposal(
            schema_version=2,
            profile=profile,
            upstream=upstream,
            expected_revision=expected_revision,
            previous_entry=previous_entry,
            change=change,
            evidence=[evidence],
        )
    else:
        proposal = Proposal(
            profile=profile,
            upstream=upstream,
            expected_revision=expected_revision,
            evidence=[evidence],
        )
    approval = ExperimentalApproval(
        method="administrative-standard-v1",
        approver=approver,
        account=profile.account,
        decision="approve-experimental",
        proposal_sha256=digest(proposal.document()),
        issued_at=profile.created_at,
        expires_at=profile.expires_at,
        reason="Explicit operator model promotion via administrative-standard-v1",
        limitations=limitations,
    )
    entry = Entry(proposal=proposal, approval=approval)
    evidence_name = f"{profile.id}-{profile.revision}.evidence"
    package = {"entry": entry.document(), "bundle": [{"kind": "experimental-record", "path": evidence_name}]}
    package_name = digest(package) + ".json"
    return entry, report, evidence_name, package, package_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True, choices=("chat", "responses", "messages"))
    parser.add_argument("--account", required=True, help="Snowflake account locator")
    parser.add_argument("--upstream", required=True, help="https://<host>/api/v2/cortex/v1")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--previous-entry", default=None)
    parser.add_argument("--reasoning-none", action="store_true")
    parser.add_argument("--revision", default="promoted-v2")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-inbox", default=None, help="Optional Snowflake stage inbox e.g. @DB.SCHEMA.STAGE/inbox")
    parser.add_argument("--connection", default=None, help="Snowflake CLI connection name for stage upload")
    args = parser.parse_args()

    profile = build_universal_profile(
        args.model, args.protocol, args.artifact,
        account=args.account, revision=args.revision,
        reasoning_none=args.reasoning_none, days=args.days,
    )
    entry, report, evidence_name, package, package_name = build_package_for_profile(
        profile, args.expected_revision,
        upstream=args.upstream, approver=args.approver,
        previous_entry=args.previous_entry,
    )
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    ev_path = args.output_dir / evidence_name
    pkg_path = args.output_dir / package_name
    with os.fdopen(os.open(ev_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "wb") as f:
        f.write(canonical(report))
    with os.fdopen(os.open(pkg_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "wb") as f:
        f.write(canonical(package))

    if args.stage_inbox:
        for p in (ev_path, pkg_path):
            cmd = ["snow", "sql", "-q", f"PUT 'file://{p}' {args.stage_inbox.rstrip('/')}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"]
            if args.connection:
                cmd.extend(["--connection", args.connection])
            subprocess.check_call(cmd)

    print(json.dumps({
        "model": profile.id,
        "protocol": profile.protocol,
        "expires_at": profile.expires_at,
        "entry_sha256": digest(entry.document()),
        "package": package_name,
        "evidence": evidence_name,
        "uploaded": bool(args.stage_inbox),
    }))


if __name__ == "__main__":
    main()
