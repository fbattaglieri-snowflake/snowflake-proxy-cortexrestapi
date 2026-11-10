"""Autonomous SPCS Job Worker for Daily Model Scouting, Qualification & Promotion.

Runs inside a Snowpark Container Services (SPCS) compute pool mounting the Registry artifacts stage
at `/artifacts` and using the in-container `/snowflake/session/token` (OAUTH) against
`https://<SNOWFLAKE_HOST>/api/v2/cortex/v1`.
"""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

for p in ("/opt/proxy", "/artifacts/scouting"):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_proxy_standard.contracts import (
    Entry,
    Evidence,
    ExperimentalApproval,
    OptionConstraint,
    ProfileV2,
    Proposal,
    RevisionProposal,
    canonical,
    digest,
    parse_snapshot,
    utcnow,
)
from model_probe.campaign import Campaign
from model_probe.transport import Transport

ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "EXAMPLE_ACCOUNT")
HOST = os.environ.get("SNOWFLAKE_HOST", "example-account.snowflakecomputing.com").replace("_", "-")
GATEWAY_NAME = os.environ.get("PROXY_GATEWAY_NAME", "snowflake").strip()
CORTEX_UPSTREAM = f"https://{HOST}/api/v2/cortex/v1"
GATEWAY_UPSTREAM = f"https://{HOST}/api/v2/aigateways/{GATEWAY_NAME}/v1"
UPSTREAM = os.environ.get("PROXY_UPSTREAM_URL", CORTEX_UPSTREAM)
ARTIFACT_DIGEST = os.environ.get("PROXY_ARTIFACT_DIGEST", "0" * 64)
APPROVER = os.environ.get("PROXY_ADMIN_APPROVER", "operator-admin")
REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://standard-model-registry.namespace.svc.spcs.internal:8090")
PROXY_ORIGIN = os.environ.get("PROXY_TARGET_ORIGIN", "http://llm-proxy-standard.namespace.svc.spcs.internal:8080")
TOKEN_FILE = Path(os.environ.get("SNOWFLAKE_TOKEN_FILE", "/snowflake/session/token"))
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", "/artifacts"))
INBOX_DIR = ARTIFACTS_ROOT / os.environ.get("SNOWFLAKE_INBOX_PREFIX", "inbox")


def build_profile_v2(
    model_id: str,
    protocol: str,
    *,
    revision: str,
    reasoning_none: bool = False,
    supports_re: bool = False,
    capabilities: list[str] | None = None,
    days: int = 365,
) -> ProfileV2:
    now = utcnow()
    expires = (now + timedelta(days=days)).isoformat()
    caps = list(capabilities or ["text", "tools", "parallel_tools", "streaming", "reasoning"])
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
        elif supports_re:
            constraints["reasoning_effort"] = OptionConstraint(kind="enum", values=["none", "low", "medium", "high"])
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
    else:
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

    return ProfileV2(
        schema_version=2,
        id=model_id,
        model=model_id,
        protocol=protocol,
        account=ACCOUNT,
        artifact=ARTIFACT_DIGEST,
        revision=revision,
        capabilities=caps,
        response_stream=stream_mode,
        normalize_client_options=True,
        constraints=constraints,
        created_at=now.isoformat(),
        expires_at=expires,
    )


def fast_probe_oauth(model_id: str, protocol: str, reasoning: str | None = None, upstream_url: str = UPSTREAM) -> dict:
    token = TOKEN_FILE.read_text().strip()
    if protocol == "chat":
        path = "/chat/completions"
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_completion_tokens": 64,
        }
        if reasoning:
            body["reasoning_effort"] = reasoning
    elif protocol == "responses":
        path = "/responses"
        body = {
            "model": model_id,
            "input": [{"role": "user", "content": "Reply with OK."}],
            "max_output_tokens": 256,
        }
    else:
        path = "/messages"
        body = {
            "model": model_id,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Snowflake-Authorization-Token-Type": "OAUTH",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if protocol == "messages":
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(upstream_url + path, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return {"protocol": protocol, "reasoning": reasoning, "reachable": True, "http_status": resp.status}
    except urllib.error.HTTPError as exc:
        return {
            "protocol": protocol,
            "reasoning": reasoning,
            "reachable": False,
            "http_status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace")[:400],
        }
    except Exception as exc:
        return {"protocol": protocol, "reasoning": reasoning, "reachable": False, "error": str(exc)}


def main() -> None:
    req_file = Path(os.environ.get("SCOUT_JOB_REQUEST", "/artifacts/scouting/state/job_request.json"))
    out_file = Path(os.environ.get("SCOUT_JOB_RESULT", "/artifacts/scouting/state/job_result.json"))
    if not req_file.is_file():
        print(json.dumps({"status": "NO_REQUEST_FILE"}))
        return
    job_req = json.loads(req_file.read_text())
    candidates = job_req.get("candidates", [])
    disappeared = job_req.get("disappeared", [])
    repeats = int(job_req.get("repeats", 3))
    os.environ["SNOWFLAKE_ACCOUNT"] = ACCOUNT

    reg_snap = json.loads(urllib.request.urlopen(f"{REGISTRY_URL.rstrip('/')}/snapshot", timeout=10).read())
    snap = parse_snapshot(reg_snap)
    working_doc = snap.document()
    expected_rev = digest(working_doc)
    rev_tag = f"spcs-scout-{utcnow().strftime('%Y%m%d%H%M%S')}"

    tested = []
    promoted = []
    revoked = []
    blocked = []

    for model_id in candidates:
        low = model_id.lower()
        if low.startswith("claude-"):
            order = [("messages", None), ("chat", None)]
        elif low.startswith("openai-gpt-"):
            order = [("responses", None), ("chat", "none"), ("chat", None)]
        else:
            order = [("chat", None), ("chat", "none"), ("responses", None), ("messages", None)]

        chosen = None
        preflight = []
        for proto, reas in order:
            pr = fast_probe_oauth(model_id, proto, reas, UPSTREAM)
            preflight.append(pr)
            if pr.get("reachable"):
                chosen = (proto, reas)
                break

        if not chosen:
            rec = {
                "model": model_id,
                "status": "PROFILE_BLOCKED",
                "promotable": False,
                "reason": "Rejected by all candidate channels (messages, chat, responses) at preflight.",
                "preflight": preflight,
            }
            tested.append(rec)
            blocked.append(rec)
            continue

        proto, reas = chosen
        companion_url = CORTEX_UPSTREAM if UPSTREAM == GATEWAY_UPSTREAM else GATEWAY_UPSTREAM
        companion_probe = fast_probe_oauth(model_id, proto, reas, companion_url)
        gateway_reachable = True if UPSTREAM == GATEWAY_UPSTREAM else bool(companion_probe.get("reachable"))
        cortex_rest_reachable = True if UPSTREAM == CORTEX_UPSTREAM else bool(companion_probe.get("reachable"))

        transport = Transport(UPSTREAM, HOST, token_file=str(TOKEN_FILE), auth_type="OAUTH")
        out_dir = Path("/tmp") / f"qualify_{model_id}_{proto}"
        campaign = Campaign(
            transport,
            model_id,
            proto,
            out_dir,
            repeats=repeats,
            reasoning=reas,
            max_requests=60,
            deadline=1800,
            scope="service-rest",
        )
        summary = campaign.run()
        results = summary.get("results", [])
        core_cases = {
            "text",
            "stream_text",
            "history",
            "single",
            "single_roundtrip",
            "parallel",
            "parallel_roundtrip",
            "stream_parallel",
            "stream_parallel_roundtrip",
            "error_setup",
            "error_roundtrip",
        }
        core_rows = [r for r in results if (r.get("name") or r.get("case")) in core_cases]
        full_suite_ok = summary.get("status") == "PROFILE_QUALIFIED"
        core_ok = (full_suite_ok or (
            len(core_rows) == len(core_cases) * repeats and all(r.get("status") == "PASS" for r in core_rows)
        )) and gateway_reachable and cortex_rest_reachable
        struct_rows = [r for r in results if (r.get("name") or r.get("case")) in {"structured", "structured_negative"}]
        struct_ok = full_suite_ok or (
            len(struct_rows) == 2 * repeats and all(r.get("status") == "PASS" for r in struct_rows)
        )

        rec = {
            "model": model_id,
            "channel": proto,
            "reasoning": reas,
            "status": "PROFILE_QUALIFIED" if core_ok else "PROFILE_BLOCKED",
            "promotable": core_ok,
            "cortex_rest_reachable": cortex_rest_reachable,
            "ai_gateway_reachable": gateway_reachable,
            "counts": summary.get("counts"),
            "preflight": preflight,
        }
        tested.append(rec)
        if not core_ok:
            blocked.append(rec)
        if core_ok:
            caps = ["text", "tools", "parallel_tools", "streaming"]
            if struct_ok:
                caps.append("structured_output")
            supports_re = False
            if proto == "chat" and reas is None:
                supports_re = bool(fast_probe_oauth(model_id, "chat", "low").get("reachable"))
            if proto in ("responses", "messages") or reas == "none" or supports_re:
                caps.append("reasoning")
            profile = build_profile_v2(
                model_id,
                proto,
                revision=rev_tag,
                reasoning_none=(reas == "none"),
                supports_re=supports_re,
                capabilities=caps,
            )
            target = f"{PROXY_ORIGIN.rstrip('/')}/{proto}/v1"
            limitations = [
                "Universal ProfileV2 release with client option normalization and 365-day validity",
                "Automated replay cache enabled for Responses encrypted_content and Messages thinking signatures",
            ]
            report = {
                "schema_version": 1,
                "profile_sha256": digest(profile.document()),
                "target": target,
                "expires_at": profile.expires_at,
                "limitations": limitations,
                "sources": [{"scope": f"automated SPCS scouting ({rev_tag})"}],
            }
            ev = Evidence(
                kind="experimental-record",
                sha256=digest(report),
                profile_sha256=digest(profile.document()),
                target=target,
                expires_at=profile.expires_at,
                coverage=[],
            )
            prior_entry = working_doc.get("entries", {}).get(model_id)
            if prior_entry is not None:
                proposal = RevisionProposal(
                    schema_version=2,
                    profile=profile,
                    upstream=UPSTREAM,
                    expected_revision=expected_rev,
                    previous_entry=digest(prior_entry),
                    change="compatible",
                    evidence=[ev],
                )
            else:
                proposal = Proposal(
                    profile=profile,
                    upstream=UPSTREAM,
                    expected_revision=expected_rev,
                    evidence=[ev],
                )
            approval = ExperimentalApproval(
                method="administrative-standard-v1",
                approver=APPROVER,
                account=ACCOUNT,
                decision="approve-experimental",
                proposal_sha256=digest(proposal.document()),
                issued_at=profile.created_at,
                expires_at=profile.expires_at,
                reason="Automated model promotion via administrative-standard-v1",
                limitations=limitations,
            )
            entry = Entry(proposal=proposal, approval=approval)
            ev_name = f"{profile.id}-{profile.revision}.evidence"
            pkg = {"entry": entry.document(), "bundle": [{"kind": "experimental-record", "path": ev_name}]}
            pkg_name = digest(pkg) + ".json"

            INBOX_DIR.mkdir(parents=True, exist_ok=True)
            (INBOX_DIR / ev_name).write_bytes(canonical(report))
            (INBOX_DIR / pkg_name).write_bytes(canonical(pkg))

            entry_doc = entry.document()
            entry_sha = digest(entry_doc)
            working_doc.setdefault("history", {})[entry_sha] = entry_doc
            if prior_entry is not None:
                working_doc.setdefault("activations", []).append(entry_sha)
            working_doc.setdefault("entries", {})[model_id] = entry_doc
            working_doc["generation"] = int(working_doc.get("generation", 0)) + 1
            expected_rev = digest(working_doc)
            promoted.append({"model": model_id, "protocol": proto, "package": pkg_name, "expires_at": profile.expires_at})

    for rm_id in disappeared:
        if rm_id in working_doc.get("revoked", []):
            continue
        rev_pkg = {
            "revoke": {
                "account": ACCOUNT,
                "approver": APPROVER,
                "expected_revision": expected_rev,
                "id": rm_id,
                "issued_at": utcnow().isoformat(),
                "method": "administrative-standard-v1",
                "reason": f"Automated removal of disappeared model '{rm_id}'",
            }
        }
        rev_name = digest(rev_pkg) + ".json"
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        (INBOX_DIR / rev_name).write_bytes(canonical(rev_pkg))
        working_doc.setdefault("revoked", []).append(rm_id)
        working_doc["generation"] = int(working_doc.get("generation", 0)) + 1
        expected_rev = digest(working_doc)
        revoked.append({"model": rm_id, "package": rev_name})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "tested": tested,
                "promoted": promoted,
                "revoked": revoked,
                "blocked": blocked,
                "expected_generation": working_doc.get("generation"),
                "expected_revision": expected_rev,
            },
            indent=2,
        )
    )
    print(json.dumps({"status": "COMPLETED", "promoted": len(promoted), "revoked": len(revoked), "blocked": len(blocked)}))


if __name__ == "__main__":
    main()
