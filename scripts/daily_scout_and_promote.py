"""Automated Daily Cortex Model Scouting, Multi-Channel Qualification, Promotion & Removal.

Compares the daily `SHOW MODELS IN ACCOUNT` snapshot against the previous snapshot,
qualifies newly added `CORTEX_BASE` models across their candidate Cortex REST API channels
(`messages`, `chat`, `responses`), promotes passing models via the Single-Writer Registry stage
inbox, automatically revokes active models that disappear from the account, persists a structured
report for every run, and sends an email notification ONLY when an action occurs (`action_taken = True`).

All environment-specific parameters (account locator, host, stage, schema, notification integration,
and recipient email) are supplied via CLI arguments or environment variables — no account identifiers
or credentials are hardcoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm_proxy_standard.contracts import (
    RevisionSnapshot,
    canonical,
    digest,
    parse_snapshot,
    strict_json,
    utcnow,
)
from promote_model import (
    build_package_for_profile,
    build_revoke_package,
    build_universal_profile,
)


class ScoutConfig:
    def __init__(self, args: argparse.Namespace) -> None:
        self.account: str = args.account or os.environ.get("SNOWFLAKE_ACCOUNT", "EXAMPLE_ACCOUNT")
        self.host: str = args.host or os.environ.get("SNOWFLAKE_HOST", "example-account.snowflakecomputing.com")
        self.connection: str | None = args.connection or os.environ.get("SNOWFLAKE_CONNECTION")
        self.stage: str = (args.stage or os.environ.get("SNOWFLAKE_STAGE", "@LLM_PROXY_DB.CORE.LLM_PROXY_ARTIFACTS")).rstrip("/")
        self.inbox_prefix: str = (args.inbox_prefix or os.environ.get("SNOWFLAKE_INBOX_PREFIX", "inbox")).strip("/")
        self.scouting_prefix: str = (args.scouting_prefix or os.environ.get("SNOWFLAKE_SCOUTING_PREFIX", "scouting")).strip("/")
        self.scout_schema: str = args.scout_schema or os.environ.get("SNOWFLAKE_SCOUT_SCHEMA", "LLM_PROXY_DB.PUBLIC")
        self.artifact_digest: str = args.artifact or os.environ.get("PROXY_ARTIFACT_DIGEST", "0" * 64)
        self.approver: str = args.approver or os.environ.get("PROXY_ADMIN_APPROVER", "operator-admin")
        self.target_origin: str = args.target_origin or os.environ.get(
            "PROXY_TARGET_ORIGIN", "http://llm-proxy-standard.namespace.svc.spcs.internal:8080"
        )
        self.notification_integration: str | None = (
            args.notification_integration or os.environ.get("SNOWFLAKE_NOTIFICATION_INTEGRATION")
        )
        self.notification_email: str | None = args.notification_email or os.environ.get("SNOWFLAKE_NOTIFICATION_EMAIL")
        self.state_dir: Path = Path(args.state_dir or (ROOT / "scouting" / "state"))
        self.reports_dir: Path = Path(args.reports_dir or (ROOT / "scouting" / "reports"))
        self.runs_dir: Path = Path(args.runs_dir or (ROOT / "scouting" / "runs"))
        self.repeats: int = int(args.repeats)
        self.notify: bool = not bool(args.no_notify)
        self.retest_blocked: bool = bool(args.retest_blocked)
        self.force_models: list[str] = list(args.force_models or [])

    @property
    def upstream_url(self) -> str:
        return f"https://{self.host}/api/v2/cortex/v1"


def run_snow_sql(query: str, connection: str | None = None) -> Any:
    cmd = ["snow", "sql", "--enable-templating", "NONE", "--format", "json", "-q", query]
    if connection:
        cmd.extend(["--connection", connection])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Snowflake SQL execution failed: {proc.stderr[-1500:]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def load_auth_token() -> tuple[str, str]:
    """Return `(token, auth_type)` from `SNOWFLAKE_PAT` or SPCS `/snowflake/session/token`."""
    pat = os.environ.get("SNOWFLAKE_PAT", "").strip()
    if pat:
        return pat, "PROGRAMMATIC_ACCESS_TOKEN"
    token_file = Path(os.environ.get("SNOWFLAKE_TOKEN_FILE", "/snowflake/session/token"))
    if token_file.is_file():
        return token_file.read_text().strip(), "OAUTH"
    raise RuntimeError("Set SNOWFLAKE_PAT or mount /snowflake/session/token for Cortex REST API probing")


def fetch_current_account_inventory(cfg: ScoutConfig) -> dict[str, Any]:
    ctx_rows = run_snow_sql(
        "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_REGION() AS REGION, CURRENT_ROLE() AS ROLE, CURRENT_USER() AS USER",
        cfg.connection,
    )
    identity = ctx_rows[0] if isinstance(ctx_rows, list) and ctx_rows and isinstance(ctx_rows[0], dict) else {}
    models = run_snow_sql("SHOW MODELS IN ACCOUNT", cfg.connection)
    return {
        "captured_at": utcnow().isoformat(),
        "host": cfg.host,
        "context": identity,
        "models": models if isinstance(models, list) else [],
    }


def fetch_latest_registry_snapshot(cfg: ScoutConfig) -> tuple[RevisionSnapshot, str, int]:
    rows = run_snow_sql(f"LIST {cfg.stage}/checkpoints/", cfg.connection)
    checkpoint_names = []
    for row in rows:
        name = row.get("name", "")
        base = name.split("/")[-1]
        m = re.fullmatch(r"checkpoint-(\d{20})-([0-9a-f]{64})\.json", base)
        if m:
            checkpoint_names.append((int(m.group(1)), m.group(2), base))
    if not checkpoint_names:
        raise RuntimeError(f"No checkpoints found on {cfg.stage}/checkpoints/")
    checkpoint_names.sort(key=lambda item: item[0])
    latest_gen, latest_rev, latest_file = checkpoint_names[-1]
    with tempfile.TemporaryDirectory() as tmp:
        run_snow_sql(f"GET {cfg.stage}/checkpoints/{latest_file} 'file://{tmp}/'", cfg.connection)
        raw = (Path(tmp) / latest_file).read_bytes()
    doc = strict_json(raw)
    snap = parse_snapshot(doc)
    actual_rev = digest(snap.document())
    if actual_rev != latest_rev:
        raise RuntimeError(f"Checkpoint digest mismatch: {actual_rev} != {latest_rev}")
    return snap, actual_rev, latest_gen


def active_catalog_from_snapshot(snap: RevisionSnapshot) -> dict[str, list[str]]:
    by_channel: dict[str, list[str]] = {"messages": [], "chat": [], "responses": []}
    revoked = set(snap.revoked)
    for model_id, entry in sorted(snap.entries.items()):
        if model_id in revoked:
            continue
        proto = entry.proposal.profile.protocol
        if proto in by_channel:
            by_channel[proto].append(model_id)
    return by_channel


def load_previous_snapshot(cfg: ScoutConfig, fallback_inventory: dict[str, Any]) -> tuple[dict[str, Any], str]:
    cfg.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    latest_local = cfg.state_dir / "models_snapshot_latest.json"
    if latest_local.exists():
        return json.loads(latest_local.read_text()), str(latest_local)
    try:
        rows = run_snow_sql(f"LIST {cfg.stage}/{cfg.scouting_prefix}/state/models_snapshot_latest.json", cfg.connection)
        if rows:
            with tempfile.TemporaryDirectory() as tmp:
                run_snow_sql(
                    f"GET {cfg.stage}/{cfg.scouting_prefix}/state/models_snapshot_latest.json 'file://{tmp}/'",
                    cfg.connection,
                )
                fetched = Path(tmp) / "models_snapshot_latest.json"
                if fetched.exists():
                    data = json.loads(fetched.read_text())
                    latest_local.write_text(json.dumps(data, indent=2))
                    return data, f"{cfg.stage}/{cfg.scouting_prefix}/state/models_snapshot_latest.json"
    except Exception:
        pass
    return fallback_inventory, "INITIAL_SNAPSHOT"


def load_scouting_ledger(cfg: ScoutConfig) -> dict[str, Any]:
    cfg.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    ledger_path = cfg.state_dir / "scouting_ledger.json"
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text())
    else:
        data = {}
    data.setdefault("blocked", {})
    data.setdefault("promoted", {})
    return data


def save_state_and_stage(cfg: ScoutConfig, snapshot: dict[str, Any], ledger: dict[str, Any]) -> None:
    cfg.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    snap_path = cfg.state_dir / "models_snapshot_latest.json"
    dated_path = cfg.state_dir / f"models_snapshot_{utcnow().strftime('%Y%m%d')}.json"
    ledger_path = cfg.state_dir / "scouting_ledger.json"

    snap_path.write_text(json.dumps(snapshot, indent=2))
    dated_path.write_text(json.dumps(snapshot, indent=2))
    ledger_path.write_text(json.dumps(ledger, indent=2))

    for p in (snap_path, dated_path, ledger_path):
        run_snow_sql(
            f"PUT 'file://{p}' {cfg.stage}/{cfg.scouting_prefix}/state AUTO_COMPRESS=FALSE OVERWRITE=TRUE",
            cfg.connection,
        )


def candidate_channels_for_model(model_id: str) -> list[tuple[str, str | None]]:
    low = model_id.lower()
    if low.startswith("claude-"):
        return [("messages", None), ("chat", None)]
    if low.startswith("openai-gpt-"):
        return [("responses", None), ("chat", "none"), ("chat", None)]
    return [("chat", None), ("chat", "none"), ("responses", None), ("messages", None)]


def fast_channel_probe(
    base_url: str,
    token: str,
    auth_type: str,
    model_id: str,
    protocol: str,
    reasoning: str | None = None,
) -> dict[str, Any]:
    if protocol == "chat":
        path = "/chat/completions"
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_completion_tokens": 64,
        }
        if reasoning:
            body["reasoning_effort"] = reasoning
    elif protocol == "responses":
        path = "/responses"
        body = {
            "model": model_id,
            "input": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_output_tokens": 256,
        }
    elif protocol == "messages":
        path = "/messages"
        body = {
            "model": model_id,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        }
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Snowflake-Authorization-Token-Type": auth_type,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if protocol == "messages":
        headers["anthropic-version"] = "2023-06-01"

    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Snowflake-Request-Id")
            return {
                "protocol": protocol,
                "reasoning": reasoning,
                "reachable": True,
                "http_status": resp.status,
                "latency_s": round(time.monotonic() - t0, 3),
                "request_id": req_id,
                "response_preview": raw[:300],
            }
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return {
            "protocol": protocol,
            "reasoning": reasoning,
            "reachable": False,
            "http_status": exc.code,
            "latency_s": round(time.monotonic() - t0, 3),
            "error": err_body[:500],
        }
    except Exception as exc:
        return {
            "protocol": protocol,
            "reasoning": reasoning,
            "reachable": False,
            "http_status": None,
            "latency_s": round(time.monotonic() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def qualify_model_on_channel(
    cfg: ScoutConfig,
    token: str,
    auth_type: str,
    model_id: str,
    protocol: str,
    reasoning: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    """Run the cortex-model-compatibility-matrix Campaign on `(model_id, protocol, reasoning)`."""
    from model_probe.campaign import Campaign
    from model_probe.transport import Transport

    os.environ["SNOWFLAKE_PAT"] = token
    out_dir = run_dir / f"{model_id}_{protocol}_{reasoning or 'default'}"
    transport = Transport(
        cfg.upstream_url,
        cfg.host,
        token_env="SNOWFLAKE_PAT",
        auth_type=auth_type,
    )
    campaign = Campaign(
        transport,
        model_id,
        protocol,
        out_dir,
        repeats=cfg.repeats,
        reasoning=reasoning,
        max_requests=60,
        deadline=1800,
        scope="direct-rest",
        capabilities=None,
    )
    summary = campaign.run()
    counts = summary.get("counts", {})
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
    structured_cases = {"structured", "structured_negative"}
    core_rows = [r for r in results if (r.get("name") or r.get("case")) in core_cases]
    struct_rows = [r for r in results if (r.get("name") or r.get("case")) in structured_cases]

    full_suite_qualified = summary.get("status") == "PROFILE_QUALIFIED"
    core_all_pass = full_suite_qualified or (
        len(core_rows) == len(core_cases) * cfg.repeats and all(r.get("status") == "PASS" for r in core_rows)
    )
    struct_all_pass = full_suite_qualified or (
        len(struct_rows) == len(structured_cases) * cfg.repeats and all(r.get("status") == "PASS" for r in struct_rows)
    )

    qualified_caps: list[str] = []
    if core_all_pass:
        qualified_caps = ["text", "tools", "parallel_tools", "streaming"]
        if struct_all_pass:
            qualified_caps.append("structured_output")

    return {
        "model": model_id,
        "protocol": protocol,
        "reasoning": reasoning,
        "status": summary.get("status"),
        "promotable": bool(core_all_pass),
        "full_suite_pass": full_suite_qualified,
        "core_agentic_pass": bool(core_all_pass),
        "structured_output_pass": bool(struct_all_pass),
        "qualified_capabilities": qualified_caps,
        "counts": counts,
        "evidence_dir": str(out_dir),
        "profile_sha256": summary.get("profile_sha256"),
    }


def scout_and_qualify_new_model(
    cfg: ScoutConfig,
    token: str,
    auth_type: str,
    model_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    candidates = candidate_channels_for_model(model_id)
    preflight_results = []
    chosen: tuple[str, str | None] | None = None

    for proto, reas in candidates:
        probe = fast_channel_probe(cfg.upstream_url, token, auth_type, model_id, proto, reasoning=reas)
        preflight_results.append(probe)
        if probe.get("reachable"):
            chosen = (proto, reas)
            break

    if chosen is None:
        return {
            "model": model_id,
            "status": "PROFILE_BLOCKED",
            "promotable": False,
            "channel": None,
            "reason": "All candidate Cortex REST API channels (messages, chat, responses) rejected the model at preflight.",
            "preflight": preflight_results,
            "qualification": None,
        }

    proto, reas = chosen
    qual = qualify_model_on_channel(cfg, token, auth_type, model_id, proto, reas, run_dir)
    supports_re_effort = False
    if proto == "chat" and reas is None and qual.get("promotable"):
        supports_re_effort = bool(
            fast_channel_probe(cfg.upstream_url, token, auth_type, model_id, "chat", reasoning="low").get("reachable")
        )
    qual["supports_reasoning_effort"] = supports_re_effort

    return {
        "model": model_id,
        "status": "PROFILE_QUALIFIED" if qual.get("promotable") else "PROFILE_BLOCKED",
        "promotable": qual.get("promotable", False),
        "channel": proto,
        "reasoning": reas,
        "supports_reasoning_effort": supports_re_effort,
        "reason": (
            f"Qualified on channel '{proto}' ({qual['counts'].get('PASS', 0)} PASS / {qual['counts'].get('FAIL', 0)} FAIL)"
            if qual.get("promotable")
            else f"Blocked on channel '{proto}' ({qual['counts'].get('PASS', 0)} PASS / {qual['counts'].get('FAIL', 0)} FAIL)"
        ),
        "preflight": preflight_results,
        "qualification": qual,
    }


def compute_next_snapshot_revision(snap_doc: dict[str, Any], entry_doc: dict[str, Any]) -> dict[str, Any]:
    next_doc = json.loads(json.dumps(snap_doc))
    entry_sha = digest(entry_doc)
    model_id = entry_doc["proposal"]["profile"]["id"]
    if next_doc.get("schema_version") == 2:
        next_doc.setdefault("history", {})[entry_sha] = entry_doc
        if "previous_entry" in entry_doc.get("proposal", {}):
            next_doc.setdefault("activations", []).append(entry_sha)
    next_doc.setdefault("entries", {})[model_id] = entry_doc
    next_doc["generation"] = int(next_doc.get("generation", 0)) + 1
    parse_snapshot(next_doc)
    return next_doc


def promote_qualified_models(
    cfg: ScoutConfig,
    qualified_items: list[dict[str, Any]],
    snap: RevisionSnapshot,
    current_revision: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    promoted_records = []
    working_doc = snap.document()
    expected_rev = current_revision
    rev_tag = f"auto-scout-{utcnow().strftime('%Y%m%d%H%M%S')}"

    for item in qualified_items:
        model_id = item["model"]
        proto = item["channel"]
        reasoning_none = item.get("reasoning") == "none"
        supports_re = bool(item.get("supports_reasoning_effort"))
        qual_caps = list(
            (item.get("qualification") or {}).get("qualified_capabilities")
            or ["text", "tools", "parallel_tools", "streaming"]
        )
        if proto in ("responses", "messages") or reasoning_none or supports_re:
            if "reasoning" not in qual_caps:
                qual_caps.append("reasoning")
        reasoning_efforts = ["none", "low", "medium", "high"] if (proto == "chat" and supports_re and not reasoning_none) else None

        prior_entry = working_doc.get("entries", {}).get(model_id)
        prev_entry_sha = digest(prior_entry) if prior_entry is not None else None

        profile = build_universal_profile(
            model_id,
            proto,
            cfg.artifact_digest,
            account=cfg.account,
            revision=rev_tag,
            reasoning_none=reasoning_none,
            reasoning_efforts=reasoning_efforts,
            capabilities=qual_caps,
            days=365,
        )
        entry, report, evidence_name, package, package_name = build_package_for_profile(
            profile,
            expected_rev,
            upstream=cfg.upstream_url,
            approver=cfg.approver,
            target_origin=cfg.target_origin,
            previous_entry=prev_entry_sha,
            sources=[{"scope": f"automated daily model scouting & qualification ({rev_tag})"}],
        )

        pkg_dir = run_dir / "promotions" / model_id
        pkg_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        ev_path = pkg_dir / evidence_name
        pkg_path = pkg_dir / package_name
        ev_path.write_bytes(canonical(report))
        pkg_path.write_bytes(canonical(package))

        run_snow_sql(
            f"PUT 'file://{ev_path}' {cfg.stage}/{cfg.inbox_prefix} AUTO_COMPRESS=FALSE OVERWRITE=FALSE",
            cfg.connection,
        )
        run_snow_sql(
            f"PUT 'file://{pkg_path}' {cfg.stage}/{cfg.inbox_prefix} AUTO_COMPRESS=FALSE OVERWRITE=FALSE",
            cfg.connection,
        )

        entry_doc = entry.document()
        working_doc = compute_next_snapshot_revision(working_doc, entry_doc)
        next_rev = digest(working_doc)

        promoted_records.append(
            {
                "model": model_id,
                "protocol": proto,
                "capabilities": qual_caps,
                "reasoning_none": reasoning_none,
                "supports_reasoning_effort": supports_re,
                "expires_at": profile.expires_at,
                "entry_sha256": digest(entry_doc),
                "package": package_name,
                "evidence": evidence_name,
                "expected_revision_before": expected_rev,
                "expected_revision_after": next_rev,
                "expected_generation_after": working_doc["generation"],
            }
        )
        expected_rev = next_rev

    return promoted_records


def revoke_disappeared_models(
    cfg: ScoutConfig,
    disappeared_models: list[str],
    snap: RevisionSnapshot,
    current_revision: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    revoked_records = []
    working_doc = snap.document()
    expected_rev = current_revision

    for model_id in disappeared_models:
        if model_id in working_doc.get("revoked", []):
            continue
        pkg, pkg_name = build_revoke_package(
            model_id,
            expected_rev,
            approver=cfg.approver,
            account=cfg.account,
            reason=f"Model '{model_id}' disappeared from SHOW MODELS IN ACCOUNT on {utcnow().strftime('%Y-%m-%d')}",
        )
        pkg_dir = run_dir / "revocations" / model_id
        pkg_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        pkg_path = pkg_dir / pkg_name
        pkg_path.write_bytes(canonical(pkg))
        run_snow_sql(
            f"PUT 'file://{pkg_path}' {cfg.stage}/{cfg.inbox_prefix} AUTO_COMPRESS=FALSE OVERWRITE=FALSE",
            cfg.connection,
        )

        working_doc = json.loads(json.dumps(working_doc))
        working_doc.setdefault("revoked", []).append(model_id)
        working_doc["generation"] = int(working_doc.get("generation", 0)) + 1
        next_rev = digest(working_doc)

        revoked_records.append(
            {
                "model": model_id,
                "action": "revoked",
                "package": pkg_name,
                "expected_revision_before": expected_rev,
                "expected_revision_after": next_rev,
                "expected_generation_after": working_doc["generation"],
            }
        )
        expected_rev = next_rev

    return revoked_records


def format_run_markdown(report: dict[str, Any]) -> str:
    ts = report["run_timestamp"]
    run_id = report["run_id"]
    diff = report["diff"]
    cat = report["active_catalog_by_channel"]
    tested_names_md = ", ".join(f"`{m['model']}`" for m in report["tested_models"]) or "none"
    revoked_names_md = ", ".join(f"`{r['model']}`" for r in report["revoked_models"]) or "none"
    added_cb_md = ", ".join(f"`{m}`" for m in diff["added_cortex_base"]) or "none"
    removed_all_md = ", ".join(f"`{m}`" for m in diff["removed_all"]) or "none"

    lines = [
        f"# Daily Model Scouting & Promotion Report — `{run_id}`",
        "",
        f"- **Timestamp (UTC):** `{ts}`",
        f"- **Target Account:** `{report['account']}`",
        f"- **Snapshot Diff:** `{diff['previous_total_models']}` models -> `{diff['current_total_models']}` models",
        f"- **Action Taken (`action_taken`):** `{report['action_taken']}`",
        f"- **Notification Sent (`notification_sent`):** `{report.get('notification_sent', False)}`",
        "",
        "## 1. Account Model Snapshot Diff (`SHOW MODELS IN ACCOUNT`)",
        "",
        f"- **New `CORTEX_BASE` models detected:** `{len(diff['added_cortex_base'])}` ({added_cb_md})",
        f"- **Candidate models tested in this run:** `{len(report['tested_models'])}` ({tested_names_md})",
        f"- **Disappeared models in account:** `{len(diff['removed_all'])}` ({removed_all_md})",
        f"- **Active Proxy models revoked:** `{len(report['revoked_models'])}` ({revoked_names_md})",
        "",
    ]

    if report["tested_models"]:
        lines.extend(
            [
                "## 2. Qualification Results Across API Channels (`messages`, `chat`, `responses`)",
                "",
                "| Model | API Channel | Status | Suite Result | Qualified Capabilities | Notes |",
                "|---|---|---|---|---|---|",
            ]
        )
        for t in report["tested_models"]:
            m_name = t["model"]
            ch = t.get("channel") or "N/A (REST Rejected)"
            st = t.get("status")
            qual = t.get("qualification")
            if qual:
                cnt = qual.get("counts", {})
                suite_str = f"`{cnt.get('PASS', 0)} PASS / {cnt.get('FAIL', 0)} FAIL / {cnt.get('BLOCKED', 0)} BLOCKED`"
                caps_str = ", ".join(f"`{c}`" for c in (qual.get("qualified_capabilities") or [])) or "-"
            else:
                suite_str = "Preflight HTTP 400"
                caps_str = "-"
            note = t.get("reason", "")
            lines.append(f"| `{m_name}` | `{ch}` | **{st}** | {suite_str} | {caps_str} | {note} |")
        lines.append("")

    lines.extend(
        [
            "## 3. Active Catalog by API Channel",
            "",
            f"- **`messages` (`/messages/v1/models`, {len(cat['messages'])} models):** {', '.join(f'`{m}`' for m in cat['messages'])}",
            f"- **`chat` (`/chat/v1/models`, {len(cat['chat'])} models):** {', '.join(f'`{m}`' for m in cat['chat'])}",
            f"- **`responses` (`/responses/v1/models`, {len(cat['responses'])} models):** {', '.join(f'`{m}`' for m in cat['responses'])}",
            "",
        ]
    )
    return "\n".join(lines)


def build_notification_html(report: dict[str, Any]) -> str:
    run_id = html.escape(report["run_id"])
    ts = html.escape(report["run_timestamp"])
    diff = report["diff"]
    cat = report["active_catalog_by_channel"]
    promoted_str = ", ".join(f"{p['model']} ({p['protocol']})" for p in report["promoted_models"]) or "None"
    revoked_str = ", ".join(r["model"] for r in report["revoked_models"]) or "None"

    return f"""<html>
<body style="font-family: sans-serif; color: #1f2328;">
  <h2>Snowflake Cortex Proxy — Daily Model Scouting &amp; Promotion Report</h2>
  <p>
    <strong>Run ID:</strong> <code>{run_id}</code><br/>
    <strong>Timestamp (UTC):</strong> <code>{ts}</code><br/>
    <strong>Model Snapshot:</strong> {diff['previous_total_models']} &rarr; {diff['current_total_models']} total models<br/>
    <strong>Promoted Models:</strong> <strong>{html.escape(promoted_str)}</strong><br/>
    <strong>Revoked (Disappeared) Models:</strong> <strong>{html.escape(revoked_str)}</strong>
  </p>
  <h3>Active Catalog by Channel</h3>
  <ul>
    <li><strong>messages (<code>/messages/v1</code>):</strong> <code>{html.escape(', '.join(cat['messages']))}</code></li>
    <li><strong>chat (<code>/chat/v1</code>):</strong> <code>{html.escape(', '.join(cat['chat']))}</code></li>
    <li><strong>responses (<code>/responses/v1</code>):</strong> <code>{html.escape(', '.join(cat['responses']))}</code></li>
  </ul>
</body>
</html>"""


def persist_run_and_notify(cfg: ScoutConfig, current_inventory: dict[str, Any], report: dict[str, Any], md_text: str) -> bool:
    run_id = report["run_id"]
    action_taken = bool(report["action_taken"])
    notification_sent = False

    if action_taken and cfg.notify and cfg.notification_integration and cfg.notification_email:
        n_prom = len(report["promoted_models"])
        n_rev = len(report["revoked_models"])
        n_blk = len(report["blocked_models"])
        subject = f"[Cortex Model Scout] Run {run_id}: {n_prom} promoted, {n_rev} removed, {n_blk} blocked"
        html_body = build_notification_html(report)
        subj_hex = subject.encode("utf-8").hex()
        html_hex = html_body.encode("utf-8").hex()
        notify_sql = f"""
        CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
            SNOWFLAKE.NOTIFICATION.TEXT_HTML(HEX_DECODE_STRING('{html_hex}')),
            SNOWFLAKE.NOTIFICATION.EMAIL_INTEGRATION_CONFIG(
                '{cfg.notification_integration}',
                HEX_DECODE_STRING('{subj_hex}'),
                ARRAY_CONSTRUCT('{cfg.notification_email}')
            )
        )
        """
        run_snow_sql(notify_sql, cfg.connection)
        notification_sent = True

    report["notification_sent"] = notification_sent
    snap_hex = json.dumps(current_inventory).encode("utf-8").hex()
    rep_hex = json.dumps(report).encode("utf-8").hex()
    md_hex = md_text.encode("utf-8").hex()
    src_hex = str(report.get("previous_snapshot_source", "")).encode("utf-8").hex()
    models_list = current_inventory.get("models", [])
    cortex_base_count = sum(1 for m in models_list if str(m.get("model_type", "")).upper() == "CORTEX_BASE")

    run_snow_sql(
        f"""
        INSERT INTO {cfg.scout_schema}.MODEL_SNAPSHOT_HISTORY (
            SNAPSHOT_ID, CAPTURED_AT, TOTAL_MODELS, CORTEX_BASE_MODELS, MODELS_JSON
        )
        SELECT
            '{run_id}',
            CURRENT_TIMESTAMP(),
            {len(models_list)},
            {cortex_base_count},
            PARSE_JSON(HEX_DECODE_STRING('{snap_hex}'))
        """,
        cfg.connection,
    )

    run_snow_sql(
        f"""
        INSERT INTO {cfg.scout_schema}.MODEL_SCOUTING_RUNS (
            RUN_ID, RUN_TIMESTAMP, PREVIOUS_SNAPSHOT_SOURCE, PREV_MODEL_COUNT, CURR_MODEL_COUNT,
            ADDED_MODELS, REMOVED_MODELS, TESTED_MODELS, PROMOTED_MODELS, REVOKED_MODELS, BLOCKED_MODELS,
            ACTIVE_CATALOG_BY_CHANNEL, CONSUMER_SYNC_STATUS, ACTION_TAKEN, NOTIFICATION_SENT,
            REPORT_MARKDOWN, REPORT_JSON
        )
        SELECT
            '{run_id}',
            CURRENT_TIMESTAMP(),
            HEX_DECODE_STRING('{src_hex}'),
            {int(report['diff']['previous_total_models'])},
            {int(report['diff']['current_total_models'])},
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['diff']['added_cortex_base']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['diff']['removed_all']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps([t['model'] for t in report['tested_models']]).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['promoted_models']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['revoked_models']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['blocked_models']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report['active_catalog_by_channel']).encode('utf-8').hex()}')),
            PARSE_JSON(HEX_DECODE_STRING('{json.dumps(report.get('consumer_sync', {})).encode('utf-8').hex()}')),
            {str(action_taken).upper()},
            {str(notification_sent).upper()},
            HEX_DECODE_STRING('{md_hex}'),
            PARSE_JSON(HEX_DECODE_STRING('{rep_hex}'))
        """,
        cfg.connection,
    )
    return notification_sent


def run_daily_scouting(cfg: ScoutConfig) -> dict[str, Any]:
    now_dt = utcnow()
    run_id = now_dt.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, mode=0o700, exist_ok=True)

    current_inv = fetch_current_account_inventory(cfg)
    prev_inv, prev_source = load_previous_snapshot(cfg, current_inv)
    ledger = load_scouting_ledger(cfg)
    snap, current_rev, current_gen = fetch_latest_registry_snapshot(cfg)
    active_by_channel = active_catalog_from_snapshot(snap)
    active_set = set().union(*active_by_channel.values())

    curr_models_map = {str(m["name"]).lower(): m for m in current_inv.get("models", [])}
    prev_models_map = {str(m["name"]).lower(): m for m in prev_inv.get("models", [])}

    added_all = sorted(set(curr_models_map) - set(prev_models_map))
    removed_all = sorted(set(prev_models_map) - set(curr_models_map))
    added_cortex_base = [
        m for m in added_all if str(curr_models_map[m].get("model_type", "")).upper() == "CORTEX_BASE"
    ]
    disappeared_active = sorted(m for m in active_set if m.lower() not in curr_models_map)

    models_to_test: list[str] = []
    for m in added_cortex_base:
        if m in active_set:
            continue
        created_on = curr_models_map[m].get("created_on")
        blocked_entry = ledger.get("blocked", {}).get(m)
        if blocked_entry and not cfg.retest_blocked and blocked_entry.get("created_on") == created_on:
            continue
        models_to_test.append(m)

    for fm in cfg.force_models:
        fm_low = fm.lower()
        if fm_low in curr_models_map and fm_low not in models_to_test:
            models_to_test.append(fm_low)

    tested_results: list[dict[str, Any]] = []
    qualified_items: list[dict[str, Any]] = []
    blocked_items: list[dict[str, Any]] = []

    if models_to_test:
        token, auth_type = load_auth_token()
        for m_id in models_to_test:
            outcome = scout_and_qualify_new_model(cfg, token, auth_type, m_id, run_dir)
            tested_results.append(outcome)
            created_on = curr_models_map[m_id].get("created_on")
            if outcome.get("promotable"):
                qualified_items.append(outcome)
                ledger["promoted"][m_id] = {
                    "model": m_id,
                    "created_on": created_on,
                    "channel": outcome["channel"],
                    "evaluated_at": utcnow().isoformat(),
                }
                ledger.get("blocked", {}).pop(m_id, None)
            else:
                blocked_record = {
                    "model": m_id,
                    "created_on": created_on,
                    "channel": outcome.get("channel"),
                    "status": outcome.get("status"),
                    "reason": outcome.get("reason"),
                    "evaluated_at": utcnow().isoformat(),
                }
                blocked_items.append(blocked_record)
                ledger.setdefault("blocked", {})[m_id] = blocked_record

    promoted_records = (
        promote_qualified_models(cfg, qualified_items, snap, current_rev, run_dir) if qualified_items else []
    )
    last_exp_rev = promoted_records[-1]["expected_revision_after"] if promoted_records else current_rev
    revoked_records = (
        revoke_disappeared_models(cfg, disappeared_active, snap, last_exp_rev, run_dir) if disappeared_active else []
    )

    for p in promoted_records:
        if p["model"] not in active_by_channel[p["protocol"]]:
            active_by_channel[p["protocol"]].append(p["model"])
            active_by_channel[p["protocol"]].sort()
    for r in revoked_records:
        for ch_list in active_by_channel.values():
            if r["model"] in ch_list:
                ch_list.remove(r["model"])

    action_taken = bool(tested_results or promoted_records or revoked_records)
    report: dict[str, Any] = {
        "run_id": run_id,
        "run_timestamp": now_dt.isoformat(),
        "account": cfg.account,
        "previous_snapshot_source": prev_source,
        "registry_generation": current_gen + len(promoted_records) + len(revoked_records),
        "registry_revision": (
            revoked_records[-1]["expected_revision_after"]
            if revoked_records
            else (promoted_records[-1]["expected_revision_after"] if promoted_records else current_rev)
        ),
        "action_taken": action_taken,
        "diff": {
            "previous_total_models": len(prev_models_map),
            "current_total_models": len(curr_models_map),
            "added_all": added_all,
            "added_cortex_base": added_cortex_base,
            "removed_all": removed_all,
            "disappeared_active": disappeared_active,
        },
        "tested_models": tested_results,
        "promoted_models": promoted_records,
        "revoked_models": revoked_records,
        "blocked_models": blocked_items,
        "active_catalog_by_channel": active_by_channel,
    }

    md_text = format_run_markdown(report)
    persist_run_and_notify(cfg, current_inv, report, md_text)
    md_text = format_run_markdown(report)

    md_file = cfg.reports_dir / f"report_{run_id}.md"
    json_file = cfg.reports_dir / f"report_{run_id}.json"
    md_file.write_text(md_text)
    json_file.write_text(json.dumps(report, indent=2))

    for f in (md_file, json_file):
        run_snow_sql(
            f"PUT 'file://{f}' {cfg.stage}/{cfg.scouting_prefix}/reports AUTO_COMPRESS=FALSE OVERWRITE=TRUE",
            cfg.connection,
        )

    save_state_and_stage(cfg, current_inv, ledger)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default=None, help="Snowflake account locator")
    parser.add_argument("--host", default=None, help="Snowflake host (<org>-<account>.snowflakecomputing.com)")
    parser.add_argument("--connection", default=None, help="Snowflake CLI connection name")
    parser.add_argument("--stage", default=None, help="Snowflake stage FQN (e.g. @DB.SCHEMA.STAGE)")
    parser.add_argument("--inbox-prefix", default=None, help="Stage prefix for Registry inbox")
    parser.add_argument("--scouting-prefix", default=None, help="Stage prefix for scouting state and reports")
    parser.add_argument("--scout-schema", default=None, help="Database.Schema for MODEL_SCOUTING_RUNS table")
    parser.add_argument("--artifact", default=None, help="64-hex Proxy artifact digest")
    parser.add_argument("--approver", default=None, help="Admin approver identity in registry-trust.json")
    parser.add_argument("--target-origin", default=None, help="Internal SPCS URL of LLM Proxy")
    parser.add_argument("--notification-integration", default=None, help="Snowflake EMAIL notification integration name")
    parser.add_argument("--notification-email", default=None, help="Verified recipient email for action alerts")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=3)
    parser.add_argument("--force-models", nargs="+", default=None)
    parser.add_argument("--retest-blocked", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    cfg = ScoutConfig(args)
    report = run_daily_scouting(cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
