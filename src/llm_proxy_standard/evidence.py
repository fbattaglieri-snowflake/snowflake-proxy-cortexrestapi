"""Administrative import only: canonical kit gate, never inference."""
import hashlib
import json
from pathlib import Path
from .contracts import Evidence, digest, strict_json, timestamp


def import_campaign(directory, profile, *, target, expires_at, synthetic=False):
    import model_probe
    from model_probe.campaign import gate, required_cases, normalize_capabilities
    directory = Path(directory)
    paths = [directory / "manifest.json", *sorted(directory.glob("case-*.json"))]
    documents, hashes = [], {}
    for path in paths:
        raw = path.read_bytes()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Evidence file too large")
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        documents.append(strict_json(raw))
    manifest, rows = documents[0], documents[1:]
    caps = normalize_capabilities([cap for cap in profile.capabilities if cap != "reasoning"])
    if (manifest["schema_version"] != 2 or manifest["version"] != model_probe.__version__ or
            manifest["scope"] != "proxy" or manifest["target"] != target or
            manifest["protocol"] != profile.protocol or manifest["model"] != profile.id or
            tuple(manifest["required_capabilities"]) != caps or
            tuple(manifest["required_cases"]) != required_cases(caps) or
            manifest["repeats"] != 3 or manifest["retries"] != 0 or manifest["release_eligible"] is not False or
            not gate(rows, 3, caps)):
        raise ValueError("Campaign does not qualify this exact proxy profile")
    source = Path(model_probe.__file__).parent
    actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob("*.py")}
    if manifest["source_sha256"] != actual:
        raise ValueError("Runner source differs from reviewed installed kit")
    identity = {"model": manifest["model"], "protocol": manifest["protocol"],
                "reasoning": manifest["reasoning"], "required_capabilities": caps,
                "required_cases": required_cases(caps)}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != manifest["profile_sha256"]:
        raise ValueError("Campaign profile hash mismatch")
    if not synthetic:
        execution_path = directory / "execution.json"
        raw = execution_path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("Execution manifest too large")
        execution = strict_json(raw)
        if (execution.get("schema_version") != 1 or execution.get("synthetic") is not False or
                profile.document() not in execution.get("profiles", []) or
                timestamp(execution["created_at"]) > timestamp(manifest["created_utc"]) or
                execution["candidate_revision"] != digest(execution["profiles"])):
            raise ValueError("Execution was not frozen before this campaign")
        runtime = Path(__file__).parent
        actual_runtime = {"runtime/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in runtime.glob("*.py")}
        recorded_runtime = {key: value for key, value in execution["source_sha256"].items() if key.startswith("runtime/")}
        if actual_runtime != recorded_runtime:
            raise ValueError("Execution runtime differs from reviewed runtime")
        expected_receipt = {"execution_sha256": digest(execution), "profile_sha256": digest(profile.document()),
                            "candidate_revision": execution["candidate_revision"]}
        if (manifest["destination_policy"].get("execution_sha256") != digest(execution) or
                any(row.get("execution_receipt") != expected_receipt for row in rows if row["status"] == "PASS")):
            raise ValueError("Missing/mismatched per-request execution attestation")
        hashes[execution_path.name] = hashlib.sha256(raw).hexdigest()
    return Evidence(kind="kit-proxy", sha256=digest(hashes), profile_sha256=digest(profile.document()),
                    target=target, expires_at=expires_at, coverage=["kit:" + cap for cap in caps], synthetic=synthetic)