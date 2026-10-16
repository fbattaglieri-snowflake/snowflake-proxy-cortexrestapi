"""Data-only adaptations. History/tool/reasoning payloads are never rewritten."""
import copy
from .contracts import OPTION_PATHS, ProfileV2, V2_EXTRA_PATHS, canonical

MISSING = object()


class RequestPolicyError(ValueError):
    """Only fixed codes and allowlisted paths may cross the HTTP boundary."""
    def __init__(self, code, field=None):
        self.code, self.field = code, field
        super().__init__(code)


def _deduplicate_history_tool_ids(history, protocol):
    if not isinstance(history, list):
        return False
    used_ids = set()
    active_remap = {}
    pending_orig = set()
    changed = False
    round_idx = 0
    for message in history:
        if not isinstance(message, dict):
            continue
        if protocol == "chat":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                if not pending_orig:
                    active_remap = {}
                round_idx += 1
                for item in tool_calls:
                    if not isinstance(item, dict):
                        continue
                    orig = item.get("id")
                    if isinstance(orig, str) and orig:
                        if orig in used_ids:
                            new_id = f"{orig}_r{round_idx}"
                            while new_id in used_ids or new_id in pending_orig:
                                round_idx += 1
                                new_id = f"{orig}_r{round_idx}"
                            active_remap[orig] = new_id
                            item["id"] = new_id
                            changed = True
                        pending_orig.add(orig)
            if message.get("role") == "tool":
                tc_id = message.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id in active_remap:
                    message["tool_call_id"] = active_remap[tc_id]
                    changed = True
                if tc_id in pending_orig:
                    pending_orig.remove(tc_id)
                    used_ids.add(active_remap.get(tc_id, tc_id))
                    if not pending_orig:
                        active_remap = {}
        elif protocol == "responses":
            if message.get("type") == "function_call":
                orig = message.get("call_id")
                if isinstance(orig, str) and orig:
                    if orig in used_ids and orig not in pending_orig:
                        round_idx += 1
                        new_id = f"{orig}_r{round_idx}"
                        while new_id in used_ids or new_id in pending_orig:
                            round_idx += 1
                            new_id = f"{orig}_r{round_idx}"
                        active_remap[orig] = new_id
                        message["call_id"] = new_id
                        changed = True
                    pending_orig.add(orig)
            elif message.get("type") == "function_call_output":
                tc_id = message.get("call_id")
                if isinstance(tc_id, str) and tc_id in active_remap:
                    message["call_id"] = active_remap[tc_id]
                    changed = True
                if tc_id in pending_orig:
                    pending_orig.remove(tc_id)
                    used_ids.add(active_remap.get(tc_id, tc_id))
                    if not pending_orig:
                        active_remap = {}
        else:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "tool_use":
                        orig = item.get("id")
                        if isinstance(orig, str) and orig:
                            if orig in used_ids and orig not in pending_orig:
                                round_idx += 1
                                new_id = f"{orig}_r{round_idx}"
                                while new_id in used_ids or new_id in pending_orig:
                                    round_idx += 1
                                    new_id = f"{orig}_r{round_idx}"
                                active_remap[orig] = new_id
                                item["id"] = new_id
                                changed = True
                            pending_orig.add(orig)
                    elif item.get("type") == "tool_result":
                        tc_id = item.get("tool_use_id")
                        if isinstance(tc_id, str) and tc_id in active_remap:
                            item["tool_use_id"] = active_remap[tc_id]
                            changed = True
                        if tc_id in pending_orig:
                            pending_orig.remove(tc_id)
                            used_ids.add(active_remap.get(tc_id, tc_id))
                            if not pending_orig:
                                active_remap = {}
    return changed


def _normalize_client_body(body, profile):
    effective = copy.deepcopy(body)
    changes = []
    protocol = profile.protocol
    common = {"model", "stream", "tools", "tool_choice", "temperature", "top_p"}
    allowed_top = common | {
        "chat": {"messages", "max_tokens", "max_completion_tokens", "reasoning_effort", "parallel_tool_calls", "stream_options", "response_format"},
        "responses": {"input", "instructions", "max_output_tokens", "reasoning", "parallel_tool_calls", "store", "include", "text", "prompt_cache_key"},
        "messages": {"messages", "system", "max_tokens", "thinking", "output_config"},
    }[protocol] | {path.split(".")[0] for path in profile.constraints}
    harmless_extra = {"frequency_penalty", "presence_penalty", "user", "metadata", "seed", "logit_bias",
                      "logprobs", "top_logprobs", "n", "service_tier", "stop", "stop_sequences", "store"}
    for key in list(effective):
        if key not in allowed_top and key in harmless_extra:
            old = effective.pop(key)
            changes.append({"id": f"strip-{key}", "path": key, "requested": old, "was_missing": False,
                            "effective": None, "removed": True, "reason": "Client compatibility strip", "effects": "Dropped unsupported client hint"})
    if "tools" in effective and not effective["tools"]:
        effective.pop("tools", None)
    if not effective.get("tools"):
        for orphan in ("tool_choice", "parallel_tool_calls"):
            if orphan in effective:
                old = effective.pop(orphan)
                changes.append({"id": f"strip-orphan-{orphan}", "path": orphan, "requested": old, "was_missing": False,
                                "effective": None, "removed": True, "reason": "No tools present", "effects": "Dropped orphan tool option"})
    if protocol == "chat":
        if "max_tokens" in effective and "max_completion_tokens" in effective:
            old = effective.pop("max_tokens")
            changes.append({"id": "dedup-max-tokens", "path": "max_tokens", "requested": old, "was_missing": False,
                            "effective": None, "removed": True, "reason": "Prefer max_completion_tokens", "effects": "Resolved token limit conflict"})
        for tok_field in ("max_tokens", "max_completion_tokens"):
            if tok_field in effective and type(effective[tok_field]) is int and tok_field in profile.constraints:
                c = profile.constraints[tok_field]
                if c.kind == "integer" and (effective[tok_field] < c.minimum or effective[tok_field] > c.maximum):
                    old = effective[tok_field]
                    effective[tok_field] = max(c.minimum, min(c.maximum, old))
                    changes.append({"id": f"clamp-{tok_field}", "path": tok_field, "requested": old, "was_missing": False,
                                    "effective": effective[tok_field], "removed": False, "reason": "Clamp to profile range", "effects": "Bounded token limit"})
        re_allowed = profile.options.get("reasoning_effort") or (profile.constraints["reasoning_effort"].values if "reasoning_effort" in profile.constraints and profile.constraints["reasoning_effort"].kind == "enum" else None)
        if re_allowed and len(re_allowed) == 1:
            target_re = re_allowed[0]
            if effective.get("reasoning_effort") != target_re:
                old = effective.get("reasoning_effort", MISSING)
                effective["reasoning_effort"] = target_re
                changes.append({"id": "normalize-reasoning-effort", "path": "reasoning_effort",
                                "requested": None if old is MISSING else old, "was_missing": old is MISSING,
                                "effective": target_re, "removed": False, "reason": "Model requires specific reasoning_effort", "effects": "Set required reasoning_effort"})
        elif not re_allowed and "reasoning_effort" in effective:
            old = effective.pop("reasoning_effort")
            changes.append({"id": "strip-reasoning-effort", "path": "reasoning_effort", "requested": old, "was_missing": False,
                            "effective": None, "removed": True, "reason": "Model does not accept reasoning_effort", "effects": "Removed unsupported reasoning_effort"})
    elif protocol == "responses":
        if effective.get("store") is not False:
            old = effective.get("store", MISSING)
            effective["store"] = False
            changes.append({"id": "force-store-false", "path": "store", "requested": None if old is MISSING else old,
                            "was_missing": old is MISSING, "effective": False, "removed": False,
                            "reason": "Stateless Cortex Responses requires store=false", "effects": "Forced store=false"})
        if "reasoning" in profile.capabilities and "include" not in effective:
            effective["include"] = ["reasoning.encrypted_content"]
            changes.append({"id": "default-include-encrypted", "path": "include", "requested": None, "was_missing": True,
                            "effective": ["reasoning.encrypted_content"], "removed": False,
                            "reason": "Preserve opaque reasoning across tool turns", "effects": "Added reasoning.encrypted_content"})
        if "max_output_tokens" in effective:
            if effective["max_output_tokens"] is None:
                effective.pop("max_output_tokens")
            elif type(effective["max_output_tokens"]) is int and "max_output_tokens" in profile.constraints:
                c = profile.constraints["max_output_tokens"]
                if c.kind == "integer" and (effective["max_output_tokens"] < c.minimum or effective["max_output_tokens"] > c.maximum):
                    old = effective["max_output_tokens"]
                    effective["max_output_tokens"] = max(c.minimum, min(c.maximum, old))
                    changes.append({"id": "clamp-max-output-tokens", "path": "max_output_tokens", "requested": old, "was_missing": False,
                                    "effective": effective["max_output_tokens"], "removed": False, "reason": "Clamp to profile range", "effects": "Bounded token limit"})
        for temp_field in ("temperature", "top_p"):
            if temp_field in effective and temp_field not in profile.constraints and temp_field not in profile.options:
                old = effective.pop(temp_field)
                changes.append({"id": f"strip-{temp_field}", "path": temp_field, "requested": old, "was_missing": False,
                                "effective": None, "removed": True, "reason": "Sampling parameter not supported on reasoning Responses model", "effects": f"Removed {temp_field}"})
        if effective.get("text") == {}:
            effective.pop("text")
            changes.append({"id": "strip-empty-text", "path": "text", "requested": {}, "was_missing": False,
                            "effective": None, "removed": True, "reason": "Cortex Responses rejects empty text object", "effects": "Removed empty text object"})
    elif protocol == "messages":
        c = profile.constraints.get("max_tokens")
        max_cap = c.maximum if c and c.kind == "integer" else 128000
        min_cap = c.minimum if c and c.kind == "integer" else 1
        if "max_tokens" not in effective or effective["max_tokens"] is None:
            effective["max_tokens"] = min(4096, max_cap)
            changes.append({"id": "default-max-tokens", "path": "max_tokens", "requested": None, "was_missing": True,
                            "effective": effective["max_tokens"], "removed": False, "reason": "Messages requires max_tokens", "effects": "Set default max_tokens"})
        elif type(effective["max_tokens"]) is int and (effective["max_tokens"] < min_cap or effective["max_tokens"] > max_cap):
            old = effective["max_tokens"]
            effective["max_tokens"] = max(min_cap, min(max_cap, old))
            changes.append({"id": "clamp-max-tokens", "path": "max_tokens", "requested": old, "was_missing": False,
                            "effective": effective["max_tokens"], "removed": False, "reason": "Clamp to profile range", "effects": "Bounded max_tokens"})
        if isinstance(effective.get("tool_choice"), str) and effective["tool_choice"] in {"auto", "none"}:
            old = effective["tool_choice"]
            effective["tool_choice"] = {"type": old}
        thinking = effective.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            for temp_field in ("temperature", "top_p"):
                if temp_field in effective:
                    old = effective.pop(temp_field)
                    changes.append({"id": f"strip-{temp_field}-thinking", "path": temp_field, "requested": old, "was_missing": False,
                                    "effective": None, "removed": True, "reason": "Thinking requires unset sampling parameters", "effects": f"Removed {temp_field}"})
            budget = thinking.get("budget_tokens")
            if type(budget) is int and type(effective.get("max_tokens")) is int and effective["max_tokens"] <= budget:
                old = effective["max_tokens"]
                effective["max_tokens"] = min(max_cap, budget + 4096)
                changes.append({"id": "bump-max-tokens-for-thinking", "path": "max_tokens", "requested": old, "was_missing": False,
                                "effective": effective["max_tokens"], "removed": False, "reason": "max_tokens must exceed thinking.budget_tokens", "effects": "Increased max_tokens"})
    key = "input" if protocol == "responses" else "messages"
    if _deduplicate_history_tool_ids(effective.get(key), protocol):
        changes.append({"id": "dedup-tool-call-ids", "path": key, "requested": "reused_ids", "was_missing": False,
                        "effective": "unique_per_turn", "removed": False, "reason": "Deduplicate reused tool call IDs across turns", "effects": "Renamed prior turn duplicate tool IDs"})
    return effective, changes


def get_path(body, path):
    current = body
    for key in path.split("."):
        if not isinstance(current, dict):
            raise ValueError("Option parent must be an object")
        current = current.get(key, MISSING)
        if current is MISSING:
            return MISSING
    return current


def apply_rules(body, profile):
    effective, changes = copy.deepcopy(body), []
    for rule in profile.rules:
        before = get_path(effective, rule.path)
        matches = (before is MISSING if rule.when == "missing" else
                   before is not MISSING and canonical(before) == canonical(rule.expected))
        if not matches:
            continue
        parent = effective
        keys = rule.path.split(".")
        for key in keys[:-1]:
            parent = parent.setdefault(key, {})
        if rule.operation == "remove":
            parent.pop(keys[-1])
        else:
            parent[keys[-1]] = copy.deepcopy(rule.value)
        changes.append({"id": rule.id, "path": rule.path, "requested": None if before is MISSING else before,
                        "was_missing": before is MISSING, "effective": rule.value if rule.operation == "set" else None,
                        "removed": rule.operation == "remove", "reason": rule.reason, "effects": rule.effects})
    paths = OPTION_PATHS[profile.protocol] | (V2_EXTRA_PATHS[profile.protocol] if isinstance(profile, ProfileV2) else set())
    for path in sorted(paths):
        value = get_path(effective, path)
        if value is MISSING:
            continue
        if isinstance(profile, ProfileV2) and path in profile.constraints:
            if not profile.constraints[path].accepts(value):
                raise RequestPolicyError("option_outside_profile", path)
            continue
        allowed = profile.options.get(path)
        if allowed is None or not any(canonical(value) == canonical(item) for item in allowed):
            raise RequestPolicyError("option_outside_profile", path)
    if isinstance(profile, ProfileV2):
        for path, dependencies in profile.requires.items():
            if get_path(effective, path) is not MISSING:
                for dependency in dependencies:
                    if get_path(effective, dependency) is MISSING:
                        raise RequestPolicyError("required_option_missing", dependency)
        for first, second in profile.conflicts:
            if get_path(effective, first) is not MISSING and get_path(effective, second) is not MISSING:
                raise RequestPolicyError("conflicting_options", first)
    return effective, changes


def prepare(body, profile):
    if not isinstance(body, dict) or body.get("model") != profile.id:
        raise ValueError("Model must match the selected profile")
    pre_changes = []
    if isinstance(profile, ProfileV2) and profile.normalize_client_options:
        body, pre_changes = _normalize_client_body(body, profile)
    protocol, caps = profile.protocol, set(profile.capabilities)
    common = {"model", "stream", "tools", "tool_choice", "temperature", "top_p"}
    allowed = common | {
        "chat": {"messages", "max_tokens", "max_completion_tokens", "reasoning_effort", "parallel_tool_calls", "stream_options", "response_format"},
        "responses": {"input", "instructions", "max_output_tokens", "reasoning", "parallel_tool_calls", "store", "include", "text"},
        "messages": {"messages", "system", "max_tokens", "thinking", "output_config"},
    }[protocol]
    if isinstance(profile, ProfileV2):
        allowed |= {path.split(".")[0] for path in profile.constraints}
    if not set(body) <= allowed:
        raise RequestPolicyError("unsupported_request_field")
    if "stream" in body and type(body["stream"]) is not bool:
        raise ValueError("stream must be boolean")
    effective, changes = apply_rules(body, profile)
    changes = pre_changes + changes
    reasoning_fields = {"effort", "summary"} if isinstance(profile, ProfileV2) else {"effort"}
    for field, known in (("reasoning", reasoning_fields), ("thinking", {"type", "budget_tokens"}),
                         ("stream_options", {"include_usage"}), ("output_config", {"effort", "format"}),
                         ("text", {"format"})):
        if field in effective and (not isinstance(effective[field], dict) or not set(effective[field]) <= known):
            raise RequestPolicyError("unsupported_nested_option", field)
    if effective.get("stream") and "streaming" not in caps:
        raise ValueError("Streaming not qualified")
    if effective.get("tools"):
        if "tools" not in caps or not isinstance(effective["tools"], list):
            raise ValueError("Tools not qualified")
        for tool in effective["tools"]:
            if not isinstance(tool, dict) or (protocol != "messages" and tool.get("type") != "function") or (protocol == "messages" and "type" in tool and tool["type"] != "custom"):
                raise ValueError("Only client-executed function tools are supported")
        if "parallel_tools" not in caps:
            disabled = (effective.get("parallel_tool_calls") is False if protocol != "messages" else
                        isinstance(effective.get("tool_choice"), dict) and effective["tool_choice"].get("disable_parallel_tool_use") is True)
            if not disabled:
                raise ValueError("Parallel tools not qualified; explicit disable required")
    choice = effective.get("tool_choice")
    if choice is not None:
        if protocol == "messages":
            if not isinstance(choice, dict) or not set(choice) <= {"type", "disable_parallel_tool_use"} or choice.get("type") not in {"auto", "none"}:
                raise ValueError("Forced tool choice not covered")
        elif choice not in ("auto", "none"):
            raise ValueError("Forced tool choice not covered")
    structured = (effective.get("response_format") or effective.get("text", {}).get("format") or
                  effective.get("output_config", {}).get("format"))
    if structured and (not isinstance(structured, dict) or
                       structured.get("type") != "text" and "structured_output" not in caps):
        raise ValueError("Structured output not qualified")
    reasoning = (effective.get("reasoning_effort") or effective.get("reasoning", {}).get("effort") or
                 effective.get("thinking", {}).get("type") or effective.get("output_config", {}).get("effort"))
    if reasoning not in (None, "none", "disabled") and "reasoning" not in caps:
        raise ValueError("Reasoning not qualified")
    key = "input" if protocol == "responses" else "messages"
    history = effective.get(key)
    if not ((protocol == "responses" and isinstance(history, str)) or isinstance(history, list) and history):
        raise ValueError("Nonempty history required")
    validate_history(history, protocol, caps)
    if protocol == "responses" and effective.get("store") is not False:
        raise ValueError("Stateless profile requires store=false (or a declared default rule)")
    if protocol == "messages" and (type(effective.get("max_tokens")) is not int or effective["max_tokens"] <= 0):
        raise ValueError("Positive max_tokens required")
    if protocol == "chat" and "max_tokens" in effective and "max_completion_tokens" in effective:
        raise ValueError("Conflicting token limit fields")
    effective["model"] = profile.model
    return effective, changes


def validate_history(history, protocol, caps):
    if isinstance(history, str):
        return
    pending = set()
    closing = False
    for message in history:
        if not isinstance(message, dict):
            raise ValueError("History items must be objects")
        calls, results = [], []
        if protocol == "chat":
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
                raise ValueError("Tool calls must be objects")
            calls = [item.get("id") for item in tool_calls]
            if message.get("role") == "tool":
                results = [message.get("tool_call_id")]
        elif protocol == "responses":
            if message.get("type") == "function_call":
                calls = [message.get("call_id")]
            if message.get("type") == "function_call_output":
                results = [message.get("call_id")]
        else:
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(item, dict) for item in content):
                    raise ValueError("Content blocks must be objects")
                calls = [item.get("id") for item in content if item.get("type") == "tool_use"]
                results = [item.get("tool_use_id") for item in content if item.get("type") == "tool_result"]
        if (calls or results) and "tools" not in caps:
            raise ValueError("Tool history not qualified")
        if calls and pending and (protocol != "responses" or closing):
            raise ValueError("Missing tool results in replay")
        for identifier in calls:
            if not isinstance(identifier, str) or not identifier or identifier in pending:
                raise ValueError("Missing or reused tool call identity")
            pending.add(identifier)
        if len(pending) > 1 and "parallel_tools" not in caps:
            raise ValueError("Parallel tool history not qualified")
        for identifier in results:
            if not isinstance(identifier, str) or identifier not in pending:
                raise ValueError("Unassociated or duplicate tool result")
            pending.remove(identifier)
            closing = bool(pending)
    if pending:
        raise ValueError("Missing tool results in replay")