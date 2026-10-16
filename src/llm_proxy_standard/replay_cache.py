"""Bounded in-memory LRU cache to rehydrate opaque reasoning and thinking blocks.

When stateless or partially stateful clients (such as unpatched n8n Agent engine
serialization) drop `reasoning.encrypted_content` on Responses or `thinking` /
`redacted_thinking` blocks on Messages across tool-calling turns, this cache
restores the exact blocks emitted in the prior response keyed by tool call ID.
"""
from collections import OrderedDict
import copy
import time


class ReplayCache:
    def __init__(self, max_entries=512, ttl=3600.0, monotonic=time.monotonic):
        self.max_entries = max_entries
        self.ttl = ttl
        self.monotonic = monotonic
        self._store = OrderedDict()

    def _evict(self, now):
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def put(self, protocol, call_ids, blocks):
        if not call_ids or not blocks:
            return
        now = self.monotonic()
        self._evict(now)
        frozen = copy.deepcopy(blocks)
        for cid in call_ids:
            if isinstance(cid, str) and cid:
                key = (protocol, cid)
                self._store.pop(key, None)
                self._store[key] = (now, frozen)
        self._evict(now)

    def get(self, protocol, call_id):
        if not isinstance(call_id, str) or not call_id:
            return None
        now = self.monotonic()
        self._evict(now)
        key = (protocol, call_id)
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return copy.deepcopy(entry[1])

    def record_document(self, protocol, document):
        if not isinstance(document, dict):
            return
        if protocol == "responses":
            output = document.get("output")
            if not isinstance(output, list):
                return
            reasoning = [item for item in output if isinstance(item, dict)
                         and item.get("type") == "reasoning" and item.get("encrypted_content")]
            calls = [item.get("call_id") for item in output if isinstance(item, dict)
                     and item.get("type") == "function_call" and item.get("call_id")]
            if reasoning and calls:
                self.put("responses", calls, reasoning)
        elif protocol == "messages":
            content = document.get("content")
            if not isinstance(content, list):
                return
            thinking = [item for item in content if isinstance(item, dict)
                        and item.get("type") in {"thinking", "redacted_thinking"}
                        and (item.get("type") != "thinking" or item.get("signature"))]
            calls = [item.get("id") for item in content if isinstance(item, dict)
                     and item.get("type") == "tool_use" and item.get("id")]
            if thinking and calls:
                self.put("messages", calls, thinking)

    def record_stream(self, protocol, validator):
        if protocol == "responses":
            output = [state["done"] for _, state in sorted(validator.items.items()) if state.get("done")]
            self.record_document("responses", {"output": output})
        elif protocol == "messages":
            content = [state["block"] for _, state in sorted(validator.blocks.items()) if state.get("block")]
            self.record_document("messages", {"content": content})

    def rehydrate(self, body, protocol):
        if not isinstance(body, dict):
            return body, []
        changes = []
        if protocol == "responses":
            items = body.get("input")
            if not isinstance(items, list):
                return body, []
            updated = copy.deepcopy(body)
            new_input = []
            idx = 0
            raw_items = updated["input"]
            rehydrated_any = False
            while idx < len(raw_items):
                item = raw_items[idx]
                if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id"):
                    cached = self.get("responses", item["call_id"])
                    if cached:
                        has_valid_reasoning = (
                            bool(new_input) and isinstance(new_input[-1], dict)
                            and new_input[-1].get("type") == "reasoning"
                            and bool(new_input[-1].get("encrypted_content"))
                        )
                        if not has_valid_reasoning:
                            if new_input and isinstance(new_input[-1], dict) and new_input[-1].get("type") == "reasoning":
                                new_input.pop()
                            for r_block in cached:
                                new_input.append(r_block)
                            rehydrated_any = True
                    while idx < len(raw_items) and isinstance(raw_items[idx], dict) and raw_items[idx].get("type") == "function_call":
                        new_input.append(raw_items[idx])
                        idx += 1
                    continue
                new_input.append(item)
                idx += 1
            if rehydrated_any:
                updated["input"] = new_input
                changes.append({"id": "rehydrate-responses-reasoning", "path": "input", "requested": "missing_encrypted_content",
                                "was_missing": True, "effective": "restored_from_cache", "removed": False,
                                "reason": "Restored dropped reasoning.encrypted_content for tool roundtrip",
                                "effects": "Rehydrated opaque reasoning items"})
                return updated, changes
        elif protocol == "messages":
            messages = body.get("messages")
            if not isinstance(messages, list):
                return body, []
            updated = copy.deepcopy(body)
            rehydrated_any = False
            for msg in updated["messages"]:
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                tool_ids = [b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")]
                if not tool_ids:
                    continue
                cached = self.get("messages", tool_ids[0])
                if not cached:
                    continue
                existing_thinking = [b for b in content if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}]
                valid_existing = (
                    len(existing_thinking) == len(cached)
                    and all(
                        (a.get("type") == "thinking" and a.get("signature") == b.get("signature") and a.get("thinking") == b.get("thinking"))
                        or (a.get("type") == "redacted_thinking" and a.get("data") == b.get("data"))
                        for a, b in zip(existing_thinking, cached)
                    )
                )
                if not valid_existing:
                    non_thinking = [b for b in content if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})]
                    msg["content"] = cached + non_thinking
                    rehydrated_any = True
            if rehydrated_any:
                changes.append({"id": "rehydrate-messages-thinking", "path": "messages", "requested": "missing_thinking_signature",
                                "was_missing": True, "effective": "restored_from_cache", "removed": False,
                                "reason": "Restored dropped thinking/redacted_thinking blocks for tool roundtrip",
                                "effects": "Rehydrated signed thinking blocks"})
                return updated, changes
        return body, []
