"""Incremental native SSE validation; never invent a success terminal or tool ID."""
import copy
from .contracts import canonical, strict_json


class ProtocolError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ProtocolError(message)


def map_model(body, protocol, profile):
    value = copy.deepcopy(body)
    targets = [value]
    if protocol == "responses" and isinstance(value.get("response"), dict):
        targets.append(value["response"])
    if protocol == "messages" and value.get("type") == "message_start" and isinstance(value.get("message"), dict):
        targets.append(value["message"])
    for target in targets:
        if "model" in target:
            require(target["model"] == profile.model, "Unexpected upstream model")
            target["model"] = profile.id
    return value


def validate_document(body, protocol):
    require(isinstance(body, dict), "Invalid response object")
    if protocol == "chat":
        require(isinstance(body.get("choices"), list) and body["choices"], "Missing choices")
        for choice in body["choices"]:
            require(isinstance(choice, dict), "Invalid choice")
            require(choice.get("finish_reason") in {"stop", "length", "tool_calls", "content_filter"}, "Missing/unknown finish reason")
            message = choice.get("message")
            require(isinstance(message, dict), "Missing assistant message")
            if choice["finish_reason"] == "tool_calls":
                validate_tools(message.get("tool_calls", []), "chat")
    elif protocol == "messages":
        require(body.get("type") == "message" and isinstance(body.get("content"), list), "Invalid message")
        require(all(isinstance(item, dict) for item in body["content"]), "Invalid message blocks")
        require(body.get("stop_reason") in {"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn"}, "Missing/unknown stop reason")
        tools = [item for item in body["content"] if item.get("type") == "tool_use"]
        if body["stop_reason"] == "tool_use":
            validate_tools(tools, "messages")
    else:
        require(body.get("object") == "response" and isinstance(body.get("output"), list), "Invalid Responses object")
        require(all(isinstance(item, dict) for item in body["output"]), "Invalid output items")
        require(body.get("status") in {"completed", "incomplete", "failed"}, "Missing terminal response status")
        if body["status"] == "completed":
            tools = [item for item in body["output"] if item.get("type") == "function_call"]
            if tools:
                validate_tools(tools, "responses")


def validate_tools(items, protocol):
    require(isinstance(items, list) and bool(items), "Tool stop without tool calls")
    ids = set()
    for item in items:
        require(isinstance(item, dict), "Invalid tool call")
        identifier = item.get("call_id" if protocol == "responses" else "id")
        require(isinstance(identifier, str) and identifier and identifier not in ids, "Missing/duplicate tool identity")
        ids.add(identifier)
        function = item.get("function", {}) if protocol == "chat" else item
        require(isinstance(function, dict), "Invalid function")
        require(isinstance(function.get("name"), str) and function["name"], "Missing tool name")
        arguments = function.get("input") if protocol == "messages" else strict_json(function.get("arguments", ""))
        require(isinstance(arguments, dict), "Tool arguments must be an object")


def validate_capabilities(body, protocol, profile):
    if protocol == "chat":
        counts = [len(choice.get("message", {}).get("tool_calls") or []) for choice in body.get("choices", [])]
    elif protocol == "messages":
        counts = [sum(item.get("type") == "tool_use" for item in body.get("content", []))]
    else:
        counts = [sum(item.get("type") == "function_call" for item in body.get("output", []))]
    for count in counts:
        require(not count or "tools" in profile.capabilities, "Unqualified tool output")
        require(count <= 1 or "parallel_tools" in profile.capabilities, "Unqualified parallel output")


class SSEParser:
    def __init__(self, limit=2 * 1024 * 1024):
        self.buffer = bytearray()
        self.limit = limit
        self.skip_lf = False
        self.lines = []
        self.size = 0

    def feed(self, chunk):
        # Normalize framing, never JSON values or opaque content.
        for value in chunk:
            if self.skip_lf and value == 10:
                self.skip_lf = False
                continue
            self.skip_lf = value == 13
            self.size += 1
            require(self.size <= self.limit, "SSE frame exceeds limit")
            if value in (10, 13):
                if self.buffer:
                    self.lines.append(bytes(self.buffer))
                    self.buffer.clear()
                else:
                    yield b"\n".join(self.lines) + b"\n\n"
                    self.lines.clear()
                    self.size = 0
            else:
                self.buffer.append(value)

    def finish(self):
        require(not self.buffer.strip() and not self.lines, "Unterminated SSE event")


class Stream:
    def __init__(self, protocol, profile):
        self.protocol, self.profile = protocol, profile
        self.terminal = False
        self.done = False
        self.started = False
        self.choices, self.calls, self.blocks, self.items = {}, {}, {}, {}
        self.stop_reason = None
        self.response_id = None
        self.sequence = -1
        self.message_stopped = False

    def event(self, raw):
        lines = raw.decode("utf-8").splitlines()
        data = "\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:"))
        if not data:
            return raw
        if data == "[DONE]":
            require(not self.done, "Duplicate DONE")
            if self.protocol == "chat":
                require(self.choices and all(self.choices.values()), "DONE without stopped choices")
                self.terminal = True
            else:
                require(self.protocol == "responses" and self.terminal, "Unexpected DONE")
            self.done = True
            return raw
        require(not self.terminal, "Data after terminal")
        body = strict_json(data)
        require(isinstance(body, dict), "Invalid SSE object")
        names = [line[6:].strip() for line in lines if line.startswith("event:")]
        if names and self.protocol != "chat":
            require(len(names) == 1 and names[0] == body.get("type"), "Event type mismatch")
        if body.get("error") or body.get("type") == "error":
            raise ProtocolError("Upstream stream error")
        if self.protocol == "chat":
            self.chat(body)
        elif self.protocol == "messages":
            self.messages(body)
        else:
            self.responses(body)
        mapped = map_model(body, self.protocol, self.profile)
        if mapped == body:
            return raw
        metadata = [line for line in lines if line and not line.startswith("data:")]
        return ("\n".join(metadata) + ("\n" if metadata else "") + "data: " + canonical(mapped).decode() + "\n\n").encode()

    def chat(self, body):
        for choice in body.get("choices", []):
            index = choice.get("index")
            require(type(index) is int and index >= 0 and not self.choices.get(index), "Invalid choice index/state")
            self.choices.setdefault(index, False)
            for fragment in (choice.get("delta") or {}).get("tool_calls", []):
                require("tools" in self.profile.capabilities, "Unqualified tool stream")
                tool_index = fragment.get("index")
                require(type(tool_index) is int and tool_index >= 0, "Missing tool index")
                key = (index, tool_index)
                state = self.calls.setdefault(key, {"id": None, "function": {"name": "", "arguments": ""}})
                require(sum(other[0] == index for other in self.calls) <= 1 or "parallel_tools" in self.profile.capabilities, "Unqualified parallel stream")
                if fragment.get("id"):
                    require(state["id"] in (None, fragment["id"]), "Tool ID collision")
                    state["id"] = fragment["id"]
                    require(not any(other != key and item["id"] == state["id"] for other, item in self.calls.items()), "Tool ID moved index")
                for field in ("name", "arguments"):
                    state["function"][field] += fragment.get("function", {}).get(field) or ""
            reason = choice.get("finish_reason")
            if reason:
                require(reason in {"stop", "length", "tool_calls", "content_filter"}, "Unknown finish reason")
                calls = [item for key, item in self.calls.items() if key[0] == index]
                if reason == "tool_calls":
                    validate_tools(calls, "chat")
                elif calls:
                    require(reason in {"length", "content_filter"}, "Tools incorrectly terminated")
                self.choices[index] = True

    def messages(self, body):
        kind = body.get("type")
        if kind == "ping":
            return
        if kind == "message_start":
            require(not self.started, "Duplicate message start")
            message = body.get("message")
            require(isinstance(message, dict) and message.get("type") == "message" and
                    message.get("role") == "assistant" and message.get("id") and
                    message.get("content") == [] and message.get("stop_reason") is None,
                    "Invalid message start")
            self.started = True
            return
        require(self.started, "Missing message start")
        require(not self.message_stopped or kind == "message_stop", "Content after message delta")
        if kind == "content_block_start":
            index = body.get("index")
            require(type(index) is int and index == len(self.blocks) and
                    all(state["closed"] for state in self.blocks.values()), "Invalid block index/lifecycle")
            block = body.get("content_block")
            require(isinstance(block, dict) and block.get("type") in
                    {"text", "tool_use", "thinking", "redacted_thinking"}, "Unsupported block")
            self.blocks[index] = {"block": copy.deepcopy(block), "arguments": "", "closed": False,
                                  "signature_started": False}
            if block["type"] == "thinking":
                require(isinstance(block.get("thinking"), str) and
                        isinstance(block.get("signature", ""), str), "Invalid thinking block")
            elif block["type"] == "redacted_thinking":
                require(isinstance(block.get("data"), str) and block["data"], "Invalid opaque thinking")
            elif block["type"] == "text":
                require(isinstance(block.get("text"), str), "Invalid text block")
            if block.get("type") == "tool_use":
                require("tools" in self.profile.capabilities, "Unqualified tool stream")
                require(isinstance(block.get("name"), str) and block["name"] and
                        block.get("input") == {}, "Invalid tool start")
                ids = [state["block"].get("id") for state in self.blocks.values() if state["block"].get("type") == "tool_use"]
                require(all(ids) and len(set(ids)) == len(ids), "Missing/duplicate tool identity")
                require(len(ids) <= 1 or "parallel_tools" in self.profile.capabilities, "Unqualified parallel stream")
        elif kind in {"content_block_delta", "content_block_stop"}:
            index = body.get("index")
            require(type(index) is int, "Invalid block index")
            state = self.blocks.get(index)
            require(state is not None and not state["closed"], "Block not open")
            if kind.endswith("delta"):
                delta = body.get("delta")
                require(isinstance(delta, dict), "Invalid block delta")
                allowed = {"text": {"text_delta": "text"}, "tool_use": {"input_json_delta": "partial_json"},
                           "thinking": {"thinking_delta": "thinking", "signature_delta": "signature"},
                           "redacted_thinking": {}}
                field = allowed[state["block"]["type"]].get(delta.get("type"))
                require(field is not None and isinstance(delta.get(field), str), "Delta/block type mismatch")
                if field == "partial_json":
                    state["arguments"] += delta[field]
                else:
                    require(field != "thinking" or not state["signature_started"], "Thinking after signature")
                    state["signature_started"] |= field == "signature"
                    state["block"][field] = state["block"].get(field, "") + delta[field]
            else:
                state["closed"] = True
        elif kind == "message_delta":
            require(all(state["closed"] for state in self.blocks.values()), "Message delta before block stop")
            self.stop_reason = body.get("delta", {}).get("stop_reason")
            require(self.stop_reason is not None, "Missing message stop reason")
            self.message_stopped = True
        elif kind == "message_stop":
            require(self.message_stopped and all(state["closed"] for state in self.blocks.values()), "Unclosed message")
            content = []
            for state in self.blocks.values():
                block = copy.deepcopy(state["block"])
                if block.get("type") == "tool_use" and state["arguments"]:
                    if self.stop_reason == "tool_use":
                        block["input"] = strict_json(state["arguments"])
                if block.get("type") == "thinking" and self.stop_reason not in {"max_tokens", "refusal"}:
                    require(bool(block.get("signature")), "Missing thinking signature")
                content.append(block)
            validate_document({"type": "message", "content": content, "stop_reason": self.stop_reason}, "messages")
            self.terminal = True
        else:
            raise ProtocolError("Unsupported Messages event")

    def responses(self, body):
        kind = body.get("type")
        sequence = body.get("sequence_number")
        require(type(sequence) is int and sequence > self.sequence, "Invalid Responses sequence")
        self.sequence = sequence
        if kind == "response.created":
            require(self.response_id is None and body.get("response", {}).get("id"), "Invalid response start")
            self.response_id = body["response"]["id"]
            return
        require(self.response_id is not None, "Missing response.created")
        if kind in {"response.in_progress", "response.completed", "response.incomplete", "response.failed"}:
            document = body.get("response", {})
            require(document.get("id") == self.response_id, "Response identity changed")
            if kind != "response.in_progress":
                require(document.get("status") == kind.split(".")[1], "Status/event mismatch")
                validate_document(document, "responses")
                if kind == "response.completed":
                    require(len(self.items) == len(document["output"]), "Missing output items")
                    for index, item in enumerate(document["output"]):
                        require(self.items.get(index, {}).get("done") == item, "Done/completed mismatch")
                self.terminal = True
            return
        index = body.get("output_index")
        require(type(index) is int and index >= 0, "Missing output index")
        if kind == "response.output_item.added":
            item = body["item"]
            require(index not in self.items and item.get("id"), "Duplicate/missing output identity")
            require(not any(state["initial"]["id"] == item["id"] for state in self.items.values()), "Duplicate item ID")
            if item.get("type") == "function_call":
                require("tools" in self.profile.capabilities, "Unqualified tool stream")
                require(item.get("call_id") and not any(state["initial"].get("call_id") == item["call_id"] for state in self.items.values()), "Duplicate/missing call ID")
                require(not any(state["initial"].get("type") == "function_call" for state in self.items.values()) or "parallel_tools" in self.profile.capabilities, "Unqualified parallel stream")
            self.items[index] = {"initial": item, "done": None, "arguments": item.get("arguments", ""), "arguments_done": False, "text": {}, "text_done": set(), "parts": {}}
            return
        state = self.items.get(index)
        require(state is not None and state["done"] is None, "Item not open")
        if kind == "response.output_item.done":
            item = body["item"]
            for key in ("id", "type", "call_id", "name"):
                require(item.get(key) == state["initial"].get(key), "Output identity changed")
            if item.get("type") == "function_call":
                require(state["arguments_done"] and item["arguments"] == state["arguments"], "Arguments incomplete")
                if item.get("status") != "incomplete":
                    validate_tools([item], "responses")
            for (field, part_index), part in state["parts"].items():
                parts = item.get(field, [])
                require(part["done"] is not None and part_index < len(parts) and
                        parts[part_index] == part["done"], "Part done/item mismatch")
            for part_key, text in state["text"].items():
                field, part_index, text_key = part_key
                parts = item.get(field, [])
                require(part_key in state["text_done"] and part_index < len(parts) and parts[part_index].get(text_key) == text, "Text delta/done/item mismatch")
            if state["initial"].get("encrypted_content") is not None:
                require(item.get("encrypted_content") == state["initial"]["encrypted_content"], "Opaque reasoning changed")
            state["done"] = item
            return
        require(body.get("item_id") == state["initial"]["id"], "Item identity mismatch")
        if "call_id" in body:
            require(body["call_id"] == state["initial"].get("call_id"), "Call identity mismatch")
        if kind == "response.function_call_arguments.delta":
            require(state["initial"].get("type") == "function_call" and
                    not state["arguments_done"] and isinstance(body.get("delta"), str), "Invalid arguments delta")
            state["arguments"] += body["delta"]
        elif kind == "response.function_call_arguments.done":
            require(state["initial"].get("type") == "function_call" and
                    not state["arguments_done"] and body["arguments"] == state["arguments"], "Argument delta mismatch")
            state["arguments_done"] = True
        else:
            families = {"response.output_text", "response.refusal", "response.content_part", "response.reasoning_summary_part", "response.reasoning_summary_text", "response.reasoning_text"}
            family, _, action = kind.rpartition(".")
            require(family in families and action in {"added", "delta", "done"}, "Unsupported Responses event")
            if family in {"response.content_part", "response.reasoning_summary_part"}:
                field = "summary" if family == "response.reasoning_summary_part" else "content"
                part_index = body.get("summary_index" if field == "summary" else "content_index")
                require(type(part_index) is int and part_index >= 0, "Invalid part index")
                key = (field, part_index)
                part = body.get("part")
                require(isinstance(part, dict), "Invalid part")
                if action == "added":
                    require(key not in state["parts"], "Duplicate part")
                    state["parts"][key] = {"initial": part, "done": None}
                else:
                    require(action == "done" and key in state["parts"] and
                            state["parts"][key]["done"] is None, "Part not open")
                    require(part.get("type") == state["parts"][key]["initial"].get("type"), "Part type changed")
                    for part_key, text in state["text"].items():
                        if part_key[:2] == key:
                            require(part_key in state["text_done"] and part.get(part_key[2]) == text,
                                    "Part/text mismatch")
                    state["parts"][key]["done"] = part
            if family in {"response.output_text", "response.refusal", "response.reasoning_summary_text", "response.reasoning_text"}:
                field = "summary" if family == "response.reasoning_summary_text" else "content"
                part_index = body.get("summary_index" if field == "summary" else "content_index")
                require(type(part_index) is int and part_index >= 0, "Invalid content index")
                text_key = "refusal" if family == "response.refusal" else "text"
                part_key = (field, part_index, text_key)
                part = state["parts"].get((field, part_index))
                require(part is None or part["done"] is None, "Text after part done")
                require(part_key not in state["text_done"], "Text already done")
                if action == "delta":
                    require(isinstance(body.get("delta"), str), "Invalid text delta")
                    state["text"][part_key] = state["text"].get(part_key, "") + body["delta"]
                else:
                    require(action == "done" and body.get(text_key) == state["text"].get(part_key, ""), "Text delta/done mismatch")
                    state["text"].setdefault(part_key, "")
                    state["text_done"].add(part_key)

    def finish(self):
        require(self.terminal, "Missing terminal event")