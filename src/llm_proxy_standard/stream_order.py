"""Opt-in item-ordered Responses SSE, after upstream validation.

Never manufacture completion. Text in the leading item remains incremental;
later items wait, and reasoning waits for its final opaque payload. This trades
latency for compatibility and must be declared in the model profile.
"""
import copy
from .contracts import canonical, strict_json
from .protocol import ProtocolError


class OrderedItems:
    def __init__(self, limit):
        self.limit = limit
        self.size = 0
        self.next_index = 0
        self.sequence = 0
        self.pending = {}
        self.pending_sizes = {}
        self.reasoning = set()
        self.closed = set()

    def encode(self, body):
        value = copy.deepcopy(body)
        value["sequence_number"] = self.sequence
        self.sequence += 1
        return ("event: " + value["type"] + "\ndata: " + canonical(value).decode() + "\n\n").encode()

    def feed(self, raw):
        data = "\n".join(line[5:].removeprefix(" ") for line in raw.decode().splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            return [raw]
        body = strict_json(data)
        index = body.get("output_index")
        if index is None:
            if body["type"] in {"response.completed", "response.incomplete", "response.failed"}:
                remaining = []
                for pending_index in sorted(self.pending):
                    remaining.extend(self.encode(item) for item in self.pending[pending_index])
                self.pending.clear()
                self.pending_sizes.clear()
                self.size = 0
                return remaining + [self.encode(body)]
            return [self.encode(body)]
        if body["type"] == "response.output_item.added" and body["item"]["type"] == "reasoning":
            self.reasoning.add(index)
        self.pending.setdefault(index, []).append(body)
        self.pending_sizes[index] = self.pending_sizes.get(index, 0) + len(raw)
        self.size += len(raw)
        if self.size > self.limit:
            raise ProtocolError("Ordered item buffer exceeds limit")
        if body["type"] == "response.output_item.done":
            self.closed.add(index)
            if index in self.reasoning and "encrypted_content" in body["item"]:
                # Copy only data actually received in the validated final item.
                for initial in self.pending[index]:
                    if initial["type"] == "response.output_item.added":
                        initial["item"]["encrypted_content"] = body["item"]["encrypted_content"]
                        extra = len(canonical(body["item"]["encrypted_content"]))
                        self.size += extra
                        self.pending_sizes[index] += extra
                        if self.size > self.limit:
                            raise ProtocolError("Ordered item buffer exceeds limit")
        output = []
        while self.next_index in self.pending:
            if self.next_index in self.reasoning and self.next_index not in self.closed:
                break
            events = self.pending.pop(self.next_index)
            output.extend(self.encode(item) for item in events)
            self.size -= self.pending_sizes.pop(self.next_index)
            if self.next_index not in self.closed:
                break
            self.next_index += 1
        return output