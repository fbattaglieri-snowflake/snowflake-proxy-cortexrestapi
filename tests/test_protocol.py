import copy
import unittest
from llm_proxy_standard.protocol import SSEParser, Stream, ProtocolError, validate_document
from llm_proxy_standard.rules import prepare
from tests.support import profile
from tests.test_app import events, sse, output, payload


class ProtocolTests(unittest.TestCase):
    def test_chunk_splitting_cr_lf_crlf_and_unicode(self):
        for separator in (b"\n", b"\r", b"\r\n"):
            parser = SSEParser()
            raw = b'data: "' + "caffe\u0301".encode() + b'"' + separator + separator
            frames = []
            for value in raw:
                frames.extend(parser.feed(bytes([value])))
            parser.finish()
            self.assertEqual(len(frames), 1)
            self.assertIn("caffe\u0301", frames[0].decode())

    def test_incomplete_oversize_duplicate_json(self):
        parser = SSEParser(limit=10)
        with self.assertRaises(ProtocolError):
            list(parser.feed(b"x" * 11))
        parser = SSEParser()
        list(parser.feed(b"data: {}\n"))
        with self.assertRaises(ProtocolError):
            parser.finish()

    def test_missing_terminal_rejected_all_protocols(self):
        for protocol in ("chat", "messages", "responses"):
            validator = Stream(protocol, profile(protocol))
            parser = SSEParser()
            values = events(protocol)[:-1]
            for event in parser.feed(sse(protocol, values)):
                validator.event(event)
            with self.assertRaises(ProtocolError):
                validator.finish()

    def test_parallel_chat_interleaved_and_collision(self):
        values = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "lookup", "arguments": "{"}},
                {"index": 1, "id": "b", "function": {"name": "lookup", "arguments": "{"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "}"}}, {"index": 0, "function": {"arguments": "}"}}]}, "finish_reason": "tool_calls"}]},
            "[DONE]"]
        validator = Stream("chat", profile(capabilities=["text", "tools", "parallel_tools"]))
        for event in SSEParser().feed(sse("chat", values)):
            validator.event(event)
        validator.finish()
        bad = copy.deepcopy(values)
        bad[0]["choices"][0]["delta"]["tool_calls"][1]["index"] = 0
        with self.assertRaises(ProtocolError):
            validator = Stream("chat", profile(capabilities=["text", "tools", "parallel_tools"]))
            for event in SSEParser().feed(sse("chat", bad)):
                validator.event(event)

    def test_responses_text_parity_and_opaque_reasoning(self):
        values = events("responses")
        values[3]["text"] = "WRONG"
        with self.assertRaises(ProtocolError):
            validator = Stream("responses", profile("responses"))
            for event in SSEParser().feed(sse("responses", values)):
                validator.event(event)

    def test_history_pairing_and_text_only_tool_bypass(self):
        body = payload("chat")
        body["messages"].append({"role": "tool", "tool_call_id": "missing", "content": "x"})
        for caps in (["text"], ["text", "tools", "parallel_tools"]):
            with self.assertRaises(ValueError):
                prepare(body, profile(capabilities=caps))

    def test_malformed_tool_history_is_rejected_as_input(self):
        for fragment in ({"tool_calls": [None]}, {"tool_calls": None},
                         {"role": "tool", "tool_call_id": []}):
            request = payload("chat")
            request["messages"].append(fragment)
            with self.assertRaises(ValueError):
                prepare(request, profile(capabilities=["text", "tools"]))

    def test_unqualified_structured_reasoning_stream_and_options(self):
        for extra in ({"stream": True}, {"response_format": {"type": "json_schema"}},
                      {"reasoning_effort": "high"}, {"temperature": 0.5}):
            with self.assertRaises(ValueError):
                prepare({**payload("chat"), **extra}, profile())

    def test_length_and_refusal_are_native_states(self):
        for reason in ("length", "content_filter"):
            value = output("chat")
            value["choices"][0]["finish_reason"] = reason
            validate_document(value, "chat")
        value = output("messages")
        value["stop_reason"] = "refusal"
        validate_document(value, "messages")