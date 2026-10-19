"""Run existing replay assertions with explicit ordered-items profiles."""
import copy
import unittest
import httpx
from llm_proxy_standard.contracts import ProfileV2, strict_json
from llm_proxy_standard.protocol import SSEParser, Stream, ProtocolError
from llm_proxy_standard.stream_order import OrderedItems
from tests import test_stream_replay, test_app
from tests.support import profile


class OrderedSDKTests(test_stream_replay.SDKStreamReplayTests):
    async def asyncSetUp(self):
        await test_app.AppTests.asyncSetUp(self)
        entries = []
        for entry in self.cache.snapshot.entries.values():
            value = entry.proposal.profile
            if value.protocol == "responses":
                value = ProfileV2.model_validate({**value.document(), "schema_version": 2,
                                                  "response_stream": "ordered-items"})
            entries.append(self.authority.entry(value))
        self.document["entries"] = {entry.proposal.profile.id: entry.document() for entry in entries}
        self.cache.snapshot = None
        self.assertTrue(await self.cache.refresh())

    async def test_invalid_stream_never_retries_or_fabricates_completion(self):
        values, _ = test_stream_replay.responses_tools(2)
        wrong_identity = copy.deepcopy(values)
        wrong_identity[-2]["item"]["call_id"] = "wrong-call"
        duplicate_terminal = values + [copy.deepcopy(values[-1])]
        duplicate_terminal[-1]["sequence_number"] += 1
        for broken in (values[:-1], wrong_identity, duplicate_terminal):
            sent, streams = [], []
            def handler(request):
                sent.append(request)
                stream = test_app.ByteStream(test_app.sse("responses", broken))
                streams.append(stream)
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
            self.handler = handler
            with self.assertRaises(Exception):
                await self.http.post("/responses/v1/responses", json={**test_app.payload("responses"), "stream": True})
            self.assertEqual(len(sent), 1)
            self.assertTrue(all(stream.closed for stream in streams))
            self.assertEqual(self.app.state.admission.slots._value, 8)

    async def test_native_failure_and_incomplete_remain_terminal_states(self):
        for status in ("incomplete", "failed"):
            values = test_app.events("responses")
            values[-1]["type"] = "response." + status
            values[-1]["response"]["status"] = status
            values[-1]["response"]["incomplete_details"] = {"reason": "max_output_tokens"} if status == "incomplete" else None
            self.handler = lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"},
                stream=test_app.ByteStream(test_app.sse("responses", values)))
            response = await self.http.post("/responses/v1/responses", json={**test_app.payload("responses"), "stream": True})
            self.assertEqual(response.headers["x-proxy-stream-mode"], "ordered-items")
            frames = list(SSEParser().feed(response.content))
            last = strict_json(next(line[6:] for line in frames[-1].decode().splitlines() if line.startswith("data: ")))
            self.assertEqual(last["type"], "response." + status)
            self.assertEqual(last["response"]["status"], status)
            self.assertNotIn(b'"type":"response.completed"', response.content)


class OrderingTests(unittest.TestCase):
    def test_reordered_wire_validates_and_preserves_completed_document(self):
        for initial_mode in ("missing", "empty", "placeholder"):
            values, expected = test_stream_replay.responses_tools(2)
            if initial_mode == "missing":
                del values[1]["item"]["encrypted_content"]
            elif initial_mode == "empty":
                values[1]["item"]["encrypted_content"] = ""
            else:
                values[1]["item"]["encrypted_content"] = "initial-placeholder"
            value = profile("responses", capabilities=["text", "tools", "parallel_tools", "reasoning", "streaming"])
            order = OrderedItems(1024 * 1024)
            frames = []
            for raw in SSEParser().feed(test_app.sse("responses", values)):
                frames.extend(order.feed(raw))
            validator = Stream("responses", value)
            for frame in frames:
                validator.event(frame)
            validator.finish()
            self.assertEqual([validator.items[index]["done"] for index in range(3)], expected)
            self.assertEqual(validator.items[0]["initial"]["encrypted_content"], "synthetic-encrypted")

    def test_text_remains_incremental_and_buffer_bounded(self):
        order = OrderedItems(4096)
        for raw in SSEParser().feed(test_app.sse("responses")):
            self.assertEqual(len(order.feed(raw)), 1)
        values, _ = test_stream_replay.responses_tools(2)
        order = OrderedItems(8)
        with self.assertRaises(ProtocolError):
            for raw in SSEParser().feed(test_app.sse("responses", values)):
                order.feed(raw)