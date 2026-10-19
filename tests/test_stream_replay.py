"""SDK reconstruction and exact replay of opaque data; no real signing/inference."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from llm_proxy_standard.contracts import strict_json
from llm_proxy_standard.protocol import ProtocolError, SSEParser, Stream
from tests import test_app
from tests.support import profile


def messages_tools(count):
    values = test_app.events("messages")[:1]
    blocks = [{"type": "thinking", "thinking": "synthetic thought", "signature": "synthetic-signature"},
              {"type": "redacted_thinking", "data": "synthetic-opaque"}]
    blocks += [{"type": "tool_use", "id": f"call-{index}", "name": "lookup", "input": {"key": index}}
               for index in range(count)]
    for index, block in enumerate(blocks):
        initial = copy.deepcopy(block)
        deltas = []
        if block["type"] == "thinking":
            initial.update(thinking="", signature="")
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]},
                      {"type": "signature_delta", "signature": block["signature"]}]
        elif block["type"] == "tool_use":
            initial["input"] = {}
            deltas = [{"type": "input_json_delta", "partial_json": '{"key":'},
                      {"type": "input_json_delta", "partial_json": str(block["input"]["key"]) + "}"}]
        values.append({"type": "content_block_start", "index": index, "content_block": initial})
        values += [{"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas]
        values.append({"type": "content_block_stop", "index": index})
    values += [{"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 10}},
               {"type": "message_stop"}]
    return values, blocks


def responses_tools(count):
    document = test_app.output("responses")
    output = [{"id": "reasoning-1", "type": "reasoning", "summary": [], "encrypted_content": "synthetic-encrypted"}]
    output += [{"id": f"fc-{index}", "type": "function_call", "call_id": f"call-{index}",
                "name": "lookup", "arguments": '{"key":' + str(index) + "}", "status": "completed"}
               for index in range(count)]
    values = [{"type": "response.created", "response": {**document, "status": "in_progress", "output": []}}]
    for index, item in enumerate(output):
        initial = copy.deepcopy(item)
        if item["type"] == "function_call":
            initial.update(arguments="", status="in_progress")
        values.append({"type": "response.output_item.added", "output_index": index, "item": initial})
    for index, item in enumerate(output):
        if item["type"] == "function_call":
            values += [{"type": "response.function_call_arguments.delta", "output_index": index,
                        "item_id": item["id"], "delta": item["arguments"]},
                       {"type": "response.function_call_arguments.done", "output_index": index,
                        "item_id": item["id"], "arguments": item["arguments"]}]
        values.append({"type": "response.output_item.done", "output_index": index, "item": copy.deepcopy(item)})
    values.append({"type": "response.completed", "response": {**document, "output": copy.deepcopy(output)}})
    for sequence, value in enumerate(values):
        value["sequence_number"] = sequence
    return values, output


class SDKStreamReplayTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown

    async def test_langchain_streamed_parallel_tools_and_opaque_replay(self):
        for protocol in ("messages", "responses"):
            sent = []
            values, _ = messages_tools(2) if protocol == "messages" else responses_tools(2)
            def handler(request):
                sent.append(strict_json(request.content))
                if len(sent) == 1:
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                          stream=test_app.ByteStream(test_app.sse(protocol, values)))
                return httpx.Response(200, json=test_app.output(protocol))
            self.handler = handler
            if protocol == "messages":
                model = ChatAnthropic(model="fixture-messages", api_key="synthetic", max_tokens=32,
                                      max_retries=0, base_url="http://proxy.invalid/messages",
                                      thinking={"type": "adaptive"})
            else:
                model = ChatOpenAI(model="fixture-responses", api_key="synthetic", max_retries=0,
                                   base_url="http://proxy.invalid/responses/v1", http_async_client=self.http,
                                   use_responses_api=True, extra_body={"store": False})
            model = model.bind_tools([{"name": "lookup", "description": "Synthetic lookup", "parameters":
                                       {"type": "object", "properties": {"key": {"type": "integer"}}}}])
            with patch("langchain_anthropic.chat_models._get_default_async_httpx_client", return_value=self.http):
                chunks = [chunk async for chunk in model.astream([HumanMessage(content="test")])]
                response = chunks[0]
                for chunk in chunks[1:]:
                    response += chunk
                self.assertEqual({call["id"] for call in response.tool_calls}, {"call-0", "call-1"})
                followup = [HumanMessage(content="test"), response] + [
                    ToolMessage(content=str(index), tool_call_id=f"call-{index}") for index in range(2)]
                final = await model.ainvoke(followup)
                self.assertIn("OK", str(final.content))
            key = "input" if protocol == "responses" else "messages"
            history = sent[1][key]
            if protocol == "messages":
                blocks = next(message["content"] for message in history if message["role"] == "assistant")
                self.assertEqual(blocks[0]["signature"], "synthetic-signature")
                self.assertEqual(blocks[0]["thinking"], "synthetic thought")
                self.assertEqual(blocks[1]["data"], "synthetic-opaque")
            else:
                reasoning = next(item for item in history if item.get("type") == "reasoning")
                self.assertEqual(reasoning["encrypted_content"], "synthetic-encrypted")

    async def test_langchain_responses_replays_reasoning_finalized_at_item_done(self):
        sent = []
        values, _ = responses_tools(1)
        # Providers can finish the opaque payload at item.done, not item.added.
        del values[1]["item"]["encrypted_content"]
        def handler(request):
            sent.append(strict_json(request.content))
            if len(sent) == 1:
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      stream=test_app.ByteStream(test_app.sse("responses", values)))
            return httpx.Response(200, json=test_app.output("responses"))
        self.handler = handler
        model = ChatOpenAI(model="fixture-responses", api_key="synthetic", max_retries=0,
                           base_url="http://proxy.invalid/responses/v1", http_async_client=self.http,
                           use_responses_api=True, extra_body={"store": False})
        chunks = [chunk async for chunk in model.astream("test")]
        response = chunks[0]
        for chunk in chunks[1:]:
            response += chunk
        blocks = [block for block in response.content if isinstance(block, dict) and block.get("type") == "reasoning"]
        self.assertTrue(blocks)
        self.assertEqual(blocks[0].get("encrypted_content"), "synthetic-encrypted")

    async def test_streaming_tools_and_reasoning_replay(self):
        for protocol in ("messages", "responses"):
            for count in (1, 2):
                with self.subTest(protocol=protocol, count=count):
                    sent, streams = [], []
                    values, expected = messages_tools(count) if protocol == "messages" else responses_tools(count)
                    def handler(request):
                        sent.append(strict_json(request.content))
                        stream = test_app.ByteStream(test_app.sse(protocol, values if len(sent) == 1 else None), size=1)
                        streams.append(stream)
                        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
                    self.handler = handler
                    request = test_app.payload(protocol)
                    if protocol == "messages":
                        sdk = AsyncAnthropic(api_key="synthetic", base_url="http://proxy.invalid/messages", http_client=self.http, max_retries=0)
                        request["thinking"] = {"type": "adaptive"}
                        request["tools"] = [{"name": "lookup", "input_schema": {"type": "object", "properties": {"key": {"type": "integer"}}}}]
                        async with sdk.messages.stream(**request) as stream:
                            first = await stream.get_final_message()
                        blocks = [block.model_dump(exclude_none=True) for block in first.content]
                        self.assertEqual(blocks, expected)
                        request["messages"] += [{"role": "assistant", "content": blocks}, {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": f"call-{index}", "content": str(index)} for index in range(count)]}]
                        async with sdk.messages.stream(**request) as stream:
                            final = await stream.get_final_message()
                        self.assertEqual(final.content[0].text, "OK")
                    else:
                        sdk = AsyncOpenAI(api_key="synthetic", base_url="http://proxy.invalid/responses/v1", http_client=self.http, max_retries=0)
                        request["tools"] = [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"key": {"type": "integer"}}}}]
                        async with sdk.responses.stream(**request) as stream:
                            first = await stream.get_final_response()
                        blocks = [block.model_dump(exclude_none=True) for block in first.output]
                        self.assertEqual(blocks, expected)
                        request["input"] += blocks + [{"type": "function_call_output", "call_id": f"call-{index}", "output": str(index)} for index in range(count)]
                        # The low-level iterator also verifies native streamed termination.
                        final = [event async for event in await sdk.responses.create(**request, stream=True)]
                        self.assertEqual(final[-1].type, "response.completed")
                    key = "input" if protocol == "responses" else "messages"
                    self.assertEqual(len(sent), 2)
                    self.assertEqual(sent[1][key], request[key])
                    self.assertTrue(all(stream.closed for stream in streams))


class StreamLifecycleTests(unittest.TestCase):
    def validate(self, protocol, values):
        validator = Stream(protocol, profile(protocol, capabilities=["text", "tools", "parallel_tools", "reasoning", "streaming"]))
        for event in SSEParser().feed(test_app.sse(protocol, values)):
            validator.event(event)
        validator.finish()

    def test_messages_invalid_delta_signature_and_message_order(self):
        values, _ = messages_tools(2)
        cases = []
        wrong = copy.deepcopy(values)
        wrong[2]["delta"] = {"type": "input_json_delta", "partial_json": "{}"}
        cases.append(wrong)
        cases.append([event for event in values if event.get("delta", {}).get("type") != "signature_delta"])
        wrong = copy.deepcopy(values)
        wrong.insert(-1, {"type": "content_block_start", "index": 4, "content_block": {"type": "text", "text": ""}})
        cases.append(wrong)
        wrong = copy.deepcopy(values)
        wrong.insert(4, {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "late"}})
        cases.append(wrong)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ProtocolError):
                self.validate("messages", case)

    def test_responses_opaque_and_arguments_identity(self):
        values, _ = responses_tools(2)
        self.validate("responses", values)
        for initial_opaque in ("", "initial-placeholder"):
            progressed = copy.deepcopy(values)
            progressed[1]["item"]["encrypted_content"] = initial_opaque
            self.validate("responses", progressed)
        for mutation in ("opaque", "opaque_missing", "arguments", "identity"):
            broken = copy.deepcopy(values)
            done = next(event for event in broken if event["type"] == "response.output_item.done")
            if mutation == "opaque":
                done["item"]["encrypted_content"] = "changed"
            elif mutation == "opaque_missing":
                done["item"]["encrypted_content"] = ""
            elif mutation == "identity":
                done["item"]["id"] = "changed"
            else:
                delta = next(event for event in broken if event["type"] == "response.function_call_arguments.delta")
                delta["delta"] = "{}"
            with self.subTest(mutation=mutation), self.assertRaises(ProtocolError):
                self.validate("responses", broken)

    def test_responses_parts_lifecycle(self):
        values = test_app.events("responses")
        part = values[-1]["response"]["output"][0]["content"][0]
        base = {"output_index": 0, "content_index": 0, "item_id": "item-fixture"}
        values.insert(2, {**base, "type": "response.content_part.added", "part": {**part, "text": ""}})
        values.insert(5, {**base, "type": "response.content_part.done", "part": part})
        def renumber(events):
            for index, event in enumerate(events):
                event["sequence_number"] = index
            return events
        self.validate("responses", renumber(values))
        cases = [values[:5] + values[6:], values[:3] + [copy.deepcopy(values[2])] + values[3:]]
        wrong = copy.deepcopy(values)
        wrong[5]["part"]["text"] = "wrong"
        cases.append(wrong)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ProtocolError):
                self.validate("responses", renumber(copy.deepcopy(case)))