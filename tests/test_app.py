import asyncio
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from llm_proxy_standard.app import Settings, create_app
from llm_proxy_standard.cache import SnapshotCache
from llm_proxy_standard.contracts import strict_json
from tests.support import Authority, profile, UPSTREAM


def output(protocol, model="fixture-upstream"):
    if protocol == "chat":
        return {"id": "chat-fixture", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}
    if protocol == "messages":
        return {"id": "msg-fixture", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}}
    return {"id": "resp-fixture", "object": "response", "created_at": 0, "model": model,
            "status": "completed", "error": None, "incomplete_details": None,
            "output": [{"id": "item-fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}]}]}


def payload(protocol):
    data = {"model": "fixture-" + protocol}
    data["input" if protocol == "responses" else "messages"] = [{"role": "user", "content": "test"}]
    if protocol == "responses":
        data["store"] = False
    if protocol == "messages":
        data["max_tokens"] = 32
    return data


def events(protocol):
    body = output(protocol)
    if protocol == "chat":
        return [{"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture-upstream",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, "[DONE]"]
    if protocol == "messages":
        return [{"type": "message_start", "message": {**body, "content": [], "stop_reason": None}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"}]
    item = body["output"][0]
    return [{"type": "response.created", "sequence_number": 0, "response": {**body, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0, "item_id": item["id"], "content_index": 0, "delta": "OK"},
            {"type": "response.output_text.done", "sequence_number": 3, "output_index": 0, "item_id": item["id"], "content_index": 0, "text": "OK"},
            {"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": item},
            {"type": "response.completed", "sequence_number": 5, "response": body}]


def sse(protocol, values=None):
    raw = bytearray()
    for value in values if values is not None else events(protocol):
        if isinstance(value, str):
            raw.extend(("data: " + value + "\n\n").encode())
        else:
            prefix = "event: " + value["type"] + "\n" if "type" in value else ""
            raw.extend((prefix + "data: " + json.dumps(value) + "\n\n").encode())
    return bytes(raw)


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, raw, size=7):
        self.raw, self.size, self.closed = raw, size, False

    async def __aiter__(self):
        for start in range(0, len(self.raw), self.size):
            yield self.raw[start:start + self.size]

    async def aclose(self):
        self.closed = True


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.token = Path(self.directory.name) / "token"
        self.token.write_text("synthetic-token-first")
        self.authority, self.requests = Authority(), []
        profiles = [profile(protocol, capabilities=["text", "tools", "parallel_tools", "streaming", "structured_output", "reasoning"],
                    options={"store": [False]} if protocol == "responses" else
                    {"max_tokens": [32], "thinking.type": ["adaptive"]} if protocol == "messages" else {})
                    for protocol in ("chat", "responses", "messages")]
        self.document = self.authority.snapshot(*profiles).document()
        async def fetch():
            return self.document
        self.cache = SnapshotCache(fetch, self.authority.trust)
        def handler(request):
            self.requests.append(request)
            protocol = "chat" if request.url.path.endswith("completions") else request.url.path.rsplit("/", 1)[1]
            if strict_json(request.content).get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ByteStream(sse(protocol)))
            return httpx.Response(200, json=output(protocol))
        self.handler = handler
        self.app = create_app(Settings(UPSTREAM, self.token), self.cache, transport=httpx.MockTransport(lambda request: self.handler(request)))
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://proxy.invalid")

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.life.__aexit__(None, None, None)
        self.directory.cleanup()

    async def test_native_sdk_three_protocols_discovery_and_inference(self):
        chat = AsyncOpenAI(api_key="client-synthetic", base_url="http://proxy.invalid/chat/v1", http_client=self.http, max_retries=0)
        self.assertEqual((await chat.models.list()).data[0].id, "fixture-chat")
        response = await chat.chat.completions.create(**payload("chat"))
        self.assertEqual(response.model, "fixture-chat")
        self.assertEqual(response.choices[0].message.content, "OK")
        responses = AsyncOpenAI(api_key="client-synthetic", base_url="http://proxy.invalid/responses/v1", http_client=self.http, max_retries=0)
        response = await responses.responses.create(**payload("responses"))
        self.assertEqual(response.output_text, "OK")
        self.assertEqual(response.model, "fixture-responses")
        messages = AsyncAnthropic(api_key="client-synthetic", base_url="http://proxy.invalid/messages", http_client=self.http, max_retries=0)
        self.assertEqual((await messages.models.list()).data[0].id, "fixture-messages")
        response = await messages.messages.create(**payload("messages"))
        self.assertEqual(response.content[0].text, "OK")
        self.assertEqual(response.model, "fixture-messages")

    async def test_native_sdk_three_protocols_streaming(self):
        for protocol in ("chat", "responses", "messages"):
            with self.subTest(protocol=protocol):
                if protocol == "messages":
                    client = AsyncAnthropic(api_key="synthetic", base_url="http://proxy.invalid/messages", http_client=self.http, max_retries=0)
                    stream = await client.messages.create(**payload(protocol), stream=True)
                else:
                    client = AsyncOpenAI(api_key="synthetic", base_url=f"http://proxy.invalid/{protocol}/v1", http_client=self.http, max_retries=0)
                    stream = await (client.chat.completions if protocol == "chat" else client.responses).create(**payload(protocol), stream=True)
                received = [item async for item in stream]
                self.assertTrue(received)
                if protocol == "messages":
                    self.assertEqual(received[-1].type, "message_stop")
                if protocol == "responses":
                    self.assertEqual(received[-1].type, "response.completed")

    async def test_credential_isolation_rotation_and_exact_destination(self):
        for token in ("synthetic-token-first", "synthetic-token-second"):
            self.token.write_text(token)
            result = await self.http.post("/chat/v1/chat/completions", json=payload("chat"),
                headers={"Authorization": "Bearer client-secret", "x-api-key": "client-key", "x-user": "admin"})
            self.assertEqual(result.status_code, 200)
            sent = self.requests[-1]
            self.assertEqual(sent.headers["authorization"], "Bearer " + token)
            self.assertNotIn("x-api-key", sent.headers)
            self.assertNotIn("x-user", sent.headers)
            self.assertEqual(str(sent.url), UPSTREAM + "/chat/completions")
            self.assertEqual(strict_json(sent.content)["model"], "fixture-upstream")

    async def test_history_and_tool_error_preserved_all_protocols(self):
        for protocol in ("chat", "responses", "messages"):
            body = payload(protocol)
            key = "input" if protocol == "responses" else "messages"
            if protocol == "chat":
                body[key] += [{"role": "assistant", "tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                              {"role": "tool", "tool_call_id": "call-a", "content": "UNAVAILABLE"}]
            elif protocol == "messages":
                body[key] += [{"role": "assistant", "content": [{"type": "thinking", "thinking": "opaque", "signature": "signed"},
                    {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-a", "is_error": True, "content": "UNAVAILABLE"}]}]
            else:
                body[key] += [{"type": "reasoning", "id": "rs-a", "summary": [], "encrypted_content": "opaque"},
                    {"type": "function_call", "id": "fc-a", "call_id": "call-a", "name": "lookup", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call-a", "output": "UNAVAILABLE"}]
            suffix = "chat/completions" if protocol == "chat" else protocol
            result = await self.http.post(f"/{protocol}/v1/{suffix}", json=body)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(strict_json(self.requests[-1].content)[key], body[key])

    async def test_no_bypass_routes_unknown_options_and_revocation(self):
        self.assertEqual((await self.http.post("/v1/chat/completions", json=payload("chat"))).status_code, 404)
        self.assertEqual((await self.http.post("/admin/promote", json={})).status_code, 404)
        self.assertEqual((await self.http.post("/chat/v1/chat/completions", json={**payload("chat"), "upstream_url": "https://evil.invalid"})).status_code, 400)
        self.document["generation"] += 1
        self.document["revoked"] = ["fixture-chat"]
        await self.cache.refresh()
        self.assertEqual((await self.http.get("/chat/v1/models")).json()["data"], [])
        self.assertEqual((await self.http.post("/chat/v1/chat/completions", json=payload("chat"))).status_code, 404)
        self.assertEqual(self.requests, [])

    async def test_upstream_failure_no_retry_and_malformed_response(self):
        self.handler = lambda request: (self.requests.append(request) or httpx.Response(429, json={"private": "secret"}))
        response = await self.http.post("/chat/v1/chat/completions", json=payload("chat"))
        self.assertEqual(response.status_code, 429)
        self.assertNotIn("secret", response.text)
        self.assertEqual(len(self.requests), 1)
        self.handler = lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "partial"}}]})
        self.assertEqual((await self.http.post("/chat/v1/chat/completions", json=payload("chat"))).status_code, 502)

    async def test_native_incomplete_and_refusal_not_changed_to_success(self):
        body = output("responses")
        body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        self.handler = lambda _: httpx.Response(200, json=body)
        result = await self.http.post("/responses/v1/responses", json=payload("responses"))
        self.assertEqual(result.json()["status"], "incomplete")
        self.assertEqual(result.json()["incomplete_details"], body["incomplete_details"])

    async def test_messages_pagination_and_model_detail(self):
        self.assertEqual((await self.http.get("/messages/v1/models?limit=0")).status_code, 400)
        result = await self.http.get("/messages/v1/models?limit=1")
        self.assertFalse(result.json()["has_more"])
        self.assertEqual((await self.http.get("/messages/v1/models?after_id=fixture-messages")).json()["data"], [])
        self.assertEqual((await self.http.get("/messages/v1/models/fixture-messages")).json()["type"], "model")

    async def test_declared_override_observable_and_protected_input_unchanged(self):
        rule = {"id": "default-limit", "path": "max_tokens", "operation": "set", "when": "missing", "value": 32,
                "reason": "synthetic fixture", "effects": "limits output"}
        value = profile("messages", rules=[rule], options={"max_tokens": [32]})
        doc = self.authority.snapshot(value).document()
        async def fetch():
            return doc
        cache = SnapshotCache(fetch, self.authority.trust)
        app = create_app(Settings(UPSTREAM, self.token), cache, transport=httpx.MockTransport(self.handler))
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.invalid") as client:
            body = payload("messages")
            del body["max_tokens"]
            result = await client.post("/messages/v1/messages", json=body)
            self.assertEqual(result.status_code, 200)
            change = strict_json(base64.urlsafe_b64decode(result.headers["x-proxy-adjustments"]))[0]
            self.assertTrue(change["was_missing"])
            self.assertEqual(change["effective"], 32)
            self.assertEqual(strict_json(self.requests[-1].content)["messages"], body["messages"])