"""Real LangGraph ToolNode loops through Standard with synthetic upstream only."""
import os
import unittest
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from llm_proxy_standard.contracts import strict_json, ProfileV2
from tests import test_app, test_stream_replay
from tests.support import profile


def tool_reply(protocol):
    document = test_app.output(protocol)
    if protocol == "responses":
        _, document["output"] = test_stream_replay.responses_tools(2)
    elif protocol == "messages":
        _, document["content"] = test_stream_replay.messages_tools(2)
        document["stop_reason"] = "tool_use"
    else:
        document["choices"][0].update(finish_reason="tool_calls", message={"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {
                "name": "lookup", "arguments": '{"key":' + str(index) + '}'}} for index in range(2)]})
    return document


def tool_events(protocol):
    if protocol == "responses":
        events, _ = test_stream_replay.responses_tools(2)
        del events[1]["item"]["encrypted_content"]
        return events
    if protocol == "messages":
        return test_stream_replay.messages_tools(2)[0]
    events = []
    for index in range(2):
        events.append({"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture-upstream", "choices": [{"index": 0, "finish_reason": None,
            "delta": {"role": "assistant", "tool_calls": [{"index": index, "id": f"call-{index}",
                "type": "function", "function": {"name": "lookup", "arguments": '{"key":'}}]}}]})
    for index in range(2):
        events.append({"choices": [{"index": 0, "finish_reason": None, "delta": {
            "tool_calls": [{"index": index, "function": {"arguments": str(index) + '}'}}]}}]})
    return events + [{"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, "[DONE]"]


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    streaming_modes = (False, True)
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown

    async def model_for(self, protocol):
        if protocol == "messages":
            return ChatAnthropic(model="fixture-messages", api_key="synthetic", max_tokens=32,
                                 base_url="http://proxy.invalid/messages", max_retries=0)
        return ChatOpenAI(model="fixture-" + protocol, api_key="synthetic", max_retries=0,
                          base_url=f"http://proxy.invalid/{protocol}/v1", http_async_client=self.http,
                          use_responses_api=protocol == "responses",
                          extra_body={"store": False} if protocol == "responses" else {})

    async def install_profile(self, protocol):
        options = {"store": [False]} if protocol == "responses" else {"max_tokens": [32]} if protocol == "messages" else {}
        value = profile(protocol, capabilities=["text", "tools", "parallel_tools", "reasoning", "streaming"], options=options)
        if protocol == "responses":
            value = ProfileV2.model_validate({**value.document(), "schema_version": 2, "response_stream": "ordered-items"})
        self.document = self.authority.snapshot(value).document()
        self.cache.snapshot = None
        self.assertTrue(await self.cache.refresh())

    async def test_parallel_tools_error_history_and_checkpoint(self):
        for protocol in ("chat", "responses", "messages"):
            for streaming in self.streaming_modes:
                for tool_error in (False, True):
                    with self.subTest(protocol=protocol, streaming=streaming, tool_error=tool_error):
                        await self.install_profile(protocol)
                        sent, effects = [], []
                        @tool
                        def lookup(key: int) -> str:
                            """Return a synthetic value, with no external effects."""
                            effects.append(key)
                            if tool_error and key == 1:
                                raise ValueError("SYNTHETIC_UNAVAILABLE")
                            return str(key + 17)

                        def handler(request):
                            body = strict_json(request.content)
                            sent.append(body)
                            if body.get("stream"):
                                values = tool_events(protocol) if len(sent) == 1 else test_app.events(protocol)
                                for event in values:
                                    if isinstance(event, dict):
                                        if "response" in event:
                                            event["response"]["id"] = f"resp-fixture-{len(sent)}"
                                        if "message" in event:
                                            event["message"]["id"] = f"msg-fixture-{len(sent)}"
                                        if "id" in event:
                                            event["id"] = f"chat-fixture-{len(sent)}"
                                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                                      stream=test_app.ByteStream(test_app.sse(protocol, values)))
                            document = tool_reply(protocol) if len(sent) == 1 else test_app.output(protocol)
                            # MessagesState replaces messages with repeated IDs;
                            # independent synthetic responses need distinct IDs.
                            document["id"] = f"fixture-{len(sent)}"
                            return httpx.Response(200, json=document)
                        self.handler = handler
                        model = (await self.model_for(protocol)).bind_tools([lookup])
                        async def infer(state):
                            if streaming:
                                chunks = [chunk async for chunk in model.astream(state["messages"])]
                                answer = chunks[0]
                                for chunk in chunks[1:]:
                                    answer += chunk
                            else:
                                answer = await model.ainvoke(state["messages"])
                            return {"messages": [answer]}
                        graph = StateGraph(MessagesState)
                        graph.add_node("infer", infer)
                        graph.add_node("tools", ToolNode([lookup], handle_tool_errors=True))
                        graph.add_edge(START, "infer")
                        graph.add_conditional_edges("infer", lambda state: "tools" if state["messages"][-1].tool_calls else END)
                        graph.add_edge("tools", "infer")
                        saver = InMemorySaver()
                        compiled = graph.compile(checkpointer=saver)
                        config = {"configurable": {"thread_id": "fixture"}, "recursion_limit": 5}
                        with patch("langchain_anthropic.chat_models._get_default_async_httpx_client", return_value=self.http):
                            result = await compiled.ainvoke({"messages": [HumanMessage(content="Synthetic fixture BLUE7")]}, config)
                            self.assertEqual(sorted(effects), [0, 1])
                            self.assertEqual(len(sent), 2)
                            tool_results = [item for item in result["messages"] if isinstance(item, ToolMessage)]
                            self.assertEqual({item.tool_call_id for item in tool_results}, {"call-0", "call-1"})
                            self.assertEqual(sorted(item.status for item in tool_results), ["error", "success"] if tool_error else ["success", "success"])
                            wire = sent[1]["input" if protocol == "responses" else "messages"]
                            if protocol == "responses":
                                self.assertEqual(next(item for item in wire if item.get("type") == "reasoning")["encrypted_content"], "synthetic-encrypted")
                                self.assertEqual({item["call_id"] for item in wire if item.get("type") == "function_call_output"}, {"call-0", "call-1"})
                            elif protocol == "messages":
                                blocks = next(item["content"] for item in wire if item["role"] == "assistant")
                                self.assertEqual(blocks[0]["signature"], "synthetic-signature")
                                self.assertEqual(blocks[1]["data"], "synthetic-opaque")
                            self.assertIn("OK", str(result["messages"][-1].content))
                            restored = graph.compile(checkpointer=saver)
                            await restored.ainvoke({"messages": [HumanMessage(content="Follow up BLUE7")]}, config)
                            self.assertEqual(len(sent), 3)
                            self.assertEqual(sorted(effects), [0, 1])
                            self.assertIn("BLUE7", str(sent[2]))
                            isolated = await restored.aget_state({"configurable": {"thread_id": "other"}})
                            self.assertFalse(isolated.values)

    async def test_error_attempts_separate_client_and_proxy(self):
        for protocol in ("chat", "responses", "messages"):
            await self.install_profile(protocol)
            model = await self.model_for(protocol)
            for status in (429, 500, 503, "timeout"):
                upstream, incoming = [], []
                def handler(request):
                    upstream.append(request)
                    if status == "timeout":
                        raise httpx.ReadTimeout("synthetic", request=request)
                    return httpx.Response(status, json={"error": "private-fixture"})
                async def observe(request):
                    incoming.append(request)
                self.handler = handler
                self.http.event_hooks["request"] = [observe]
                with self.subTest(protocol=protocol, status=status), patch(
                        "langchain_anthropic.chat_models._get_default_async_httpx_client", return_value=self.http):
                    with self.assertRaises(Exception):
                        await model.ainvoke("Synthetic failure")
                    self.assertEqual(len(incoming), 1)
                    self.assertEqual(len(upstream), 1)
                self.http.event_hooks["request"] = []