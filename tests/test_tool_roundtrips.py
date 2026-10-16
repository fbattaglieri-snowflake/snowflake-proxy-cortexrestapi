import copy
import unittest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import httpx
from llm_proxy_standard.contracts import strict_json
from tests import test_app


class ToolRoundtripTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown

    async def test_native_sdk_single_and_parallel_roundtrips_all_protocols(self):
        for protocol in ("chat", "responses", "messages"):
            for count in (1, 2):
                with self.subTest(protocol=protocol, calls=count):
                    sent = []
                    calls = [{"id": f"call-{index}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}} for index in range(count)]
                    def handler(request):
                        sent.append(strict_json(request.content))
                        body = test_app.output(protocol)
                        if len(sent) == 1:
                            if protocol == "chat":
                                body["choices"][0].update(finish_reason="tool_calls", message={"role": "assistant", "content": None, "tool_calls": calls})
                            elif protocol == "messages":
                                body["content"] = [{"type": "tool_use", "id": call["id"], "name": "lookup", "input": {}} for call in calls]
                                body["stop_reason"] = "tool_use"
                            else:
                                body["output"] = [{"id": "fc-" + call["id"], "type": "function_call", "call_id": call["id"], "name": "lookup", "arguments": "{}", "status": "completed"} for call in calls]
                        return httpx.Response(200, json=body)
                    self.handler = handler
                    request = test_app.payload(protocol)
                    if protocol == "chat":
                        request["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
                    elif protocol == "messages":
                        request["tools"] = [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}]
                    else:
                        request["tools"] = [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]
                    if protocol == "messages":
                        sdk = AsyncAnthropic(api_key="synthetic", base_url="http://proxy.invalid/messages", http_client=self.http, max_retries=0)
                        first = await sdk.messages.create(**request)
                        self.assertEqual(len(first.content), count)
                        request["messages"] += [{"role": "assistant", "content": [item.model_dump(exclude_none=True) for item in first.content]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": str(index), "is_error": False} for index, call in enumerate(calls)]}]
                        second = await sdk.messages.create(**request)
                        self.assertEqual(second.content[0].text, "OK")
                    else:
                        sdk = AsyncOpenAI(api_key="synthetic", base_url=f"http://proxy.invalid/{protocol}/v1", http_client=self.http, max_retries=0)
                        api = sdk.chat.completions if protocol == "chat" else sdk.responses
                        first = await api.create(**request)
                        if protocol == "chat":
                            self.assertEqual(len(first.choices[0].message.tool_calls), count)
                            request["messages"] += [first.choices[0].message.model_dump(exclude_none=True)] + [
                                {"role": "tool", "tool_call_id": call["id"], "content": str(index)} for index, call in enumerate(calls)]
                        else:
                            self.assertEqual(len(first.output), count)
                            request["input"] += [item.model_dump(exclude_none=True) for item in first.output] + [
                                {"type": "function_call_output", "call_id": call["id"], "output": str(index)} for index, call in enumerate(calls)]
                        second = await api.create(**request)
                        self.assertIn("OK", str(second))
                    self.assertEqual(len(sent), 2)
                    key = "input" if protocol == "responses" else "messages"
                    self.assertEqual(sent[1][key], request[key])

    async def test_structured_output_passthrough_all_protocols(self):
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
        for protocol in ("chat", "responses", "messages"):
            request = test_app.payload(protocol)
            if protocol == "chat":
                request["response_format"] = {"type": "json_schema", "json_schema": {"name": "fixture", "strict": True, "schema": schema}}
            elif protocol == "responses":
                request["text"] = {"format": {"type": "json_schema", "name": "fixture", "strict": True, "schema": schema}}
            else:
                request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            suffix = "chat/completions" if protocol == "chat" else protocol
            response = await self.http.post(f"/{protocol}/v1/{suffix}", json=request)
            self.assertEqual(response.status_code, 200)
            expected = copy.deepcopy(request)
            expected["model"] = "fixture-upstream"
            self.assertEqual(strict_json(self.requests[-1].content), expected)