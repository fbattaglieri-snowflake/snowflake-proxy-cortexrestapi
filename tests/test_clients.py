"""Standard LangChain clients and LangGraph node, using the real proxy app offline."""
import os
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import unittest
from unittest.mock import patch
from tests import test_app


class ClientsTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown

    async def test_langchain_chat_and_responses(self):
        for protocol in ("chat", "responses"):
            model = ChatOpenAI(model="fixture-" + protocol, api_key="synthetic",
                base_url=f"http://proxy.invalid/{protocol}/v1", http_async_client=self.http,
                max_retries=0, use_responses_api=protocol == "responses",
                extra_body={"store": False} if protocol == "responses" else {})
            response = await model.ainvoke("test")
            self.assertIn("OK", str(response.content))

    async def test_langchain_messages_and_langgraph(self):
        model = ChatAnthropic(model="fixture-messages", api_key="synthetic", max_tokens=32,
            max_retries=0, base_url="http://proxy.invalid/messages")

        class State(TypedDict):
            question: str
            answer: str

        async def answer(state):
            result = await model.ainvoke(state["question"])
            return {"answer": result.content}
        graph = StateGraph(State)
        graph.add_node("answer_node", answer)
        graph.add_edge(START, "answer_node")
        graph.add_edge("answer_node", END)
        # Transport injection for offline testing, not a custom production client.
        with patch("langchain_anthropic.chat_models._get_default_async_httpx_client", return_value=self.http):
            result = await graph.compile().ainvoke({"question": "test"})
        self.assertEqual(result["answer"], "OK")