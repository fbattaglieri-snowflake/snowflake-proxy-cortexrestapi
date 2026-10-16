"""Exercise the shipped client factory, not a separately configured test client."""
import copy
import unittest

from examples.standard_clients import langchain_model
from llm_proxy_standard.contracts import ProfileV2
from tests import test_agent_loop, test_multiround_replay
from tests.support import profile


class ExampleConfigurationTests(unittest.TestCase):
    def test_responses_defaults_preserve_input(self):
        options = {"max_tokens": 2048}
        before = copy.deepcopy(options)
        model = langchain_model("http://proxy.invalid", "fixture-responses", "responses", options=options)
        self.assertEqual(options, before)
        self.assertIs(model.store, False)
        self.assertEqual(model.include, ["reasoning.encrypted_content"])
        self.assertEqual(model.output_version, "responses/v1")
        self.assertEqual(model.max_retries, 0)

    def test_reject_lossy_or_server_stored_continuation(self):
        for options in (
            {"store": True}, {"store": 0}, {"include": []},
            {"include": "reasoning.encrypted_content"},
            {"output_version": "v0"}, {"previous_response_id": "resp-old"},
            {"use_previous_response_id": True},
            {"extra_body": {"store": True}},
            {"extra_body": {"include": []}},
            {"extra_body": {"previous_response_id": "resp-old"}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                langchain_model("http://proxy.invalid", "fixture-responses", "responses", options=options)


class ExampleMultiRoundTests(test_multiround_replay.MultiRoundTests):
    async def install_profile(self, protocol):
        if protocol != "responses":
            return await test_agent_loop.AgentLoopTests.install_profile(self, protocol)
        value = profile(protocol, capabilities=["text", "tools", "parallel_tools", "reasoning", "streaming"],
                        options={"store": [False], "include": [["reasoning.encrypted_content"]]})
        value = ProfileV2.model_validate({**value.document(), "schema_version": 2,
                                         "response_stream": "ordered-items"})
        self.document = self.authority.snapshot(value).document()
        self.cache.snapshot = None
        self.assertTrue(await self.cache.refresh())

    async def model_for(self, protocol):
        options = {"max_tokens": 32} if protocol == "messages" else {"http_async_client": self.http}
        return langchain_model("http://proxy.invalid", "fixture-" + protocol, protocol, options=options)