"""Synthetic protocol policy checks; not live model qualification."""
import copy
import unittest

from llm_proxy_standard.contracts import Profile, ProfileV2, Proposal, canonical, parse_profile, required_coverage
from llm_proxy_standard.rules import RequestPolicyError, prepare
from tests.support import Authority, profile
from tests import test_app


def flexible(protocol="messages", **kwargs):
    data = profile(protocol, **kwargs).document()
    data.update(schema_version=2, constraints={"max_tokens" if protocol != "responses" else "max_output_tokens":
                {"kind": "integer", "minimum": 1, "maximum": 128000}})
    return ProfileV2.model_validate(data)


class OptionTests(unittest.TestCase):
    def test_v1_serialization_and_signed_proposal_preserved(self):
        value = profile()
        raw = canonical(value.document())
        self.assertEqual(raw, canonical(parse_profile(value.document()).document()))
        entry = Authority().entry(value)
        restored = Proposal.model_validate(entry.proposal.document())
        self.assertEqual(restored.document(), entry.proposal.document())
        self.assertIs(type(restored.profile), Profile)

    def test_integer_domain_preserves_explicit_budget(self):
        value = flexible()
        body = {"model": value.id, "messages": [{"role": "user", "content": "test"}]}
        for cap in (1, 2048, 32768, 128000):
            requested = {**body, "max_tokens": cap}
            before = copy.deepcopy(requested)
            effective, changes = prepare(requested, value)
            self.assertEqual(effective["max_tokens"], cap)
            self.assertEqual(requested, before)
            self.assertEqual(changes, [])
        for cap in (0, 128001, True, 2048.0, "2048", None):
            with self.subTest(cap=cap), self.assertRaises(RequestPolicyError):
                prepare({**body, "max_tokens": cap}, value)

    def test_responses_hints_are_bounded_pass_through_not_dropped(self):
        data = flexible("responses", capabilities=["text", "reasoning"], options={"store": [False]}).document()
        data["constraints"].update({"prompt_cache_key": {"kind": "string", "max_length": 128},
                                    "reasoning.summary": {"kind": "enum", "values": ["auto"]},
                                    "reasoning.effort": {"kind": "enum", "values": ["medium"]}})
        value = ProfileV2.model_validate(data)
        body = {"model": value.id, "input": "test", "store": False, "prompt_cache_key": "fixture-key",
                "reasoning": {"effort": "medium", "summary": "auto"}}
        effective, changes = prepare(body, value)
        self.assertEqual(effective, {**body, "model": value.model})
        self.assertEqual(changes, [])
        self.assertIn("constraint:reasoning.summary", required_coverage(value))
        for key in ("", "x" * 129, "line\nbreak", None, 42):
            with self.assertRaises(RequestPolicyError):
                prepare({**body, "prompt_cache_key": key}, value)

    def test_dependencies_conflicts_and_missing_default(self):
        data = flexible("chat", options={"max_completion_tokens": [2048], "reasoning_effort": ["none"]}).document()
        data["requires"] = {"max_tokens": ["reasoning_effort"]}
        data["conflicts"] = [["max_tokens", "max_completion_tokens"]]
        value = ProfileV2.model_validate(data)
        body = {"model": value.id, "messages": [{"role": "user", "content": "test"}], "max_tokens": 100}
        with self.assertRaises(RequestPolicyError):
            prepare(body, value)
        prepare({**body, "reasoning_effort": "none"}, value)
        with self.assertRaises(RequestPolicyError):
            prepare({**body, "reasoning_effort": "none", "max_completion_tokens": 2048}, value)
        data["rules"] = [dict(id="default-cap", path="max_tokens", operation="set", when="missing",
                              value=2048, reason="Fixture default", effects="Bounded output")]
        value = ProfileV2.model_validate(data)
        effective, changes = prepare({"model": value.id, "messages": body["messages"], "reasoning_effort": "none"}, value)
        self.assertEqual(effective["max_tokens"], 2048)
        self.assertTrue(changes[0]["was_missing"])

    def test_constraints_cannot_authorize_protected_fields(self):
        for path in ("messages", "model", "input", "tools", "previous_response_id", "metadata"):
            data = flexible().document()
            data["constraints"][path] = {"kind": "string", "max_length": 10}
            with self.assertRaises(ValueError):
                ProfileV2.model_validate(data)
        data = flexible().document()
        data["constraints"]["max_tokens"] = {"kind": "integer", "minimum": 0, "maximum": 2048}
        with self.assertRaises(ValueError):
            ProfileV2.model_validate(data)

    def test_large_payload_defaults_and_adaptive_thinking_normalization(self):
        from pathlib import Path
        from llm_proxy_standard.app import Settings
        s = Settings("https://acct.snowflakecomputing.com/api/v2/cortex/v1", Path("/tmp/token"))
        self.assertEqual(s.max_body, 32 * 1024 * 1024)
        self.assertEqual(s.max_response, 32 * 1024 * 1024)
        self.assertEqual(s.deadline, 600)
        msg_data = flexible("messages", capabilities=["text", "reasoning"]).document()
        msg_data["normalize_client_options"] = True
        msg_data["constraints"].update({
            "thinking.type": {"kind": "enum", "values": ["enabled", "disabled"]},
            "thinking.budget_tokens": {"kind": "integer", "minimum": 1024, "maximum": 64000},
            "output_config.effort": {"kind": "enum", "values": ["low", "medium", "high"]},
        })
        msg_prof = ProfileV2.model_validate(msg_data)
        eff, changes = prepare({
            "model": msg_prof.id,
            "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}],
            "max_tokens": 2048,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "medium"},
        }, msg_prof)
        self.assertNotIn("thinking", eff)
        self.assertNotIn("output_config", eff)
        self.assertTrue(any(c["id"] == "strip-unsupported-thinking-type" for c in changes))
        resp_data = flexible("responses", capabilities=["text", "reasoning"]).document()
        resp_data["normalize_client_options"] = True
        resp_data["constraints"].update({
            "store": {"kind": "enum", "values": [False]},
            "include": {"kind": "enum", "values": [["reasoning.encrypted_content"], []]},
            "reasoning.effort": {"kind": "enum", "values": ["low", "medium", "high"]},
            "reasoning.summary": {"kind": "enum", "values": ["auto", "concise", "detailed"]},
        })
        resp_prof = ProfileV2.model_validate(resp_data)
        eff_r, changes_r = prepare({
            "model": resp_prof.id,
            "input": [{"role": "user", "content": "summarize"}],
            "reasoning": {"effort": "none", "summary": "invalid"},
            "prompt_cache_retention": "in_memory",
        }, resp_prof)
        self.assertEqual(eff_r["reasoning"], {"effort": "low", "summary": "auto"})
        self.assertNotIn("prompt_cache_retention", eff_r)
        self.assertTrue(any(c["id"] == "clamp-reasoning-effort" for c in changes_r))


class HTTPOptionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown

    async def test_hermes_builder_options_preserved_without_inference_retry(self):
        # Reconstructed synthetic bodies from actual Hermes builders. SDK-only
        # timeout is deliberately not an HTTP body field. No real profiles here.
        for protocol in ("chat", "responses", "messages"):
            data = flexible(protocol, capabilities=["text", "reasoning"]).document()
            body = {"model": data["id"]}
            if protocol == "chat":
                data["constraints"]["reasoning_effort"] = {"kind": "enum", "values": ["medium"]}
                body.update(messages=[{"role": "user", "content": "Synthetic fixture"}], reasoning_effort="medium")
            elif protocol == "responses":
                data["options"] = {"store": [False], "include": [["reasoning.encrypted_content"]]}
                data["constraints"].update({"prompt_cache_key": {"kind": "string", "max_length": 128},
                    "reasoning.effort": {"kind": "enum", "values": ["medium"]},
                    "reasoning.summary": {"kind": "enum", "values": ["auto"]}})
                body.update(input=[{"role": "user", "content": "Synthetic fixture"}], store=False,
                    include=["reasoning.encrypted_content"], prompt_cache_key="synthetic-key",
                    reasoning={"effort": "medium", "summary": "auto"})
            else:
                body.update(messages=[{"role": "user", "content": "Synthetic fixture"}], max_tokens=128000)
            self.document = self.authority.snapshot(ProfileV2.model_validate(data)).document()
            self.cache.snapshot = None
            self.assertTrue(await self.cache.refresh())
            suffix = "chat/completions" if protocol == "chat" else protocol
            response = await self.http.post(f"/{protocol}/v1/{suffix}", json=body)
            self.assertEqual(response.status_code, 200)
            from llm_proxy_standard.contracts import strict_json
            self.assertEqual(strict_json(self.requests[-1].content), {**body, "model": "fixture-upstream"})
        self.assertEqual(len(self.requests), 3)

    async def test_actual_n8n_serialized_requests(self):
        # Exact synthetic bodies observed from n8n2.37.9/node package2.37.4.
        cases = [("chat", {"model": "fixture-chat", "stream": False, "max_tokens": 2048,
                   "messages": [{"role": "user", "content": "Synthetic fixture"}]}, {"max_tokens": [2048]}),
                 ("responses", {"input": [{"type": "message", "role": "user", "content": "Synthetic fixture"}],
                   "model": "fixture-responses", "stream": False, "include": ["reasoning.encrypted_content"],
                   "text": {}, "max_output_tokens": 2048, "store": False},
                   {"max_output_tokens": [2048], "store": [False], "include": [["reasoning.encrypted_content"]]}),
                 ("messages", {"model": "fixture-messages", "stream": False, "max_tokens": 2048,
                   "messages": [{"role": "user", "content": "Synthetic fixture"}]}, {"max_tokens": [2048]})]
        for protocol, body, options in cases:
            self.document = self.authority.snapshot(profile(protocol, options=options)).document()
            self.cache.snapshot = None
            self.assertTrue(await self.cache.refresh())
            suffix = "chat/completions" if protocol == "chat" else protocol
            response = await self.http.post(f"/{protocol}/v1/{suffix}", json=body)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.requests), 3)

    async def test_v2_native_sdk_errors_are_safe_and_precede_inference(self):
        value = flexible()
        self.document = self.authority.snapshot(value).document()
        self.document["generation"] = 2
        self.cache.snapshot = None  # isolated fresh catalog, not a live update bypass
        self.assertTrue(await self.cache.refresh())
        body = {**test_app.payload("messages"), "max_tokens": 128000}
        result = await self.http.post("/messages/v1/messages", json=body)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(self.requests), 1)
        for addition in ({"max_tokens": 128001}, {"secret-field-value": "do-not-reflect"}):
            result = await self.http.post("/messages/v1/messages", json={**body, **addition})
            self.assertEqual(result.status_code, 400)
            self.assertIn("code", result.json()["error"])
            self.assertNotIn("do-not-reflect", result.text)
            self.assertNotIn("secret-field-value", result.text)
            self.assertEqual(len(self.requests), 1)