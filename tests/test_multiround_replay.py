"""Two dependent parallel tool rounds through real LangGraph and native SDKs."""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from llm_proxy_standard.contracts import strict_json
from tests import test_agent_loop, test_app


def round_value(value, turn):
    if isinstance(value, list):
        return [round_value(item, turn) for item in value]
    if isinstance(value, dict):
        return {key: item + 17 if key == 'key' and type(item) is int and turn == 2
                else round_value(item, turn) for key, item in value.items()}
    if isinstance(value, str):
        if value.startswith('{"key":') and turn == 2:
            # Only complete argument strings change; streamed fragments use helper below.
            try:
                arguments = json.loads(value)
            except json.JSONDecodeError:
                return value
            arguments['key'] += 17
            return json.dumps(arguments)
        for prefix in ('call-', 'fc-', 'reasoning-', 'resp-fixture', 'msg-fixture', 'chat-fixture'):
            if value.startswith(prefix):
                return f'round-{turn}-' + value
    return value


class MultiRoundTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_app.AppTests.asyncSetUp
    asyncTearDown = test_app.AppTests.asyncTearDown
    install_profile = test_agent_loop.AgentLoopTests.install_profile
    model_for = test_agent_loop.AgentLoopTests.model_for

    async def test_two_dependent_parallel_rounds_reverse_finish_and_opaque_history(self):
        for protocol in ('chat', 'responses', 'messages'):
            for streaming in (False, True):
                with self.subTest(protocol=protocol, streaming=streaming):
                    await self.install_profile(protocol)
                    sent, effects = [], []
                    first_started = asyncio.Event()
                    second_finished = asyncio.Event()

                    @tool
                    async def lookup(key: int) -> str:
                        """Return a synthetic value; the next round consumes prior outputs."""
                        if key == 0:
                            first_started.set()
                            await asyncio.wait_for(second_finished.wait(), 2)
                        elif key == 1:
                            await asyncio.wait_for(first_started.wait(), 2)
                            effects.append(key)
                            second_finished.set()
                            return '18'
                        effects.append(key)
                        return str(key + 17)

                    def upstream(request):
                        body = strict_json(request.content)
                        sent.append(body)
                        turn = len(sent)
                        history = body['input' if protocol == 'responses' else 'messages']
                        if turn > 1:
                            if protocol == 'responses':
                                results = {item['call_id']: item['output'] for item in history
                                           if item.get('type') == 'function_call_output'}
                            elif protocol == 'chat':
                                results = {item['tool_call_id']: item['content'] for item in history
                                           if item.get('role') == 'tool'}
                            else:
                                results = {block['tool_use_id']: block['content'] for item in history
                                           if isinstance(item.get('content'), list) for block in item['content']
                                           if block.get('type') == 'tool_result'}
                            expected = {'round-1-call-0': '17', 'round-1-call-1': '18'}
                            if turn == 3:
                                expected.update({'round-2-call-0': '34', 'round-2-call-1': '35'})
                            self.assertEqual(results, expected)
                        if turn > 3:
                            raise AssertionError('Unexpected retry or agent loop')
                        if streaming:
                            values = (test_agent_loop.tool_events(protocol) if turn < 3 else test_app.events(protocol))
                            values = round_value(values, turn)
                            if turn == 2:
                                # Chat/Messages split JSON arguments across separate events.
                                for event in values:
                                    if not isinstance(event, dict):
                                        continue
                                    delta = event.get('delta', {})
                                    if isinstance(delta, dict) and delta.get('partial_json') in ('0}', '1}'):
                                        delta['partial_json'] = str(int(delta['partial_json'][0]) + 17) + '}'
                                    for choice in event.get('choices', []):
                                        for call in choice.get('delta', {}).get('tool_calls', []):
                                            function = call.get('function', {})
                                            if function.get('arguments') in ('0}', '1}'):
                                                function['arguments'] = str(int(function['arguments'][0]) + 17) + '}'
                            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                stream=test_app.ByteStream(test_app.sse(protocol, values), size=1))
                        document = test_agent_loop.tool_reply(protocol) if turn < 3 else test_app.output(protocol)
                        document = round_value(document, turn)
                        document['id'] = f'round-{turn}-response'
                        return httpx.Response(200, json=document)

                    self.handler = upstream
                    model = (await self.model_for(protocol)).bind_tools([lookup])

                    async def infer(state):
                        if streaming:
                            chunks = [chunk async for chunk in model.astream(state['messages'])]
                            answer = chunks[0]
                            for chunk in chunks[1:]:
                                answer += chunk
                        else:
                            answer = await model.ainvoke(state['messages'])
                        return {'messages': [answer]}

                    graph = StateGraph(MessagesState)
                    graph.add_node('infer', infer)
                    graph.add_node('tools', ToolNode([lookup], handle_tool_errors=False))
                    graph.add_edge(START, 'infer')
                    graph.add_conditional_edges('infer', lambda state: 'tools' if state['messages'][-1].tool_calls else END)
                    graph.add_edge('tools', 'infer')
                    with patch('langchain_anthropic.chat_models._get_default_async_httpx_client', return_value=self.http):
                        result = await graph.compile().ainvoke({'messages': [HumanMessage(content='Synthetic two-round fixture')]},
                                                               {'recursion_limit': 7})
                    self.assertEqual(len(sent), 3)
                    self.assertEqual(effects[:2], [1, 0])
                    self.assertEqual(sorted(effects), [0, 1, 17, 18])
                    observed = {message.tool_call_id: message.content for message in result['messages'] if isinstance(message, ToolMessage)}
                    self.assertEqual(observed, {'round-1-call-0': '17', 'round-1-call-1': '18',
                                                'round-2-call-0': '34', 'round-2-call-1': '35'})
                    history = sent[2]['input' if protocol == 'responses' else 'messages']
                    if protocol == 'responses':
                        opaque = [item['encrypted_content'] for item in history if item.get('type') == 'reasoning']
                        self.assertEqual(opaque, ['synthetic-encrypted'] * 2)
                    elif protocol == 'messages':
                        assistants = [item['content'] for item in history if item['role'] == 'assistant']
                        self.assertEqual([blocks[0]['signature'] for blocks in assistants], ['synthetic-signature'] * 2)
                        self.assertEqual([blocks[1]['data'] for blocks in assistants], ['synthetic-opaque'] * 2)
                    self.assertIn('OK', str(result['messages'][-1].content))

    async def test_reused_tool_ids_across_closed_rounds_and_incomplete_function_call_stream(self):
        from llm_proxy_standard.protocol import Stream
        from llm_proxy_standard.rules import validate_history
        from tests.support import profile

        caps = {'text', 'tools', 'parallel_tools', 'streaming'}
        # Closed round 1 with call_0/call_1 followed by round 2 reusing call_0/call_1 (Grok behavior) passes:
        chat_history = [
            {'role': 'user', 'content': 'Start'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_0', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":0}'}},
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":1}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': '17'},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': '18'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_0', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":2}'}},
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":3}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': '19'},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': '20'},
        ]
        validate_history(chat_history, 'chat', caps)

        # Duplicate tool ID within the same open batch still fails:
        with self.assertRaisesRegex(ValueError, 'Missing or reused tool call identity'):
            validate_history([
                {'role': 'user', 'content': 'Start'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'call_0', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}},
                    {'id': 'call_0', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}},
                ]},
            ], 'chat', caps)

        # Starting a new tool batch while a prior batch still has unclosed calls fails:
        with self.assertRaisesRegex(ValueError, 'Missing tool results in replay'):
            validate_history(chat_history[:3] + [chat_history[4]], 'chat', caps)

        # Responses stream where a function_call is cut off by max_output_tokens (status='incomplete')
        # reaches response.incomplete without raising JSONDecodeError on the partial arguments string:
        validator = Stream('responses', profile('responses', capabilities=list(caps)))
        truncated_item = {
            'id': 'fc_trunc', 'type': 'function_call', 'call_id': 'call_trunc',
            'name': 'lookup', 'arguments': '{"code": "unterminated', 'status': 'incomplete',
        }
        events = [
            {'type': 'response.created', 'sequence_number': 0, 'response': {'id': 'resp_1', 'object': 'response', 'status': 'in_progress', 'output': []}},
            {'type': 'response.output_item.added', 'sequence_number': 1, 'output_index': 0, 'item': {**truncated_item, 'arguments': ''}},
            {'type': 'response.function_call_arguments.delta', 'sequence_number': 2, 'output_index': 0, 'item_id': 'fc_trunc', 'call_id': 'call_trunc', 'delta': '{"code": "unterminated'},
            {'type': 'response.function_call_arguments.done', 'sequence_number': 3, 'output_index': 0, 'item_id': 'fc_trunc', 'call_id': 'call_trunc', 'arguments': '{"code": "unterminated'},
            {'type': 'response.output_item.done', 'sequence_number': 4, 'output_index': 0, 'item': truncated_item},
            {'type': 'response.incomplete', 'sequence_number': 5, 'response': {
                'id': 'resp_1', 'object': 'response', 'status': 'incomplete',
                'incomplete_details': {'reason': 'max_output_tokens'}, 'output': [truncated_item],
            }},
        ]
        for ev in events:
            raw = f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()
            validator.event(raw)
        validator.finish()
        self.assertTrue(validator.terminal)

    async def test_universal_client_normalization_replay_cache_and_zero_touch_promotion(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from promote_model import build_universal_profile, build_package_for_profile
        from llm_proxy_standard.rules import prepare
        from llm_proxy_standard.replay_cache import ReplayCache
        from llm_proxy_standard.registry import Trust, Registry
        from llm_proxy_standard.publisher import Publisher
        from llm_proxy_standard.cache import SnapshotCache
        from llm_proxy_standard.contracts import digest, canonical
        import tempfile

        artifact = 'a' * 64
        # 1. Chat normalization (Luna with reasoning_none=True & Grok with reasoning_none=False)
        luna = build_universal_profile('openai-gpt-6-luna', 'chat', artifact, reasoning_none=True)
        grok = build_universal_profile('grok-4.7', 'chat', artifact, reasoning_none=False)
        raw_chat = {
            'model': 'openai-gpt-6-luna',
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'tool_calls': [{'id': 'call_0', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'ok1'},
                {'role': 'assistant', 'tool_calls': [{'id': 'call_0', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'ok2'},
            ],
            'max_tokens': 1024,
            'max_completion_tokens': 200000,
            'reasoning_effort': 'medium',
            'frequency_penalty': 0,
            'user': 'n8n-client',
            'parallel_tool_calls': True,
        }
        eff_luna, _ = prepare(raw_chat, luna)
        self.assertEqual(eff_luna['reasoning_effort'], 'none')
        self.assertNotIn('max_tokens', eff_luna)
        self.assertEqual(eff_luna['max_completion_tokens'], 128000)
        self.assertNotIn('frequency_penalty', eff_luna)
        self.assertNotIn('user', eff_luna)
        self.assertNotIn('parallel_tool_calls', eff_luna)
        # Second round call_0 was automatically deduplicated:
        self.assertEqual(eff_luna['messages'][1]['tool_calls'][0]['id'], 'call_0')
        self.assertEqual(eff_luna['messages'][3]['tool_calls'][0]['id'], 'call_0_r2')
        self.assertEqual(eff_luna['messages'][4]['tool_call_id'], 'call_0_r2')

        eff_grok, _ = prepare({**raw_chat, 'model': 'grok-4.7'}, grok)
        self.assertNotIn('reasoning_effort', eff_grok)

        # 2. Responses normalization (n8n store=True, missing include, extra temperature)
        astra = build_universal_profile('openai-gpt-6-astra', 'responses', artifact)
        self.assertEqual(astra.response_stream, 'ordered-items')
        eff_resp, _ = prepare({
            'model': 'openai-gpt-6-astra',
            'input': [{'type': 'message', 'role': 'user', 'content': 'hi'}],
            'store': True,
            'temperature': 0.7,
            'text': {},
        }, astra)
        self.assertIs(eff_resp['store'], False)
        self.assertEqual(eff_resp['include'], ['reasoning.encrypted_content'])
        self.assertNotIn('temperature', eff_resp)

        # 3. Messages normalization (thinking enabled + temperature + low max_tokens + string tool_choice)
        opus = build_universal_profile('claude-opus-5-5', 'messages', artifact)
        eff_msg, _ = prepare({
            'model': 'claude-opus-5-5',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 2048,
            'thinking': {'type': 'enabled', 'budget_tokens': 4096},
            'temperature': 0.0,
            'tools': [{'name': 'lookup', 'input_schema': {'type': 'object'}}],
            'tool_choice': 'auto',
        }, opus)
        self.assertNotIn('temperature', eff_msg)
        self.assertEqual(eff_msg['max_tokens'], 8192)
        self.assertEqual(eff_msg['tool_choice'], {'type': 'auto'})

        # 4. ReplayCache rehydration for unpatched n8n Responses and Messages
        rc = ReplayCache()
        rc.record_document('responses', {
            'output': [
                {'type': 'reasoning', 'id': 'rs-1', 'summary': [], 'encrypted_content': 'enc-secret-123'},
                {'type': 'function_call', 'id': 'fc-1', 'call_id': 'call-n8n-1', 'name': 'lookup', 'arguments': '{}'},
            ]
        })
        stripped_resp = {
            'model': 'openai-gpt-6-astra',
            'input': [
                {'type': 'message', 'role': 'user', 'content': 'hi'},
                {'type': 'function_call', 'id': 'fc-1', 'call_id': 'call-n8n-1', 'name': 'lookup', 'arguments': '{}'},
                {'type': 'function_call_output', 'call_id': 'call-n8n-1', 'output': 'done'},
            ]
        }
        rehydrated_resp, r_changes = rc.rehydrate(stripped_resp, 'responses')
        self.assertTrue(r_changes)
        self.assertEqual(rehydrated_resp['input'][1]['type'], 'reasoning')
        self.assertEqual(rehydrated_resp['input'][1]['encrypted_content'], 'enc-secret-123')

        rc.record_document('messages', {
            'content': [
                {'type': 'thinking', 'thinking': 'deep thought', 'signature': 'sig-abc'},
                {'type': 'redacted_thinking', 'data': 'opaque-xyz'},
                {'type': 'tool_use', 'id': 'toolu-n8n-1', 'name': 'lookup', 'input': {}},
            ]
        })
        stripped_msg = {
            'model': 'claude-opus-5-5',
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'toolu-n8n-1', 'name': 'lookup', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu-n8n-1', 'content': 'done'},
                ]},
            ]
        }
        rehydrated_msg, m_changes = rc.rehydrate(stripped_msg, 'messages')
        self.assertTrue(m_changes)
        self.assertEqual(rehydrated_msg['messages'][1]['content'][0]['signature'], 'sig-abc')
        self.assertEqual(rehydrated_msg['messages'][1]['content'][1]['data'], 'opaque-xyz')

        # 5. Zero-touch 365-day promotion via admin_approvers without static pin list
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox, checkpoints = root / 'inbox', root / 'checkpoints'
            inbox.mkdir()
            checkpoints.mkdir()
            trust = Trust({}, 'EXAMPLE_ACCOUNT', 'b' * 64, 'https://example-account.snowflakecomputing.com/api/v2/cortex/v1',
                          admin_approvers=['operator-admin'], historical_artifacts=[artifact], serve_historical_artifacts=True)
            reg = Registry(root / 'registry.sqlite', trust)
            pub = Publisher(reg, inbox, checkpoints)
            try:
                entry, report, ev_name, pkg, pkg_name = build_package_for_profile(grok, digest(reg.read().document()))
                (inbox / ev_name).write_bytes(canonical(report))
                (inbox / pkg_name).write_bytes(canonical(pkg))
                results = pub.scan()
                self.assertEqual(results[0]['status'], 'accepted')
                async def fetch():
                    return reg.read().document()
                cache = SnapshotCache(fetch, trust)
                self.assertTrue(await cache.refresh())
                self.assertEqual([p.id for p in cache.profiles('chat')], ['grok-4.7'])
            finally:
                pub.close()