# Architecture & Protocol Normalization

> **Disclaimer:** This project is an independent community / field-engineering implementation built on top of Snowflake Cortex REST APIs and Snowpark Container Services (SPCS). It is **NOT an official Snowflake product**.

## 1. Three Native API Surfaces

`snowflake-proxy-cortexrestapi` exposes three distinct protocol prefixes on port `8080` and routes validated requests to `https://<account-host>/api/v2/cortex/v1`:

| Protocol | Client Base URL | Discovery Route | Inference Route | Upstream Cortex Path |
|---|---|---|---|---|
| **OpenAI Chat Completions** | `http://<proxy>:8080/chat/v1` | `GET /chat/v1/models` | `POST /chat/v1/chat/completions` | `/api/v2/cortex/v1/chat/completions` |
| **OpenAI Responses** | `http://<proxy>:8080/responses/v1` | `GET /responses/v1/models` | `POST /responses/v1/responses` | `/api/v2/cortex/v1/responses` |
| **Anthropic Messages** | `http://<proxy>:8080/messages` (or `/messages/v1`) | `GET /messages/v1/models` | `POST /messages/v1/messages` | `/api/v2/cortex/v1/messages` |

## 2. Authentication Translation (SPCS OAuth)

Inside Snowpark Container Services, Snowflake mounts a fresh OAuth token at `/snowflake/session/token`. External tools (`n8n`, `Hermes`, `Langflow`, `LangChain`) inside the same SPCS network connect to the proxy over internal DNS (`*.svc.spcs.internal`) using a dummy placeholder API key, and the proxy injects on every request:

- `Authorization: Bearer <token_from_/snowflake/session/token>`
- `X-Snowflake-Authorization-Token-Type: OAUTH`
- `anthropic-version: 2023-06-01` (on `/messages/v1/messages`)

## 3. Universal Client Normalization (`rules.py`)

When `ProfileV2.normalize_client_options = True`, the proxy transparently normalizes incoming client payloads (and records all adjustments in the `x-proxy-adjustments` response header):

1. **Token Limits (`1..128000`):** Accepts any positive integer for `max_tokens`, `max_completion_tokens`, and `max_output_tokens`, clamping values above the model's upper bound. On `messages`, defaults `max_tokens = 4096` if omitted. On `responses`, leaves `max_output_tokens` unset when omitted by the client so long reasoning blocks do not truncate downstream `function_call` JSON arguments.
2. **Responses Protocol Adjustments:**
   - Forces `store = false` (required by stateless Cortex Responses, even when n8n sends `store = true`).
   - Adds `include = ["reasoning.encrypted_content"]` when `reasoning` capability is enabled.
   - Strips empty `"text": {}` objects sent by n8n that would otherwise trigger `HTTP 400` upstream.
   - Strips unsupported `temperature` / `top_p` fields and `prompt_cache_retention` hints on reasoning Responses models.
   - Clamps out-of-domain `reasoning.effort` values (e.g. `"none"` / `"minimal"` -> `"low"`) and `reasoning.summary` values to the profile's declared enum domain.
   - Enables `response_stream = "ordered-items"` (`stream_order.py` + `protocol.py`), reordering interleaved SSE events, accepting empty or provisional `encrypted_content` on `response.output_item.added`, verifying final `encrypted_content` equality between `response.output_item.done` and `response.completed`, and backfilling `response.output_item.added` so `langchain-openai`, `Langflow`, `Hermes`, and `n8n` receive the finalized opaque payload cleanly.
3. **Chat Protocol Adjustments:**
   - Resolves conflicts when a client sends both `max_tokens` and `max_completion_tokens`.
   - Sets `reasoning_effort = "none"` when required by a profile (even if the client sends `"medium"` or omits it), and removes `reasoning_effort` on Chat models that do not support it.
4. **Messages Protocol Adjustments:**
   - When `thinking.type` is outside the profile's supported enum (for example `"adaptive"` sent by clients before profile negotiation), strips the unsupported `thinking` block and paired `output_config.effort` instead of failing with `HTTP 400`.
   - When `thinking.type == "enabled"`, strips incompatible `temperature` / `top_p` parameters and bumps `max_tokens` if `max_tokens <= thinking.budget_tokens`.
   - Normalizes string `tool_choice` (`"auto"` / `"none"`) to `{"type": ...}`.
5. **Multi-Round Tool Call ID Deduplication:**
   - Automatically renames reused tool call IDs (`call_0..call_N`) from prior closed turns (`call_0` -> `call_0_r2`) in both the assistant turn and matching tool result.
6. **1M+ Token Context & Long-Stream Resource Bounds (`app.py`):**
   - Defaults `max_body = 32 MiB` (`PROXY_MAX_BODY=33554432`), `max_response = 32 MiB` (`PROXY_MAX_RESPONSE=33554432`), `deadline = 600s` (`PROXY_DEADLINE=600`), and upstream `httpx.Timeout(180, connect=10, write=60, pool=5)` so 1M+ token context windows (`openai-gpt-6-astra`, `openai-gpt-6.1-sol`, `claude-opus-5-5`) with multi-round `reasoning.encrypted_content` blobs and long context-compression streams complete without premature `HTTP 413` rejections or mid-stream cutoff (`incomplete chunked read`).

## 4. Stateless Client `ReplayCache` (`replay_cache.py`)

A bounded in-memory LRU cache (`max_entries=512`, `ttl=3600s`) inspects outgoing JSON and SSE responses for `responses` and `messages`:
- When a response emits tool calls (`function_call` or `tool_use`) alongside `reasoning.encrypted_content` (Responses) or `thinking` / `signature` / `redacted_thinking` blocks (Messages), it caches those blocks keyed by tool call ID.
- If the client's follow-up request omits those blocks (for example, unpatched n8n Agent engine serialization), `ReplayCache.rehydrate()` restores them automatically before forwarding to Cortex.
