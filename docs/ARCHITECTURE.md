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
   - Strips unsupported `temperature` / `top_p` fields on reasoning Responses models.
   - Enables `response_stream = "ordered-items"` (`stream_order.py`), reordering interleaved SSE events and ensuring `encrypted_content` finalized at `response.output_item.done` is present for `langchain-openai`.
3. **Chat Protocol Adjustments:**
   - Resolves conflicts when a client sends both `max_tokens` and `max_completion_tokens`.
   - Sets `reasoning_effort = "none"` when required by a profile (even if the client sends `"medium"` or omits it), and removes `reasoning_effort` on Chat models that do not support it.
4. **Messages Protocol Adjustments:**
   - When `thinking.type == "enabled"`, strips incompatible `temperature` / `top_p` parameters and bumps `max_tokens` if `max_tokens <= thinking.budget_tokens`.
   - Normalizes string `tool_choice` (`"auto"` / `"none"`) to `{"type": ...}`.
5. **Multi-Round Tool Call ID Deduplication:**
   - Automatically renames reused tool call IDs (`call_0..call_N`) from prior closed turns (`call_0` -> `call_0_r2`) in both the assistant turn and matching tool result.

## 4. Stateless Client `ReplayCache` (`replay_cache.py`)

A bounded in-memory LRU cache (`max_entries=512`, `ttl=3600s`) inspects outgoing JSON and SSE responses for `responses` and `messages`:
- When a response emits tool calls (`function_call` or `tool_use`) alongside `reasoning.encrypted_content` (Responses) or `thinking` / `signature` / `redacted_thinking` blocks (Messages), it caches those blocks keyed by tool call ID.
- If the client's follow-up request omits those blocks (for example, unpatched n8n Agent engine serialization), `ReplayCache.rehydrate()` restores them automatically before forwarding to Cortex.
