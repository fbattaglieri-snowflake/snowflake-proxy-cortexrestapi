# snowflake-proxy-cortexrestapi

> **Disclaimer:** This project is an independent community / field-engineering implementation built on top of **Snowflake Cortex REST APIs** and **Snowflake Snowpark Container Services (SPCS)**. It is **NOT an official Snowflake product**, is not part of the official Snowflake product offering, and is provided "as is" without official Snowflake support or warranty.

Universal, protocol-validating LLM proxy and single-writer model registry for **Snowflake Cortex REST APIs** (`/api/v2/cortex/v1`), exposing three standard protocol surfaces inside Snowpark Container Services (SPCS) or local environments:

1. **OpenAI Chat Completions API** (`/chat/v1/chat/completions` & `/chat/v1/models`)
2. **OpenAI Responses API** (`/responses/v1/responses` & `/responses/v1/models`)
3. **Anthropic Messages API** (`/messages/v1/messages` & `/messages/v1/models`)

Designed for out-of-the-box compatibility with external agentic frameworks and orchestrators — specifically **n8n**, **Hermes**, **Langflow**, and **LangChain / LangGraph** — while enforcing zero-trust model governance and zero-touch promotion of qualified models.

---

## Why This Proxy Exists

When connecting standard external tools (**n8n**, **Hermes**, **Langflow**, **LangChain**) directly to the Snowflake Cortex REST API, several protocol and client friction points arise:

- **In-Container OAuth vs External SDK Auth:** Services running inside SPCS authenticate to Cortex using the mounted `/snowflake/session/token` with `X-Snowflake-Authorization-Token-Type: OAUTH`, whereas standard OpenAI/Anthropic SDKs expect a static `Bearer` API key.
- **Client Parameter Mismatches (`HTTP 400`):** Frameworks send default parameters (`store: true` and empty `"text": {}` in n8n Responses nodes, `parallel_tool_calls: true`, `prompt_cache_key`, `frequency_penalty`, `temperature` alongside extended `thinking` or `reasoning_effort`, or `reasoning_effort: "medium"` on models that require `"none"` or reject the field).
- **Responses SSE Interleaving:** Standard `langchain-openai` (`LangChain` / `Langflow`) expects ordered SSE output items and finalized `reasoning.encrypted_content` on `response.output_item.added`.
- **Stateless Client Tool Roundtrips:** Some orchestrators (such as unpatched n8n Agent engine serialization) strip `reasoning.encrypted_content` (on Responses) or `thinking` / `signature` / `redacted_thinking` blocks (on Messages) between tool-calling turns, causing upstream multi-turn validation failures.
- **Multi-Round Tool Call ID Collisions:** Certain models (e.g., `grok-4.6` / `kimi-k3`) number tool calls sequentially per response (`call_0..call_N` on every round), colliding in client history or checkpointers on round 2+.

`snowflake-proxy-cortexrestapi` solves all of these centrally inside the proxy so that **n8n, Hermes, Langflow, and LangChain work out-of-the-box without client patches**.

---

## Prerequisite: Enable Cross-Region Inference (`ANY_REGION`)

To maximize model availability across all three APIs (`Chat`, `Responses`, and `Messages`) on your Snowflake account, it is **strongly recommended** to enable **Cross-Region Inference** (`ANY_REGION`) when your organization's data residency and compliance policies allow it.

Check your current account setting:

```sql
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;
```

With `ACCOUNTADMIN` privileges, enable `ANY_REGION`:

```sql
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```

See the [Snowflake Cross-Region Inference documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference) for regional options (such as `AWS_US`, `AWS_EU`, or `ANY_REGION`).

---

## Key Capabilities

### 1. Universal Client Normalization (`ProfileV2.normalize_client_options = True`)
- **Flexible Token Budgets (`1..128000`):** Accepts any positive integer for `max_tokens`, `max_completion_tokens`, or `max_output_tokens` (automatically clamping values that exceed the model bound, setting a default `4096` on Messages when omitted, and leaving Cortex's native budget on Responses when omitted so long reasoning blocks never truncate `function_call` JSON arguments).
- **Chat (`/chat/v1`):** Automatically resolves `max_tokens` + `max_completion_tokens` conflicts; enforces `reasoning_effort="none"` when required by a model (even if the client sends `"medium"` or omits it); strips `reasoning_effort` on Chat models that do not accept it; strips harmless client hints (`frequency_penalty`, `presence_penalty`, `user`, `metadata`, `seed`).
- **Responses (`/responses/v1`):** Forces `store = false` (even when n8n sends `store = true`); defaults `include = ["reasoning.encrypted_content"]` for reasoning models; strips empty `"text": {}` objects sent by n8n; strips incompatible sampling fields (`temperature`, `top_p`) on reasoning models; and enables **`response_stream = "ordered-items"`** so unmodified `langchain-openai` and Langflow stream tool calls and opaque reasoning losslessly.
- **Messages (`/messages/v1`):** Automatically removes `temperature`/`top_p` when `thinking.type == "enabled"` and ensures `max_tokens > thinking.budget_tokens`; normalizes string `tool_choice` (`"auto"` / `"none"`) into `{"type": ...}`; strips orphan `tool_choice` / `parallel_tool_calls` when `tools` is empty.
- **Multi-Round Tool Call ID Deduplication:** Automatically renames duplicate tool call IDs across closed rounds (`call_0` -> `call_0_r2`) in both the assistant turn and matching tool result before forwarding to Cortex.

### 2. In-Memory `ReplayCache` for Stateless Clients (`src/llm_proxy_standard/replay_cache.py`)
- Bounded LRU cache (`max_entries=512`, `ttl=3600s`) that records `reasoning.encrypted_content` (Responses) and `thinking` / `signature` / `redacted_thinking` blocks (Messages) emitted in JSON or SSE responses, keyed by `call_id` / `tool_use.id`.
- When a stateless client (such as unpatched n8n Agent engine serialization) drops those blocks on the follow-up tool-result request, the proxy automatically rehydrates them before validation and upstream forwarding.

### 3. Zero-Touch Model Promotion & Single-Writer Registry
- Models are first qualified using the companion skill and test kit: [**`cortex-model-compatibility-matrix`**](https://github.com/fbattaglieri-snowflake/cortex-model-compatibility-matrix).
- Once qualified, promoting a new model into the running proxy requires **a single CLI command** (`scripts/promote_model.py`) that uploads a content-addressed `ProfileV2` package (with 365-day validity under `administrative-standard-v1`) to the Snowflake Stage inbox.
- The `registry_service` auto-ingests the package within 30 seconds, and all proxy replicas converge within another 30 seconds — **zero `ALTER SERVICE` or container restarts required**, and all connected clients (`Hermes`, `n8n`, `Langflow`, `LangChain`) discover the new model dynamically via `GET /<protocol>/v1/models`.

---

## Repository Structure

- `src/llm_proxy_standard/`:
  - `app.py`: FastAPI data-plane serving `/chat/v1`, `/responses/v1`, `/messages/v1`, `/healthz`, `/readyz`.
  - `rules.py`: Client-agnostic request normalization, `ProfileV2` option constraints, and multi-round tool ID deduplication.
  - `replay_cache.py`: Bounded LRU cache rehydrating dropped `encrypted_content` and `thinking` signatures across tool turns.
  - `protocol.py` & `stream_order.py`: Incremental SSE framing validation and `ordered-items` stream normalizer for Responses.
  - `contracts.py`, `registry.py`, `registry_service.py`, `publisher.py`, `cache.py`, `admin.py`: Content-addressed snapshot registry, stage-inbox auto-publisher, and single-flight replica cache.
- `scripts/promote_model.py`: Zero-touch CLI to generate and upload `ProfileV2` model packages to the Registry inbox.
- `examples/standard_clients.py`: Reference client factories for OpenAI SDK, Anthropic SDK, LangChain, and LangGraph.
- `deployment/`: SPCS service specification templates (`service-spec.yaml.template`, `trust-empty.json`).
- `tests/`: Offline synthetic test suite covering all 3 protocols, streaming, multi-round tool loops, client normalization, and registry lifecycle.
- `docs/`:
  - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Data-plane, control-plane, and protocol normalization details.
  - [`docs/CONSUMERS.md`](docs/CONSUMERS.md): Configuration guide for **n8n**, **Hermes**, **Langflow**, and **LangChain / LangGraph**.
  - [`docs/PROMOTION.md`](docs/PROMOTION.md): End-to-end model qualification and zero-touch promotion runbook.

---

## Quickstart & Offline Verification

Requirements: Python 3.11+. The offline test suite uses synthetic fixtures only and requires no Snowflake credentials or network access:

```bash
git clone https://github.com/fbattaglieri-snowflake/snowflake-proxy-cortexrestapi.git
cd snowflake-proxy-cortexrestapi
python3 -m venv .venv
.venv/bin/pip install -r requirements.test.lock -e .
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false .venv/bin/python -m unittest discover -s tests -t . -v
```

Build the container image for SPCS (`linux/amd64`):

```bash
docker build --platform linux/amd64 -t snowflake-proxy-cortexrestapi:latest .
```
