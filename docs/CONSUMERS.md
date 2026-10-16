# Client Configuration Guide (n8n, Hermes, Langflow, LangChain)

> **Disclaimer:** This project is an independent community / field-engineering implementation built on top of Snowflake Cortex REST APIs and Snowpark Container Services (SPCS). It is **NOT an official Snowflake product**.

Assume your proxy service is reachable inside SPCS at:
`ORIGIN = http://llm-proxy-standard.<namespace>.svc.spcs.internal:8080`

Never paste a real Snowflake PAT into client UI fields on internal SPCS routes — use a dummy placeholder string (e.g., `internal-spcs`) because the proxy authenticates upstream using its mounted SPCS OAuth token.

---

## 1. n8n

Create three credentials in n8n:

1. **OpenAI API (Chat Completions):**
   - **Base URL:** `ORIGIN/chat/v1`
   - **API Key:** `internal-spcs`
   - In the **OpenAI Chat Model** node, leave *Use Responses API* **OFF**.
2. **OpenAI API (Responses API):**
   - **Base URL:** `ORIGIN/responses/v1`
   - **API Key:** `internal-spcs`
   - In the **OpenAI Chat Model** node, turn *Use Responses API* **ON**. (The proxy automatically normalizes `store=false`, strips empty `"text": {}` objects, and rehydrates `reasoning.encrypted_content` across tool turns via `ReplayCache`).
3. **Anthropic API (Messages API):**
   - **Base URL:** `ORIGIN/messages`
   - **API Key:** `internal-spcs`
   - (The proxy automatically rehydrates `thinking` signatures across tool turns via `ReplayCache`).

All three credentials dynamically populate the model dropdown from `GET /<protocol>/v1/models`.

---

## 2. Hermes Agent

Add the three providers in `config.yaml` with `discover_models: true` (no static model lists needed):

```yaml
providers:
  snowflake-standard-chat:
    name: Snowflake-Standard-Chat
    base_url: ORIGIN/chat/v1
    api_key: internal-spcs
    api_mode: chat_completions
    discover_models: true
  snowflake-standard-responses:
    name: Snowflake-Standard-Responses
    base_url: ORIGIN/responses/v1
    api_key: internal-spcs
    api_mode: codex_responses
    discover_models: true
  snowflake-standard-messages:
    name: Snowflake-Standard-Messages
    base_url: ORIGIN/messages
    api_key: internal-spcs
    api_mode: anthropic_messages
    discover_models: true
```

Whenever a new model is promoted into the Registry, Hermes discovers it automatically on `/model` refresh.

---

## 3. Langflow

Connect either standard Langflow model components or the reusable `Standard Chat`, `Standard Responses`, and `Standard Messages` components to:

- **Chat:** `ORIGIN/chat/v1`
- **Responses:** `ORIGIN/responses/v1` (with `use_responses_api=True`; the proxy serves `response_stream="ordered-items"` for full `langchain-openai` streaming compatibility)
- **Messages:** `ORIGIN/messages`

---

## 4. LangChain & LangGraph

See [`examples/standard_clients.py`](../examples/standard_clients.py):

```python
from examples.standard_clients import langchain_model

# Chat Completions
chat_llm = langchain_model(ORIGIN, "your-chat-model-id", "chat")

# OpenAI Responses API
responses_llm = langchain_model(ORIGIN, "your-responses-model-id", "responses")

# Anthropic Messages API
messages_llm = langchain_model(ORIGIN, "your-messages-model-id", "messages")
```
