"""No network on import. Supply an explicitly approved origin and discovered ID."""
from openai import OpenAI
from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def sdk_clients(origin):
    origin = origin.rstrip("/")
    return {
        "chat": OpenAI(base_url=origin + "/chat/v1", api_key="internal-placeholder", max_retries=0),
        "responses": OpenAI(base_url=origin + "/responses/v1", api_key="internal-placeholder", max_retries=0),
        "messages": Anthropic(base_url=origin + "/messages", api_key="internal-placeholder", max_retries=0),
    }


def langchain_model(origin, identifier, protocol, *, options=None):
    options = dict(options or {})
    if protocol == "responses":
        # Stateless continuation needs opaque reasoning in the full message log.
        options.setdefault("store", False)
        options.setdefault("include", ["reasoning.encrypted_content"])
        options.setdefault("output_version", "responses/v1")
        if options["store"] is not False:
            raise ValueError("Standard Responses requires store=False")
        if options["output_version"] != "responses/v1":
            raise ValueError("Standard Responses requires output_version='responses/v1'")
        include = options["include"]
        if not isinstance(include, list) or "reasoning.encrypted_content" not in include:
            raise ValueError("Standard Responses requires reasoning.encrypted_content")
        extra = options.get("extra_body", {})
        if not isinstance(extra, dict):
            raise ValueError("Standard extra_body must be a dictionary")
        # SDK extra_body is merged last, so it must not undo this connection policy.
        if any(key in extra for key in ("store", "include", "previous_response_id")):
            raise ValueError("Configure Responses state through top-level options only")
        if options.get("previous_response_id") is not None or options.get("use_previous_response_id"):
            raise ValueError("Standard Responses requires full conversation history")
    if protocol == "messages":
        return ChatAnthropic(model=identifier, base_url=origin.rstrip("/") + "/messages",
                             api_key="internal-placeholder", max_retries=0, **options)
    if protocol not in {"chat", "responses"}:
        raise ValueError("Unknown protocol")
    return ChatOpenAI(model=identifier, base_url=origin.rstrip("/") + f"/{protocol}/v1",
                      api_key="internal-placeholder", max_retries=0,
                      use_responses_api=protocol == "responses", **options)


def langgraph_node(model):
    async def invoke(state):
        return {"messages": [await model.ainvoke(state["messages"])]}
    return invoke