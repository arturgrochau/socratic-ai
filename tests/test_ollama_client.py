"""OllamaClient contract tests: structured outputs, context sizing, keep-alive,
batched embeddings, streaming, and the informed JSON retry, all against a fake
daemon served through httpx.MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

import config
from app.llm_client import OllamaClient, StreamState

SCHEMA = {
    "name": "test_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}


class FakeDaemon:
    """Scripted Ollama: each queued item is one response body (dict -> JSON,
    list[dict] -> NDJSON stream). Records every request it served."""

    def __init__(self, *responses: dict | list) -> None:
        self.queue = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(body, list):
            ndjson = "".join(json.dumps(line) + "\n" for line in body)
            return httpx.Response(200, content=ndjson.encode(), headers={"content-type": "application/x-ndjson"})
        return httpx.Response(200, json=body)

    def client(self) -> OllamaClient:
        return OllamaClient(host="http://test:11434", client=httpx.Client(transport=httpx.MockTransport(self.handler)))

    def payload(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def _chat_payload(content: str) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 100,
        "eval_count": 20,
    }


def test_chat_json_sends_bare_schema_and_options() -> None:
    daemon = FakeDaemon(_chat_payload('{"answer": "ok"}'))
    result = daemon.client().chat_json(
        model="qwen3:test", system="sys", user="usr", json_schema=SCHEMA, temperature=0.0,
    )
    payload = daemon.payload()
    # Structured outputs: the OpenAI envelope is unwrapped to the bare schema.
    assert payload["format"] == SCHEMA["schema"]
    assert payload["options"]["num_ctx"] == config.OLLAMA_NUM_CTX
    assert payload["options"]["num_predict"] > 0
    assert payload["keep_alive"] == config.OLLAMA_KEEP_ALIVE
    assert "top_p" not in payload["options"]  # greedy decoding: no sampling knobs
    assert payload["think"] is False  # no hidden reasoning tokens on thinking models
    assert result.usage.prompt_tokens == 100
    assert result.usage.total_tokens == 120  # total present on the schema path too


def test_chat_json_large_context_and_sampling() -> None:
    daemon = FakeDaemon(_chat_payload('{"answer": "ok"}'))
    daemon.client().chat_json(
        model="qwen3:test", system="sys", user="usr", json_schema=SCHEMA, temperature=0.7, large_context=True,
    )
    options = daemon.payload()["options"]
    assert options["num_ctx"] == config.OLLAMA_NUM_CTX_LARGE
    assert options["top_p"] == 0.8  # qwen3-instruct guidance when sampling


def test_chat_json_retry_shows_model_its_bad_output() -> None:
    daemon = FakeDaemon(_chat_payload("not json at all"), _chat_payload('{"answer": "fixed"}'))
    result = daemon.client().chat_json(model="qwen3:test", system="sys", user="usr", json_schema=SCHEMA)
    assert json.loads(result.content) == {"answer": "fixed"}
    retry = daemon.payload(1)
    # The failed output is in the retry transcript so the model can correct it,
    # and sampling is nudged off the deterministic path.
    assert any(m["role"] == "assistant" and m["content"] == "not json at all" for m in retry["messages"])
    assert retry["options"]["temperature"] >= 0.3


def test_chat_json_raises_after_two_invalid() -> None:
    daemon = FakeDaemon(_chat_payload("still not json"))
    with pytest.raises(RuntimeError, match="invalid JSON twice"):
        daemon.client().chat_json(model="qwen3:test", system="s", user="u", json_schema=SCHEMA)


def test_embed_batches_via_api_embed() -> None:
    daemon = FakeDaemon({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    vectors = daemon.client().embed(model="nomic-embed-text", inputs=["a", "b"])
    assert daemon.requests[-1].url.path == "/api/embed"
    assert daemon.payload()["input"] == ["a", "b"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_count_mismatch_raises() -> None:
    daemon = FakeDaemon({"embeddings": [[0.1]]})
    with pytest.raises(RuntimeError, match="1 embeddings for 2 inputs"):
        daemon.client().embed(model="nomic-embed-text", inputs=["a", "b"])


def test_chat_stream_yields_deltas_and_records_usage() -> None:
    daemon = FakeDaemon([
        {"model": "qwen3:test", "message": {"role": "assistant", "content": "Soc"}, "done": False},
        {"model": "qwen3:test", "message": {"role": "assistant", "content": "rates "}, "done": False},
        {"model": "qwen3:test", "message": {"role": "assistant", "content": "asks."}, "done": False},
        {"model": "qwen3:test", "message": {"role": "assistant", "content": ""}, "done": True,
         "prompt_eval_count": 50, "eval_count": 7},
    ])
    state = StreamState()
    text = "".join(daemon.client().chat_stream(model="qwen3:test", system="s", user="u", state=state))
    assert text == "Socrates asks."
    payload = daemon.payload()
    assert payload["stream"] is True
    assert "format" not in payload  # plain prose: no grammar constraint
    assert payload["think"] is False
    assert state.usage.prompt_tokens == 50 and state.usage.completion_tokens == 7
    assert state.usage.total_tokens == 57
    assert state.model == "qwen3:test"


class TestRetrievalCollectionKeying:
    """Provider/model-keyed collections + task prefixes (embedding-dim fix)."""

    def test_collection_name_tracks_model(self, monkeypatch) -> None:
        from app import retrieval

        monkeypatch.setattr(config, "embedding_provider", lambda: "ollama")
        monkeypatch.setattr(config, "retrieval_model", lambda: "nomic-embed-text")
        local_name = retrieval._collection_name()
        monkeypatch.setattr(config, "embedding_provider", lambda: "openai")
        monkeypatch.setattr(config, "retrieval_model", lambda: "text-embedding-3-small")
        api_name = retrieval._collection_name()
        assert local_name != api_name
        assert local_name.startswith("interaction_retrieval__")
        # Chroma-legal characters only.
        import re

        assert re.fullmatch(r"[a-zA-Z0-9._-]+", local_name)

    def test_nomic_prefixes(self, monkeypatch) -> None:
        from app import retrieval

        monkeypatch.setattr(config, "retrieval_model", lambda: "nomic-embed-text")
        docs = retrieval._prefixed_for_embedding(["chunk text"], kind="document")
        queries = retrieval._prefixed_for_embedding(["a question"], kind="query")
        assert docs == ["search_document: chunk text"]
        assert queries == ["search_query: a question"]

    def test_openai_no_prefixes(self, monkeypatch) -> None:
        from app import retrieval

        monkeypatch.setattr(config, "retrieval_model", lambda: "text-embedding-3-small")
        assert retrieval._prefixed_for_embedding(["x"], kind="document") == ["x"]
