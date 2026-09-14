"""OllamaClient contract tests: structured outputs, context sizing, keep-alive,
batched embeddings, and the informed JSON retry — all against a mocked daemon."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import config
from app.llm_client import OllamaClient

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


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _chat_payload(content: str) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 100,
        "eval_count": 20,
    }


def test_chat_json_sends_bare_schema_and_options() -> None:
    client = OllamaClient(host="http://test:11434")
    with patch("requests.post", return_value=_response(_chat_payload('{"answer": "ok"}'))) as post:
        result = client.chat_json(
            model="qwen3:test", system="sys", user="usr", json_schema=SCHEMA, temperature=0.0,
        )
    payload = post.call_args.kwargs["json"]
    # Structured outputs: the OpenAI envelope is unwrapped to the bare schema.
    assert payload["format"] == SCHEMA["schema"]
    assert payload["options"]["num_ctx"] == config.OLLAMA_NUM_CTX
    assert payload["options"]["num_predict"] > 0
    assert payload["keep_alive"] == config.OLLAMA_KEEP_ALIVE
    assert "top_p" not in payload["options"]  # greedy decoding: no sampling knobs
    assert result.usage.prompt_tokens == 100
    assert result.usage.total_tokens == 120  # total present on the schema path too


def test_chat_json_large_context_and_sampling() -> None:
    client = OllamaClient(host="http://test:11434")
    with patch("requests.post", return_value=_response(_chat_payload('{"answer": "ok"}'))) as post:
        client.chat_json(
            model="qwen3:test", system="sys", user="usr",
            json_schema=SCHEMA, temperature=0.7, large_context=True,
        )
    options = post.call_args.kwargs["json"]["options"]
    assert options["num_ctx"] == config.OLLAMA_NUM_CTX_LARGE
    assert options["top_p"] == 0.8  # qwen3-instruct guidance when sampling


def test_chat_json_retry_shows_model_its_bad_output() -> None:
    client = OllamaClient(host="http://test:11434")
    bad = _chat_payload("not json at all")
    good = _chat_payload('{"answer": "fixed"}')
    with patch("requests.post", side_effect=[_response(bad), _response(good)]) as post:
        result = client.chat_json(
            model="qwen3:test", system="sys", user="usr", json_schema=SCHEMA,
        )
    assert json.loads(result.content) == {"answer": "fixed"}
    retry_messages = post.call_args_list[1].kwargs["json"]["messages"]
    # The failed output is in the retry transcript so the model can correct it,
    # and sampling is nudged off the deterministic path.
    assert any(m["role"] == "assistant" and m["content"] == "not json at all" for m in retry_messages)
    assert post.call_args_list[1].kwargs["json"]["options"]["temperature"] >= 0.3


def test_chat_json_raises_after_two_invalid() -> None:
    client = OllamaClient(host="http://test:11434")
    bad = _response(_chat_payload("still not json"))
    with patch("requests.post", return_value=bad):
        with pytest.raises(RuntimeError, match="invalid JSON twice"):
            client.chat_json(model="qwen3:test", system="s", user="u", json_schema=SCHEMA)


def test_embed_batches_via_api_embed() -> None:
    client = OllamaClient(host="http://test:11434")
    with patch(
        "requests.post",
        return_value=_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
    ) as post:
        vectors = client.embed(model="nomic-embed-text", inputs=["a", "b"])
    assert post.call_args.args[0].endswith("/api/embed")
    assert post.call_args.kwargs["json"]["input"] == ["a", "b"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_count_mismatch_raises() -> None:
    client = OllamaClient(host="http://test:11434")
    with patch("requests.post", return_value=_response({"embeddings": [[0.1]]})):
        with pytest.raises(RuntimeError, match="1 embeddings for 2 inputs"):
            client.embed(model="nomic-embed-text", inputs=["a", "b"])


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
