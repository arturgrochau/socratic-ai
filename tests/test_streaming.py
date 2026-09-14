"""Streaming chat: the client-agnostic turn (prepare -> stream -> finish), the
NDJSON route, and the OpenAI stream adapter."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llm_client import OpenAIClient, StreamState


class _StreamingFake:
    """Stands in for get_llm_client(): streams a fixed answer."""

    name = "fake"

    def __init__(self, text: str = "Because the premise assumes what it sets out to prove.") -> None:
        self.text = text
        self.calls: list[dict] = []

    def chat_stream(self, *, model, system, user, temperature=0.3, large_context=False, state=None):
        self.calls.append({"model": model, "system": system, "user": user})
        for word in self.text.split(" "):
            yield word + " "
        if state is not None:
            state.usage.prompt_tokens = 10
            state.usage.completion_tokens = 5
            state.usage.total_tokens = 15


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/stream.db")
    monkeypatch.setenv("SOCRATIC_CONFIG_DIR", str(tmp_path / "cfg"))
    import config

    importlib.reload(config)
    import app.cost_logging as cost_logging
    import app.interaction as interaction
    import routes.interaction as route

    importlib.reload(cost_logging)
    importlib.reload(interaction)
    importlib.reload(route)
    cost_logging.ensure_cost_logging_tables()
    interaction.ensure_interaction_tables()
    return interaction, route


def test_stream_user_query_persists_full_answer(isolated):
    interaction, _ = isolated
    fake = _StreamingFake()
    with patch.object(interaction, "get_llm_client", return_value=fake), patch.object(
        interaction, "retrieve_context", side_effect=ValueError("no sources yet")
    ), patch.object(interaction, "_load_generated_learning_context", return_value=""):
        events = list(interaction.stream_user_query("Why?", [1], "s-1", user_id="u1"))
    deltas = [e["delta"] for e in events if "delta" in e]
    assert "".join(deltas).strip() == fake.text
    assert events[-1]["done"] is True
    assert events[-1]["answer"] == fake.text
    # Plain-prose prompt for the stream, never the JSON instruction.
    assert "Reply with the answer text only" in fake.calls[0]["system"]
    assert "matching the schema" not in fake.calls[0]["system"]
    # The turn is on the session, so the next question sees it as history.
    state = interaction._load_interaction_session("s-1", "u1")
    assert state is not None and state.turns[-1].answer == fake.text


def test_ask_stream_route_emits_ndjson(isolated):
    interaction, route = isolated
    fake = _StreamingFake("Short answer.")
    app = FastAPI()
    app.include_router(route.router)
    with patch.object(interaction, "get_llm_client", return_value=fake), patch.object(
        interaction, "retrieve_context", side_effect=ValueError("none")
    ), patch.object(interaction, "_load_generated_learning_context", return_value=""):
        with TestClient(app) as http:
            with http.stream(
                "POST",
                "/ask/stream",
                json={"session_id": "s-2", "source_ids": [1], "query": "Q?", "top_k": 4},
                headers={"X-User-ID": "u1"},
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("application/x-ndjson")
                lines = [json.loads(line) for line in resp.iter_lines() if line.strip()]
    assert lines[-1]["done"] is True
    assert "".join(e["delta"] for e in lines[:-1]).strip() == "Short answer."


def test_ask_stream_route_reports_validation_errors_inline(isolated):
    _, route = isolated
    app = FastAPI()
    app.include_router(route.router)
    with TestClient(app) as http:
        resp = http.post(
            "/ask/stream",
            json={"session_id": "s-3", "source_ids": [1], "query": "   ", "top_k": 4},
            headers={"X-User-ID": "u1"},
        )
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events and "error" in events[-1]
    assert "empty" in events[-1]["error"]


def test_openai_chat_stream_collects_deltas_and_usage():
    def chunk(text=None, usage=None):
        delta = SimpleNamespace(content=text)
        choices = [SimpleNamespace(delta=delta)] if text is not None else []
        return SimpleNamespace(choices=choices, usage=usage, model="gpt-4o-mini")

    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    stream = [chunk("Hel"), chunk("lo"), chunk(None, usage=usage)]
    client = OpenAIClient(api_key="sk-test")
    with patch.object(client._client.chat.completions, "create", return_value=iter(stream)) as create:
        state = StreamState()
        text = "".join(client.chat_stream(model="gpt-4o-mini", system="s", user="u", state=state))
    assert text == "Hello"
    assert state.usage.total_tokens == 5
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}


def test_openai_compatible_base_url_skips_stream_options_and_falls_back_on_schema():
    client = OpenAIClient(api_key="k", base_url="http://localhost:1234/v1")
    assert str(client.raw.base_url).rstrip("/") == "http://localhost:1234/v1"

    class Rejects(Exception):
        status_code = 400

    good = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "ok"}'))], usage=None
    )
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema":
            raise Rejects("400: 'response_format' json_schema is not supported")
        return good

    schema = {"name": "x", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}
    with patch.object(client._client.chat.completions, "create", side_effect=create):
        result = client.chat_json(model="m", system="s", user="u", json_schema=schema)
        client.chat_json(model="m", system="s", user="u", json_schema=schema)
    assert result.content == '{"answer": "ok"}'
    assert [c["response_format"]["type"] for c in calls] == ["json_schema", "json_object", "json_object"]
