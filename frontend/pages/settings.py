"""Settings page: switch OpenAI API <-> local Ollama at runtime (no restart)."""
from __future__ import annotations

from nicegui import ui

from frontend import api_client


PROVIDERS = ["openai", "ollama"]


async def render_settings_page() -> None:
    ui.label("Settings").classes("text-2xl font-bold")
    ui.label(
        "Switch between the OpenAI API and a local Ollama model. Changes apply "
        "immediately — no restart, no .env editing."
    ).classes("text-sm text-gray-500 mb-2")

    try:
        current = await api_client.get_settings()
    except Exception as exc:  # noqa: BLE001
        ui.label(f"Could not load settings: {exc}").classes("text-red-700")
        return

    with ui.card().classes("w-full max-w-2xl gap-2"):
        ui.label("Providers").classes("text-lg font-semibold")
        llm = ui.select(PROVIDERS, value=current["llm_provider"], label="LLM provider (generation + chat)")
        embedding = ui.select(PROVIDERS, value=current["embedding_provider"], label="Embedding provider (retrieval)")
        whisper = ui.select(PROVIDERS, value=current["whisper_provider"], label="Whisper provider (transcription)")

        ui.label("OpenAI").classes("text-lg font-semibold mt-2")
        key_placeholder = (
            f"Stored: {current['openai_api_key_masked']} — leave blank to keep"
            if current["openai_api_key_set"]
            else "sk-..."
        )
        api_key = ui.input("OpenAI API key", password=True, placeholder=key_placeholder).classes("w-full")

        ui.label("Ollama (local)").classes("text-lg font-semibold mt-2")
        ollama_host = ui.input("Ollama host", value=current["ollama_host"]).classes("w-full")

        ui.label("Models").classes("text-lg font-semibold mt-2")
        generation_model = ui.input("Generation model", value=current["generation_model"]).classes("w-full")
        chat_model = ui.input("Chat model", value=current["chat_model"]).classes("w-full")
        retrieval_model = ui.input("Retrieval/embedding model", value=current["retrieval_model"]).classes("w-full")

        with ui.expansion("Auxiliary model (cheap high-volume passes)").classes("w-full"):
            aux_use_local = ui.switch("Route aux passes to local Ollama", value=current["aux_use_local"])
            aux_llm_provider = ui.select(PROVIDERS, value=current["aux_llm_provider"], label="Aux provider")
            aux_model = ui.input("Aux model (hosted)", value=current["aux_model"]).classes("w-full")
            aux_local_model = ui.input("Aux model (local Ollama)", value=current["aux_local_model"]).classes("w-full")

        async def test(provider: str) -> None:
            try:
                result = await api_client.test_connection(provider)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"{provider}: {exc}", type="negative")
                return
            ui.notify(
                f"{provider}: {result.get('detail', '')}",
                type="positive" if result.get("ok") else "negative",
            )

        async def apply() -> None:
            payload = {
                "llm_provider": llm.value,
                "embedding_provider": embedding.value,
                "whisper_provider": whisper.value,
                "ollama_host": ollama_host.value,
                "generation_model": generation_model.value,
                "chat_model": chat_model.value,
                "retrieval_model": retrieval_model.value,
                "aux_use_local": aux_use_local.value,
                "aux_llm_provider": aux_llm_provider.value,
                "aux_model": aux_model.value,
                "aux_local_model": aux_local_model.value,
            }
            key_value = (api_key.value or "").strip()
            if key_value:
                payload["openai_api_key"] = key_value
            try:
                await api_client.put_settings(payload)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"Failed to apply settings: {exc}", type="negative")
                return
            api_key.value = ""
            ui.notify("Settings applied.", type="positive")

        with ui.row().classes("mt-3"):
            ui.button("Test OpenAI", on_click=lambda: test("openai")).props("outline")
            ui.button("Test Ollama", on_click=lambda: test("ollama")).props("outline")
            ui.button("Apply", on_click=apply).props("unelevated color=primary")

    ui.link("← Back to app", "/").classes("mt-3")
