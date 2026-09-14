"""Settings page: one-click Local ↔ API mode toggle, per-mode model choices.

Local mode runs everything on this machine (Ollama + on-device transcription);
API mode uses OpenAI. Each mode keeps its own model bundle, so toggling never
loses choices. Changes apply immediately — no restart, no .env editing.
"""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import api_client

WHISPER_PROVIDERS = ["local", "openai", "none"]


async def render_settings_page() -> None:
    ui.label("Settings").classes("text-2xl font-bold")
    ui.label(
        "Local mode keeps everything on this machine — lectures, notes, and models. "
        "API mode sends text to OpenAI for higher-end hosted models."
    ).classes("text-sm text-gray-500 mb-2")

    try:
        current = await api_client.get_settings()
    except Exception as exc:  # noqa: BLE001
        ui.label(f"Could not load settings: {exc}").classes("text-red-700")
        return

    try:
        ollama_models = (await api_client.get_ollama_models()).get("models") or []
    except Exception:  # noqa: BLE001 — daemon may be down; free-text entry still works
        ollama_models = []

    @ui.refreshable
    def page(current: dict[str, Any]) -> None:
        is_local = current["mode"] == "local"

        async def switch_mode(mode: str) -> None:
            if mode == current["mode"]:
                return
            try:
                updated = await api_client.put_mode(mode)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative", multi_line=True, close_button="OK")
                page.refresh(current)  # snap the toggle back
                return
            ui.notify(
                "Local mode: everything runs on this machine."
                if mode == "local"
                else "API mode: generation now uses OpenAI.",
                type="positive",
            )
            page.refresh(updated)

        with ui.card().classes("w-full max-w-2xl gap-2"):
            with ui.row().classes("items-center gap-4"):
                ui.label("Mode").classes("text-lg font-semibold")
                ui.toggle(
                    {"local": "Local (private)", "api": "OpenAI API"},
                    value=current["mode"],
                    on_change=lambda e: switch_mode(str(e.value)),
                ).props("color=primary")
            ui.label(
                "Local: Ollama models + on-device transcription. Nothing leaves this machine."
                if is_local
                else "API: OpenAI models + Whisper. Requires an API key."
            ).classes("text-xs text-gray-500")

        with ui.card().classes("w-full max-w-2xl gap-2 mt-3"):
            if is_local:
                ui.label("Local models (Ollama)").classes("text-lg font-semibold")
                ollama_host = ui.input("Ollama host", value=current["ollama_host"]).classes("w-full")

                def model_field(label: str, key: str) -> Any:
                    if ollama_models:
                        return ui.select(
                            sorted(set(ollama_models + [current[key]])),
                            value=current[key],
                            label=label,
                            with_input=True,
                        ).classes("w-full")
                    return ui.input(label, value=current[key]).classes("w-full")

                generation_model = model_field("Generation model", "local_generation_model")
                chat_model = model_field("Chat model", "local_chat_model")
                retrieval_model = model_field("Embedding model", "local_retrieval_model")
                with ui.expansion("Advanced").classes("w-full"):
                    aux_model = model_field(
                        "Aux model (high-volume extraction passes)", "local_aux_model"
                    )
                    whisper = ui.select(
                        WHISPER_PROVIDERS,
                        value=current["whisper_provider"],
                        label="Transcription (local = on-device, none = disable video)",
                    )
                    # The key stays enterable from local mode: switching to API
                    # mode requires one, so hiding this field would make the
                    # toggle a dead end on a keyless install.
                    key_placeholder = (
                        f"Stored: {current['openai_api_key_masked']} — leave blank to keep"
                        if current["openai_api_key_set"]
                        else "sk-... (needed to switch to API mode)"
                    )
                    api_key = ui.input(
                        "OpenAI API key", password=True, placeholder=key_placeholder
                    ).classes("w-full")
                payload_keys = {
                    "local_generation_model": generation_model,
                    "local_chat_model": chat_model,
                    "local_retrieval_model": retrieval_model,
                    "local_aux_model": aux_model,
                }
                aux_use_local = None
            else:
                ui.label("OpenAI").classes("text-lg font-semibold")
                key_placeholder = (
                    f"Stored: {current['openai_api_key_masked']} — leave blank to keep"
                    if current["openai_api_key_set"]
                    else "sk-..."
                )
                api_key = ui.input(
                    "OpenAI API key", password=True, placeholder=key_placeholder
                ).classes("w-full")
                generation_model = ui.input(
                    "Generation model", value=current["openai_generation_model"]
                ).classes("w-full")
                chat_model = ui.input("Chat model", value=current["openai_chat_model"]).classes("w-full")
                retrieval_model = ui.input(
                    "Embedding model", value=current["openai_retrieval_model"]
                ).classes("w-full")
                with ui.expansion("Advanced").classes("w-full"):
                    aux_model = ui.input(
                        "Aux model (high-volume extraction passes)",
                        value=current["openai_aux_model"],
                    ).classes("w-full")
                    aux_use_local = ui.switch(
                        "Route aux passes to local Ollama when available",
                        value=current["aux_use_local"],
                    )
                    whisper = ui.select(
                        WHISPER_PROVIDERS,
                        value=current["whisper_provider"],
                        label="Transcription (local = on-device, none = disable video)",
                    )
                    ollama_host = ui.input("Ollama host", value=current["ollama_host"]).classes("w-full")
                payload_keys = {
                    "openai_generation_model": generation_model,
                    "openai_chat_model": chat_model,
                    "openai_retrieval_model": retrieval_model,
                    "openai_aux_model": aux_model,
                }

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
                payload: dict[str, Any] = {
                    key: field.value for key, field in payload_keys.items()
                }
                payload["whisper_provider"] = whisper.value
                payload["ollama_host"] = ollama_host.value
                if aux_use_local is not None:
                    payload["aux_use_local"] = aux_use_local.value
                if api_key is not None:
                    key_value = (api_key.value or "").strip()
                    if key_value:
                        payload["openai_api_key"] = key_value
                try:
                    updated = await api_client.put_settings(payload)
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"Failed to apply settings: {exc}", type="negative", multi_line=True)
                    return
                ui.notify("Settings applied.", type="positive")
                page.refresh(updated)

            with ui.row().classes("mt-3"):
                if is_local:
                    ui.button("Test Ollama", on_click=lambda: test("ollama")).props("outline")
                else:
                    ui.button("Test OpenAI", on_click=lambda: test("openai")).props("outline")
                ui.button("Apply", on_click=apply).props("unelevated color=primary")

    page(current)

    ui.link("← Back to app", "/").classes("mt-3")
