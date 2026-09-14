"""First-run Welcome page: pick Local or Cloud API, then get everything the
choice needs (Ollama running, models pulled, ffmpeg present, or an API key)
without leaving the app."""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import api_client

OLLAMA_URL = "https://ollama.com/download"


def _chip(ok: bool, label: str, *, warn: bool = False) -> None:
    color = "positive" if ok else ("warning" if warn else "negative")
    icon = "check_circle" if ok else ("warning" if warn else "cancel")
    ui.chip(label, icon=icon, color=color).props("outline dense")


def _copy_line(command: str) -> None:
    with ui.row().classes("items-center gap-2"):
        ui.code(command).classes("text-xs")
        ui.button(
            icon="content_copy",
            on_click=lambda: (ui.clipboard.write(command), ui.notify("Copied")),
        ).props("flat dense round size=sm")


async def render_welcome_page() -> None:
    try:
        report: dict[str, Any] = await api_client.get_setup_status()
    except Exception as exc:  # noqa: BLE001
        ui.label(f"Could not read the setup status: {exc}").classes("text-negative")
        return

    chosen: dict[str, str] = {"mode": "local"}

    ui.label("Welcome to Socratic AI").classes("text-3xl font-bold mt-2")
    ui.label(
        "It turns your lectures and PDFs into questions, and a chat that only cites your material. "
        "One decision first: where should the thinking happen?"
    ).classes("text-base text-gray-500 max-w-2xl")

    with ui.stepper().props("vertical flat").classes("w-full max-w-3xl") as stepper:
        # ── Step 1: choose ────────────────────────────────────────────────────
        with ui.step("Choose where models run"):
            rec = report["recommended"]
            ram = report.get("ram_gb")
            ram_text = f"{ram:g} GB of memory detected" if ram else "memory size unknown"
            with ui.row().classes("w-full gap-4 items-stretch"):
                with ui.card().classes("flex-1 min-w-[260px] cursor-pointer") as local_card:
                    ui.label("On this Mac (private)").classes("text-lg font-semibold")
                    ui.label(
                        f"Free. Nothing leaves the machine. {ram_text}, so the recommended model is "
                        f"{rec['generation_model']} (about {rec['download_gb']:g} GB to download once)."
                    ).classes("text-sm")
                    if rec.get("note"):
                        ui.label(rec["note"]).classes("text-xs text-gray-500")
                with ui.card().classes("flex-1 min-w-[260px] cursor-pointer") as api_card:
                    ui.label("Cloud API").classes("text-lg font-semibold")
                    ui.label(
                        "OpenAI, or any OpenAI-compatible service (OpenRouter, Groq, DeepSeek). "
                        "Needs an API key. Your material is sent to that provider. "
                        "Roughly $0.01 per pack on gpt-4o-mini."
                    ).classes("text-sm")

            def pick(mode: str) -> None:
                chosen["mode"] = mode
                local_card.classes(replace="flex-1 min-w-[260px] cursor-pointer" + (" border-2 border-primary" if mode == "local" else ""))
                api_card.classes(replace="flex-1 min-w-[260px] cursor-pointer" + (" border-2 border-primary" if mode == "api" else ""))
                local_step.set_visibility(mode == "local")
                api_step.set_visibility(mode == "api")

            local_card.on("click", lambda: pick("local"))
            api_card.on("click", lambda: pick("api"))
            with ui.stepper_navigation():
                ui.button("Continue", on_click=stepper.next).props("unelevated color=primary")

        # ── Step 2a: local checklist ──────────────────────────────────────────
        with ui.step("Get the local pieces ready") as local_step:
            ui.label("Three things, all free. This page re-checks every few seconds.").classes("text-sm text-gray-500")
            checklist = ui.column().classes("w-full gap-3")
            pull_state: dict[str, Any] = {"job_id": None}

            @ui.refreshable
            def render_checklist(rep: dict[str, Any]) -> None:
                ollama = rep["ollama"]
                ffmpeg = rep["ffmpeg"]
                missing = rep["missing_models"]
                recd = rep["recommended"]

                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-3"):
                        _chip(ollama["running"], "Ollama running" if ollama["running"] else "Ollama not running")
                        ui.label("Ollama runs the models.").classes("text-sm")
                    if not ollama["running"]:
                        if ollama["installed"]:
                            async def _start() -> None:
                                await api_client.start_ollama()
                                ui.notify("Starting Ollama. Give it a few seconds.")

                            ui.button("Start Ollama", icon="play_arrow", on_click=_start).props("outline")
                        else:
                            ui.link("Download Ollama for macOS", OLLAMA_URL, new_tab=True)
                            ui.label("or, if you use Homebrew:").classes("text-xs text-gray-500")
                            _copy_line("brew install ollama && brew services start ollama")

                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-3"):
                        if not missing:
                            _chip(True, "Models ready")
                        else:
                            _chip(False, f"{len(missing)} model(s) to download", warn=True)
                        ui.label(f"{recd['generation_model']} for thinking, {recd['embedding_model']} for search.").classes("text-sm")
                    if missing and ollama["running"]:
                        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
                        progress.set_visibility(False)
                        progress_label = ui.label("").classes("text-xs text-gray-500")

                        async def _poll() -> None:
                            job_id = pull_state.get("job_id")
                            if not job_id:
                                return
                            try:
                                job = await api_client.get_ollama_pull(job_id)
                            except Exception:  # noqa: BLE001
                                return
                            progress.set_value(job["percent"] / 100.0)
                            done_gb = job["completed"] / 1024**3
                            total_gb = job["total"] / 1024**3
                            progress_label.text = (
                                f"{job['current'] or ''}: {job['percent']:g}% ({done_gb:.1f} of {total_gb:.1f} GB)"
                                if job["total"]
                                else f"{job['current'] or ''}: {job['status']}"
                            )
                            if job["done"]:
                                poll_timer.deactivate()
                                pull_state["job_id"] = None
                                if job["status"] == "failed":
                                    ui.notify(f"Download failed: {job['error']}", type="negative", multi_line=True)
                                else:
                                    ui.notify("Models downloaded.", type="positive")
                                await refresh_report()

                        poll_timer = ui.timer(1.0, _poll, active=False)

                        async def _pull() -> None:
                            job = await api_client.start_ollama_pull(missing)
                            pull_state["job_id"] = job["job_id"]
                            progress.set_visibility(True)
                            pull_button.disable()
                            poll_timer.activate()

                        pull_button = ui.button(
                            f"Download {recd['download_gb']:g} GB", icon="download", on_click=_pull
                        ).props("unelevated color=primary")
                    elif missing:
                        ui.label("Start Ollama first, then the download button appears here.").classes("text-xs text-gray-500")

                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-3"):
                        _chip(ffmpeg["found"], "ffmpeg found" if ffmpeg["found"] else "ffmpeg missing", warn=True)
                        ui.label("Only needed for video and YouTube. PDFs work without it.").classes("text-sm")
                    if not ffmpeg["found"]:
                        _copy_line("brew install ffmpeg")
                        ui.label(
                            "No Homebrew? Get it at brew.sh, or skip this and add videos later."
                        ).classes("text-xs text-gray-500")

            async def refresh_report() -> None:
                # The progress bar and its poll timer live inside the checklist;
                # refreshing mid-download would delete them. The pull poller
                # refreshes once itself when the job finishes.
                if pull_state.get("job_id"):
                    return
                try:
                    fresh = await api_client.get_setup_status()
                except Exception:  # noqa: BLE001
                    return
                report.update(fresh)
                render_checklist.refresh(fresh)

            with checklist:
                render_checklist(report)
            ui.timer(4.0, refresh_report)

            async def finish_local() -> None:
                if not report["ollama"]["running"]:
                    ui.notify("Ollama needs to be running before you continue.", type="warning")
                    return
                if report["missing_models"]:
                    ui.notify("Download the models first (or switch to the Cloud API).", type="warning")
                    return
                await api_client.complete_setup("local")
                ui.navigate.to("/")

            with ui.stepper_navigation():
                ui.button("Back", on_click=stepper.previous).props("flat")
                ui.button("Finish", on_click=finish_local).props("unelevated color=primary")

        # ── Step 2b: API key ──────────────────────────────────────────────────
        with ui.step("Connect an API") as api_step:
            api_step.set_visibility(False)
            ui.label(
                "Paste a key. Leave the URL empty for OpenAI itself. A local server such as LM Studio needs no key."
            ).classes("text-sm text-gray-500")
            key = ui.input("API key", password=True, placeholder="sk-...").classes("w-full max-w-xl")
            base_url = ui.input(
                "Base URL (optional)", placeholder="https://openrouter.ai/api/v1"
            ).classes("w-full max-w-xl")
            ui.label("You can change models and every other detail later in Settings.").classes("text-xs text-gray-500")

            async def finish_api() -> None:
                value = (key.value or "").strip()
                if not value and not (base_url.value or "").strip():
                    ui.notify("An API key is required for OpenAI.", type="warning")
                    return
                try:
                    payload: dict[str, Any] = {"openai_api_key": value, "openai_base_url": (base_url.value or "").strip()}
                    await api_client.put_settings(payload)
                    result = await api_client.test_connection("openai")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"Could not save: {exc}", type="negative", multi_line=True)
                    return
                if not result.get("ok"):
                    # Stay here so the message is readable and the key can be corrected.
                    ui.notify(
                        f"The key was saved but the connection test failed: {result.get('detail')}",
                        type="warning", multi_line=True, close_button="OK",
                    )
                    return
                await api_client.complete_setup("api")
                ui.navigate.to("/")

            with ui.stepper_navigation():
                ui.button("Back", on_click=stepper.previous).props("flat")
                ui.button("Finish", on_click=finish_api).props("unelevated color=primary")

    pick("local")
