"""Wires up the NiceGUI pages. Called once from main.py before ui.run_with()."""
from __future__ import annotations

from nicegui import ui

from frontend import api_client, state
from frontend.pages.build import render_main_body
from frontend.pages.settings import render_settings_page
from frontend.pages.welcome import render_welcome_page

# Light-blue accent applied app-wide. One place to retune the whole theme.
PRIMARY = "#4F9DDE"
SECONDARY = "#6BA8E5"


def _apply_theme() -> None:
    ui.colors(primary=PRIMARY, secondary=SECONDARY)


def _header() -> None:
    with ui.header(elevated=True).classes("items-center justify-between").props("color=primary"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("school")
            ui.label("Socratic AI").classes("text-lg font-semibold")
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Reset",
                icon="restart_alt",
                on_click=lambda: (state.reset_builder(), ui.navigate.reload()),
            ).props("flat color=white")
            ui.button(icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
                "flat color=white"
            )


async def _setup_complete() -> bool:
    """Best effort: if the status probe itself fails, don't trap the user in the wizard."""
    try:
        return bool((await api_client.get_setup_status()).get("setup_complete"))
    except Exception:  # noqa: BLE001
        return True


def init_ui() -> None:
    @ui.page("/")
    async def index() -> None:
        _apply_theme()
        _header()
        with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-2"):
            # app.storage.tab needs an established client connection.
            await ui.context.client.connected()
            if not await _setup_complete():
                ui.navigate.to("/welcome")
                return
            state.tab()
            render_main_body()

    @ui.page("/welcome")
    async def welcome_page() -> None:
        _apply_theme()
        _header()
        with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-2"):
            await ui.context.client.connected()
            await render_welcome_page()

    @ui.page("/settings")
    async def settings_page() -> None:
        _apply_theme()
        _header()
        with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-2"):
            await render_settings_page()
