"""Tiny NiceGUI page used only by test_ui_smoke to exercise the interactive quiz
component (render_quiz) without needing a real generation run."""
from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-ui-smoke")
os.environ.setdefault("SOCRATIC_CONFIG_DIR", "/tmp/socratic-ui-smoke-cfg")

from nicegui import ui

from frontend import state
from frontend.components.quiz import render_quiz


QUIZ = {
    "questions": [
        {
            "question": "ZZQUESTION about coupling?",
            "options": ["alphaOPT", "betaOPT", "gammaOPT", "deltaOPT"],
            "answer_index": 1,
            "explanation": "OVERALLWHY beta is right.",
            "option_explanations": [
                "TRAPALPHA mistakes scale.",
                "RIGHTBETA names the mechanism.",
                "TRAPGAMMA confuses polarity.",
                "TRAPDELTA ignores delay.",
            ],
        }
    ]
}

DISCUSSED: dict[str, str] = {}


@ui.page("/quizprobe")
async def quizprobe() -> None:
    await ui.context.client.connected()
    state.tab()
    render_quiz(QUIZ, on_discuss=lambda seed: DISCUSSED.update(seed=seed))


# The NiceGUI test harness stubs ui.run(); it just needs the call to exist so it
# can discover this module's pages. (The real app uses ui.run_with in main.py.)
ui.run()
