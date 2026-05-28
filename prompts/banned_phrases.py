"""
Banned phrases / AI-isms that the generator should not produce.

Used in two places:
  * The critic prompt — concrete things to flag in a draft section.
  * The chat prompt — explicit "do not begin sentences with any of" list.

Keep this list focused on tells that mark text as AI-generated: stock
openers, throat-clearing transitions, and academic hedges. Do NOT add
ordinary discourse words ("however", "because") — those would gut the
prose.
"""

BANNED_OPENERS = [
    # Stock LLM openers
    "One critical mechanism",
    "It is important to note",
    "It is worth noting",
    "It should be noted",
    "This process involves",
    "This concept refers to",
    "In essence,",
    "At its core,",
    "Notably,",
    "Importantly,",
    "Crucially,",
    # Throat-clearing transitions
    "Furthermore,",
    "Moreover,",
    "Additionally,",
    "In addition,",
    "On the other hand,",
    "That being said,",
    # Closing platitudes
    "In conclusion,",
    "To summarize,",
    "Overall,",
    "Ultimately,",
    # Academic hedges that signal padding
    "A key consideration",
    "It is essential to",
    "It is crucial to",
    "Plays a vital role",
    "Plays a critical role",
    "delve into",
    "delves into",
    # Marketing-y filler
    "navigate the complexities",
    "the intricate dance of",
    "the ever-evolving landscape",
]


def banned_phrases_block(header: str = "Do not begin sentences with any of these phrases:") -> str:
    """Return a formatted block suitable for embedding in a system prompt."""
    bullet_list = "\n".join(f"  - {phrase}" for phrase in BANNED_OPENERS)
    return f"{header}\n{bullet_list}"
