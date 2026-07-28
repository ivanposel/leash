"""
grounded — give small local LLMs the discipline to look, think, act, and verify.

    from grounded import grounded_act
    from grounded.reason import llm_reasoner
    from grounded.ollama import chat_fn

    outcome = grounded_act(
        "the login form should be submitted",
        observe=read_screen,
        act=click_submit,
        think=llm_reasoner(chat_fn(model="qwen2.5:3b")),
        verify=lambda before, after: "welcome" in after,
    )
    if outcome.ok:
        ...  # acted and verified
"""
from .core import Decision, Outcome, grounded_act
from .reason import heuristic_reasoner, llm_reasoner, parse_decision

__version__ = "0.1.0"
__all__ = [
    "grounded_act",
    "Decision",
    "Outcome",
    "llm_reasoner",
    "heuristic_reasoner",
    "parse_decision",
    "__version__",
]
