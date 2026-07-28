"""
grounded.reason — turn a local LLM (or a plain heuristic) into a go/no-go gate.

The THINK step of the loop takes (goal, observation, action_desc) and returns a
Decision. Here are two ready-made thinkers:

  * llm_reasoner(chat)     — ask a local model for a strict JSON verdict
  * heuristic_reasoner(fn) — no model at all; a deterministic predicate

Both fail closed: anything they can't parse or are unsure about becomes
`act=False`, so a garbled model response can never green-light an action.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .core import Decision

__all__ = ["DELIBERATION_PROMPT", "llm_reasoner", "heuristic_reasoner", "parse_decision"]


DELIBERATION_PROMPT = """You are the deliberation gate of an autonomous agent.
Before the agent performs an action, you decide whether acting RIGHT NOW
genuinely advances the goal. Be skeptical:

- If the current state already satisfies the goal, decline (act=false).
- If the action looks premature, unsafe, or wrong for this exact state, decline.
- Only approve when you are confident the action moves toward the goal.

GOAL:
{goal}

WHAT THE AGENT OBSERVES RIGHT NOW:
{observation}

PROPOSED ACTION:
{action}

Reply with a SINGLE JSON object and nothing else:
{{"act": true or false, "confidence": a number from 0.0 to 1.0, "rationale": "one short sentence"}}
"""


def _short(observation: Any, limit: int = 2000) -> str:
    text = observation if isinstance(observation, str) else repr(observation)
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def parse_decision(raw: str) -> Decision:
    """Pull the first JSON object out of a model reply and turn it into a Decision.

    Fail-closed: if there is no parseable verdict, we return act=False.
    """
    if not raw:
        return Decision(False, 0.0, "empty model response", raw=raw)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Decision(False, 0.0, "no JSON object in response", raw=raw)

    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return Decision(False, 0.0, "unparseable JSON verdict", raw=raw)

    act = bool(data.get("act", False))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    rationale = str(data.get("rationale", "")).strip()
    return Decision(act=act, confidence=confidence, rationale=rationale, raw=raw)


def llm_reasoner(
    chat: Callable[[str], str],
    prompt: str = DELIBERATION_PROMPT,
) -> Callable[[str, Any, str], Decision]:
    """Build a THINK function backed by a local model.

    `chat` is any callable that takes a prompt string and returns the model's
    text reply — e.g. `functools.partial(grounded.ollama.generate, model="qwen2.5:3b")`.
    """

    def think(goal: str, observation: Any, action_desc: str) -> Decision:
        message = prompt.format(
            goal=goal,
            observation=_short(observation),
            action=action_desc or "(unspecified)",
        )
        reply = chat(message)
        return parse_decision(reply)

    return think


def heuristic_reasoner(
    should_act: Callable[[str, Any, str], bool],
    confidence: float = 1.0,
    rationale: str = "heuristic gate",
) -> Callable[[str, Any, str], Decision]:
    """Build a THINK function with no model — just a deterministic predicate.

    Handy for tests, offline runs, or cheap guards you don't want to spend a
    model call on.
    """

    def think(goal: str, observation: Any, action_desc: str) -> Decision:
        act = bool(should_act(goal, observation, action_desc))
        return Decision(act=act, confidence=confidence if act else 0.0, rationale=rationale)

    return think
