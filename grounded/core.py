"""
grounded.core — the look -> think -> act -> verify loop.

A small local model is confident, fast, and wrong just often enough to be
dangerous when it drives real actions (clicking, posting, buying, deleting).
This module wraps any action in four steps so the model never acts blind:

    1. LOOK    observe the current state of the world
    2. THINK   ask the model: does acting *now* actually advance the goal?
    3. ACT     only if it is confident and the precondition holds
    4. VERIFY  observe again and confirm the action had the intended effect

Everything is a plain callable, so it works with any model, any environment,
and has zero third-party dependencies. Fail-closed is the default: if the
"think" step errors or is unsure, the agent does NOT act.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["Decision", "Outcome", "grounded_act"]


@dataclass
class Decision:
    """The verdict of the THINK step: should the agent act, and how sure is it."""

    act: bool
    confidence: float = 0.0          # 0.0 (no idea) .. 1.0 (certain)
    rationale: str = ""              # one sentence, human-readable
    raw: Any = None                  # the model's raw response, for debugging

    def __post_init__(self) -> None:
        # Clamp so a model that returns 1.7 or -0.2 can't break the gate.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class Outcome:
    """The full record of one grounded_act call — what was seen, decided, done."""

    acted: bool
    verified: Optional[bool]         # True/False if a verifier ran, else None
    decision: Decision
    observation_before: Any = None
    observation_after: Any = None
    error: Optional[str] = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        """True when we acted and verification did not fail (or wasn't needed)."""
        return self.acted and self.verified is not False and self.error is None

    def __bool__(self) -> bool:
        return self.ok


def _noop(_msg: str) -> None:
    pass


def grounded_act(
    goal: str,
    *,
    observe: Callable[[], Any],
    act: Callable[[], Any],
    think: Callable[[str, Any, str], Decision],
    verify: Optional[Callable[[Any, Any], bool]] = None,
    action_desc: str = "",
    precondition: Optional[Callable[[Any], bool]] = None,
    min_confidence: float = 0.7,
    retries: int = 0,
    on_reject: Optional[Callable[[Decision], None]] = None,
    log: Callable[[str], None] = _noop,
) -> Outcome:
    """Run one grounded action: look, think, (maybe) act, verify.

    Args:
        goal:          what the agent is ultimately trying to achieve.
        observe:       () -> state. Grounds every decision in reality. Called
                       before thinking and again after acting.
        act:           () -> Any. The side-effecting action itself.
        think:         (goal, observation, action_desc) -> Decision. Usually a
                       local LLM (see grounded.reason.llm_reasoner), but any
                       callable works.
        verify:        (before, after) -> bool. Did the action actually change
                       the world the way we wanted? Optional but recommended.
        action_desc:   human/model-readable description of `act` for the thinker.
        precondition:  (observation) -> bool. A cheap, deterministic guard
                       checked before we even ask the model. Fail-closed.
        min_confidence:the model must be at least this sure, or we skip acting.
        retries:       if verify fails, re-act up to this many extra times.
        on_reject:     called with the Decision when the agent declines to act.
        log:           optional line logger for tracing the loop.

    Returns:
        Outcome with the full trace. `bool(outcome)` / `outcome.ok` tell you
        whether the action was taken and verified.
    """
    # 1. LOOK ---------------------------------------------------------------
    before = observe()
    log(f"look: {before!r}")

    # Deterministic guard first — never spend a model call on an invalid state.
    if precondition is not None and not precondition(before):
        d = Decision(act=False, confidence=1.0, rationale="precondition not met")
        log("gate: precondition not met -> skip")
        if on_reject:
            on_reject(d)
        return Outcome(False, None, d, before, before, attempts=0)

    # 2. THINK --------------------------------------------------------------
    try:
        decision = think(goal, before, action_desc)
    except Exception as exc:  # fail-closed: an error is NOT permission to act
        d = Decision(act=False, confidence=0.0, rationale=f"think() failed: {exc}")
        log(f"think: ERROR {exc} -> fail-closed skip")
        if on_reject:
            on_reject(d)
        return Outcome(False, None, d, before, before, error=str(exc), attempts=0)

    log(f"think: act={decision.act} conf={decision.confidence:.2f} :: {decision.rationale}")

    if not decision.act or decision.confidence < min_confidence:
        log(f"gate: declined (need >= {min_confidence:.2f}) -> skip")
        if on_reject:
            on_reject(decision)
        return Outcome(False, None, decision, before, before, attempts=0)

    # 3. ACT + 4. VERIFY ----------------------------------------------------
    after = before
    verified: Optional[bool] = None
    attempts = 0
    last_error: Optional[str] = None

    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            log(f"act: attempt {attempts}")
            act()
        except Exception as exc:
            last_error = f"act() failed: {exc}"
            log(f"act: ERROR {exc}")
            break

        after = observe()
        log(f"look: {after!r}")

        if verify is None:
            verified = None
            break

        try:
            verified = bool(verify(before, after))
        except Exception as exc:
            last_error = f"verify() failed: {exc}"
            verified = False
            log(f"verify: ERROR {exc}")
            break

        log(f"verify: {'ok' if verified else 'FAILED'}")
        if verified:
            break
        # else loop to retry (if any retries remain)

    return Outcome(
        acted=True,
        verified=verified,
        decision=decision,
        observation_before=before,
        observation_after=after,
        error=last_error,
        attempts=attempts,
    )
