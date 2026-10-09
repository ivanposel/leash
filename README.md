# leash

[![tests](https://github.com/ivanposel/leash/actions/workflows/ci.yml/badge.svg)](https://github.com/ivanposel/leash/actions/workflows/ci.yml)


**Keep small local LLMs on a leash: look, think, act, and verify — so they never act blind.**

A small local model (Llama, Qwen, Phi, running on Ollama) is fast, private, and free — and *confident and wrong* just often enough to be dangerous when it drives real actions: clicking buttons, posting, buying, deleting, sending. `leash` wraps any action in a four-step loop so the model earns each action instead of guessing it:

```
  LOOK   ─▶  observe the real state of the world
  THINK  ─▶  ask the model: does acting *now* actually advance the goal?
  ACT    ─▶  only if it's confident AND the precondition holds
  VERIFY ─▶  observe again and confirm the action did what you wanted
```

If the THINK step errors or is unsure, the agent **does not act**. Fail-closed is the default, not an option you have to remember to turn on.

- 🪶 **Zero dependencies.** Pure standard library. `pip install leash` and go.
- 🔌 **Model-agnostic.** Ships an Ollama client, but any `chat(prompt) -> str` callable works — local or otherwise.
- 🧩 **Environment-agnostic.** `observe`, `act`, `verify` are just callables. A screen, an API, a shell, a game — same loop.
- 🔒 **Fail-closed.** Errors, garbled JSON, low confidence, or a failed precondition all resolve to *don't act*.

---

## Why this exists

Most "agent" loops are `prompt → parse → do it`. That's fine until the model:

- acts when the goal is **already satisfied** (double-posts, double-charges),
- acts on a **stale or wrong** view of the world,
- returns **confident nonsense** that the code trusts because it parsed,
- takes an action that **silently fails** and nobody checks.

`leash` closes all four holes with one small, boring, testable loop. The model only ever answers one narrow question — *"should I act right now?"* — and the library handles grounding, gating, and verification around it.

## Install

```bash
pip install leash             # once published
# or, from source:
pip install -e .
```

Python 3.8+. No third-party packages.

## Quickstart (no model needed)

```python
from leash import grounded_act
from leash.reason import heuristic_reasoner

world = {"light": "off"}

outcome = grounded_act(
    goal="the light should be ON",
    observe=lambda: dict(world),                       # LOOK
    act=lambda: world.update(light="on"),             # ACT
    think=heuristic_reasoner(                          # THINK
        lambda goal, obs, action: obs["light"] != "on"
    ),
    verify=lambda before, after: after["light"] == "on",  # VERIFY
)

print(outcome.ok)        # True — acted and verified
```

Run it again and the gate **declines** to act, because the goal is already met. That refusal is the whole point.

```bash
python examples/quickstart.py     # runs with zero setup
```

## With a real local model

```python
from leash import grounded_act
from leash.reason import llm_reasoner
from leash.ollama import chat_fn

think = llm_reasoner(chat_fn(model="qwen2.5:3b"))     # any Ollama model

outcome = grounded_act(
    goal="submit the completed login form",
    observe=read_screen,        # your screen reader -> a description/state
    act=click_submit,           # your click
    think=think,                # the model decides go / no-go
    verify=lambda before, after: "submitted" in after.lower(),
    min_confidence=0.7,
)
```

The model is handed the goal, what the agent sees right now, and the proposed action, and must answer with a strict verdict:

```json
{"act": false, "confidence": 0.85, "rationale": "the form fields are still empty"}
```

Anything it can't parse into that shape is treated as `act: false`. See [`examples/desktop_agent.py`](examples/desktop_agent.py) for a runnable version.

## The API

### `grounded_act(goal, *, observe, act, think, verify=None, ...) -> Outcome`

| argument         | type                                  | role |
|------------------|---------------------------------------|------|
| `goal`           | `str`                                 | what the agent is trying to achieve |
| `observe`        | `() -> state`                         | **LOOK** — grounds every decision in reality; called before thinking and after acting |
| `act`            | `() -> Any`                           | **ACT** — the side effect |
| `think`          | `(goal, obs, action_desc) -> Decision`| **THINK** — the go/no-go gate (usually a local LLM) |
| `verify`         | `(before, after) -> bool`             | **VERIFY** — did it actually work? (optional but recommended) |
| `precondition`   | `(obs) -> bool`                       | cheap deterministic guard, checked *before* the model |
| `min_confidence` | `float`                               | model must be at least this sure, or the agent skips acting |
| `retries`        | `int`                                 | if verify fails, re-act up to this many extra times |
| `action_desc`    | `str`                                 | human/model-readable description of `act` |
| `on_reject`      | `(Decision) -> None`                  | called when the agent declines to act |
| `log`            | `(str) -> None`                       | optional line logger to trace the loop |

Returns an **`Outcome`** — `acted`, `verified`, `decision`, `observation_before/after`, `error`, `attempts`. `bool(outcome)` / `outcome.ok` is `True` only when the agent acted and verification didn't fail.

### Thinkers (`leash.reason`)

- `llm_reasoner(chat, prompt=DELIBERATION_PROMPT)` — wrap any `chat(prompt) -> str` model as a gate.
- `heuristic_reasoner(predicate)` — no model; a deterministic `(goal, obs, action) -> bool`.
- `parse_decision(raw)` — pull a strict JSON verdict out of a chatty model reply, fail-closed.

### Ollama (`leash.ollama`)

- `generate(prompt, model="qwen2.5:3b", ...) -> str` — stdlib-only call to a local Ollama server.
- `chat_fn(model=...)` — a one-arg `chat(prompt)` bound to a model, ready for `llm_reasoner`.

## Design principles

1. **Ground every decision.** The model reasons over what's *actually* on screen right now, not what it remembers or assumes.
2. **Separate deciding from doing.** The model only decides *whether* to act. The library does the acting, gating, and checking — the parts you want deterministic.
3. **Verify, don't hope.** An action that isn't checked is an action you can't trust. `verify` turns "I clicked" into "it worked."
4. **Fail closed.** Every ambiguous path — error, unparseable reply, low confidence, failed precondition — resolves to *no action*.

## Tests

```bash
python -m pytest        # or: python -m unittest discover
```

## License

MIT — see [LICENSE](LICENSE).
