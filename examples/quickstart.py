"""
quickstart.py — runs with ZERO setup (no model, no network).

It shows the whole look -> think -> act -> verify loop against a tiny in-memory
"world", using a heuristic thinker. Run it:

    python examples/quickstart.py

Then try the real-model version in examples/desktop_agent.py once Ollama is up.
"""
from leash import grounded_act
from leash.reason import heuristic_reasoner

# A toy "world": a light switch we want to turn ON.
world = {"light": "off"}


def observe():
    return dict(world)                       # LOOK: snapshot the current state


def turn_on():
    world["light"] = "on"                    # ACT: the side effect


def already_satisfied(goal, obs, action):
    # THINK: only act if the light is not already on. A good gate declines to
    # act when the goal is already met — that is the whole point.
    return obs["light"] != "on"


think = heuristic_reasoner(already_satisfied, rationale="light is off, turning it on")

outcome = grounded_act(
    goal="the light should be ON",
    observe=observe,
    act=turn_on,
    think=think,
    action_desc="flip the light switch to ON",
    verify=lambda before, after: after["light"] == "on",
    log=print,
)

print("\n--- first run ---")
print("acted:", outcome.acted, "| verified:", outcome.verified, "| ok:", outcome.ok)

# Run it again. The light is already on, so the gate should DECLINE to act —
# no blind, redundant, or harmful re-action.
outcome2 = grounded_act(
    goal="the light should be ON",
    observe=observe,
    act=turn_on,
    think=think,
    action_desc="flip the light switch to ON",
    verify=lambda before, after: after["light"] == "on",
    log=print,
)

print("\n--- second run (goal already met) ---")
print("acted:", outcome2.acted, "| rationale:", outcome2.decision.rationale)
assert outcome.ok and not outcome2.acted, "expected act-then-decline"
print("\nOK — acted once, then correctly declined.")
