"""
desktop_agent.py — the same loop, but a real local model is the gate.

Requires a running Ollama server with a small model pulled, e.g.:

    ollama pull qwen2.5:3b

Here the "world" is a fake UI state machine so the example is safe to run, but
`observe` / `act` / `verify` are exactly where you would drop in a real screen
reader, a click, and a post-condition check. The model decides whether clicking
"Submit" right now actually advances the goal.
"""
from grounded import grounded_act
from grounded.reason import llm_reasoner
from grounded.ollama import chat_fn

# A fake app screen. Swap observe/act for real screen-reading + clicking.
screen = {"form": "empty", "submitted": False}


def read_screen():
    if screen["submitted"]:
        return "A confirmation page that says 'Thanks, your form was submitted.'"
    if screen["form"] == "filled":
        return "A login form with the username and password fields FILLED and a Submit button."
    return "A login form with EMPTY username and password fields and a Submit button."


def click_submit():
    # A real agent would tap the button. Here we only 'submit' if the form is
    # filled — an empty submit would fail, which is exactly what the gate should
    # prevent from happening in the first place.
    if screen["form"] == "filled":
        screen["submitted"] = True


def verify(before, after):
    return "submitted" in after.lower() or "thanks" in after.lower()


think = llm_reasoner(chat_fn(model="qwen2.5:3b"))

print("=== Case A: form is EMPTY — a good gate should DECLINE to submit ===")
screen.update({"form": "empty", "submitted": False})
a = grounded_act(
    goal="submit the completed login form",
    observe=read_screen,
    act=click_submit,
    think=think,
    action_desc="click the Submit button",
    verify=verify,
    min_confidence=0.7,
    log=print,
)
print("acted:", a.acted, "| rationale:", a.decision.rationale, "\n")

print("=== Case B: form is FILLED — the gate should APPROVE and it verifies ===")
screen.update({"form": "filled", "submitted": False})
b = grounded_act(
    goal="submit the completed login form",
    observe=read_screen,
    act=click_submit,
    think=think,
    action_desc="click the Submit button",
    verify=verify,
    min_confidence=0.7,
    log=print,
)
print("acted:", b.acted, "| verified:", b.verified, "| ok:", b.ok)
