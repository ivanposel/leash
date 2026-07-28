"""Tests for the grounded loop. Run: python -m pytest  (or python -m unittest)."""
import unittest

from leash import grounded_act
from leash.core import Decision
from leash.reason import heuristic_reasoner, parse_decision


def always(v):
    return heuristic_reasoner(lambda g, o, a: v)


class TestGate(unittest.TestCase):
    def test_declines_when_thinker_says_no(self):
        acted = []
        out = grounded_act(
            "goal", observe=lambda: "state", act=lambda: acted.append(1),
            think=always(False),
        )
        self.assertFalse(out.acted)
        self.assertEqual(acted, [])

    def test_acts_and_verifies(self):
        state = {"done": False}
        out = grounded_act(
            "goal",
            observe=lambda: dict(state),
            act=lambda: state.update(done=True),
            think=always(True),
            verify=lambda b, a: a["done"] is True,
        )
        self.assertTrue(out.acted)
        self.assertTrue(out.verified)
        self.assertTrue(out.ok)

    def test_low_confidence_blocks(self):
        think = lambda g, o, a: Decision(act=True, confidence=0.3)
        out = grounded_act("g", observe=lambda: 0, act=lambda: 1 / 0, think=think, min_confidence=0.7)
        self.assertFalse(out.acted)

    def test_precondition_fail_closed(self):
        out = grounded_act(
            "g", observe=lambda: "bad", act=lambda: 1 / 0, think=always(True),
            precondition=lambda obs: obs == "good",
        )
        self.assertFalse(out.acted)

    def test_think_error_is_fail_closed(self):
        def boom(g, o, a):
            raise RuntimeError("model down")
        out = grounded_act("g", observe=lambda: 0, act=lambda: 1 / 0, think=boom)
        self.assertFalse(out.acted)
        self.assertIsNotNone(out.error)

    def test_retry_until_verified(self):
        calls = {"n": 0}

        def act():
            calls["n"] += 1

        # verify only passes on the 2nd action
        out = grounded_act(
            "g", observe=lambda: calls["n"], act=act, think=always(True),
            verify=lambda b, a: calls["n"] >= 2, retries=2,
        )
        self.assertTrue(out.ok)
        self.assertEqual(out.attempts, 2)


class TestParse(unittest.TestCase):
    def test_parses_clean_json(self):
        d = parse_decision('{"act": true, "confidence": 0.9, "rationale": "go"}')
        self.assertTrue(d.act)
        self.assertAlmostEqual(d.confidence, 0.9)

    def test_extracts_json_from_chatter(self):
        d = parse_decision('Sure! {"act": false, "confidence": 0.2, "rationale": "no"} hope that helps')
        self.assertFalse(d.act)

    def test_garbage_fails_closed(self):
        d = parse_decision("I think you should probably click it, yeah.")
        self.assertFalse(d.act)
        self.assertEqual(d.confidence, 0.0)

    def test_confidence_clamped(self):
        d = parse_decision('{"act": true, "confidence": 5}')
        self.assertEqual(d.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
