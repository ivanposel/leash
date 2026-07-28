"""
leash.ollama — a zero-dependency client for a local Ollama server.

Uses only the standard library (urllib), so `leash` stays install-free.
If you already have an Ollama HTTP client you like, ignore this file and pass
your own `chat` callable to `llm_reasoner`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import partial
from typing import Any, Callable, Dict, Optional

__all__ = ["generate", "chat_fn"]

DEFAULT_HOST = "http://127.0.0.1:11434"


def generate(
    prompt: str,
    model: str = "qwen2.5:3b",
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Call Ollama's /api/generate and return the model's text reply.

    Deterministic by default (temperature 0) — a gate should not be creative.
    Raises on transport errors so the caller's fail-closed handling triggers.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def chat_fn(
    model: str = "qwen2.5:3b",
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
    options: Optional[Dict[str, Any]] = None,
) -> Callable[[str], str]:
    """Return a one-arg `chat(prompt) -> reply` bound to a model, for llm_reasoner."""
    return partial(generate, model=model, host=host, timeout=timeout, options=options)
