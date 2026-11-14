"""One LLM interface for the whole package.

The original experiments had three different clients (``/api/chat`` with dataclasses,
``/api/generate`` with a hand-written ``<|system|>`` template, and ad-hoc ``requests``
calls). This module replaces them with a single :class:`Backend` protocol:

* :class:`OllamaBackend` talks to a local Ollama server through ``/api/chat``, so each
  model's own chat template is applied (the hand-written template was not any model's
  real format).
* :class:`ScriptedBackend` returns canned text and is what the tests use, so the suite
  runs without Ollama or a GPU.

Every call is stateless: no conversation history is shared between requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ChatRequest:
    """A single system + user exchange sent to one model."""

    model: str
    system: str
    user: str
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass(frozen=True)
class Completion:
    """Model output plus the wall-clock latency of the call."""

    text: str
    model: str
    latency_s: float


class LLMError(RuntimeError):
    """Raised when the backend cannot produce a completion."""


@runtime_checkable
class Backend(Protocol):
    """Anything that turns a :class:`ChatRequest` into a :class:`Completion`."""

    def complete(self, request: ChatRequest) -> Completion: ...


class OllamaBackend:
    """Stateless client for a local Ollama server.

    Network I/O releases the GIL, so callers may share one instance across a
    ``ThreadPoolExecutor``; ``requests.Session`` is used only for connection reuse.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def complete(self, request: ChatRequest) -> Completion:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system.strip()},
                {"role": "user", "content": request.user.strip()},
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}; is `ollama serve` running?"
            ) from exc
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise LLMError(
                f"Ollama call failed for model {request.model!r}: {exc}"
            ) from exc
        latency = time.perf_counter() - started
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response keys: {list(data)}") from exc
        return Completion(text=text.strip(), model=request.model, latency_s=latency)


class ScriptedBackend:
    """Deterministic backend for tests: ``responder(request) -> text``.

    It also records every request so tests can assert on what was sent.
    """

    def __init__(self, responder: Callable[[ChatRequest], str]):
        self._responder = responder
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(
            text=self._responder(request), model=request.model, latency_s=0.0
        )
