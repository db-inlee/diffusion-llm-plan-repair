"""The model call, kept behind a seam.

A repairer should be readable as "build a prompt, read a plan back" without any of the API
surface bleeding into it. Everything about *how* the text is produced lives here, so a different
backend — another vendor, a local model later — is a different object rather than an edit to the
repair logic.

Two implementations ship: :class:`OpenAIClient`, which makes the real call, and
:class:`ScriptedLLMClient`, which returns answers prepared in advance. The scripted one is what
the tests use, so the whole repair path — prompt, parse, score — is exercised without a key,
without a network and without spending anything.

Determinism: there is less of a lever here than one would like. The default model rejects any
temperature other than its own (``400 Unsupported value: 'temperature' does not support 0.0 with
this model``), so the request omits the parameter rather than pretending to pin it — see
:attr:`OpenAIClient.temperature`. Even where it can be set, an API is not a deterministic
function of its input, and repeated calls may differ. Nothing here pretends otherwise; the
statistical handling of that variance belongs to the comparison ticket.
"""

import os
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class LLMError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


@runtime_checkable
class LLMClient(Protocol):
    """Turns a system and user message into text. Raises :class:`LLMError` on failure."""

    def complete(self, *, system: str, user: str) -> str: ...


class ScriptedLLMClient:
    """Returns prepared answers in order, recording what it was asked.

    An entry that is an exception is raised instead of returned, which is how API failures are
    exercised. Running out of answers is itself an error: a repairer that calls more often than
    the test expects should not pass quietly.
    """

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self._index >= len(self._responses):
            raise AssertionError(
                f"scripted client ran out of responses after {self._index} call(s)"
            )
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def call_count(self) -> int:
        return self._index


class OpenAIClient:
    """Calls an OpenAI chat model.

    The key is read from the ``OPENAI_API_KEY`` environment variable and never accepted as a
    literal, so it cannot end up in the source or in a commit. ``max_output_tokens`` is always
    sent: a repairer loops over corruptions and domains, and an unbounded response length is how
    a small experiment becomes an expensive one.

    ``temperature`` defaults to ``None``, meaning the parameter is left out of the request. The
    default model accepts only its own value and fails the call for any other, so sending a zero
    would not buy determinism — it would buy a 400.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self._client: Any | None = None

    def complete(self, *, system: str, user: str) -> str:
        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:  # the SDK raises its own hierarchy; the repairer only needs one
            raise LLMError(f"{self.model} call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError(f"{self.model} returned an empty response")
        text: str = content
        return text

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
            raise LLMError(
                "the openai package is required for live calls; install it or use a scripted client"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, timeout=self.timeout)
        return self._client


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "LLMClient",
    "LLMError",
    "OpenAIClient",
    "ScriptedLLMClient",
]
