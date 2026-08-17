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

**An empty answer is a measurement too.** Six of sixteen ``ar_local`` cases came back with no
content at all, and the call that produced them was a successful one — HTTP 200, a well-formed
response object, an empty string inside it. Giving up on that without looking is how "the model
did not answer" stayed a guess for a whole stage. Every call is now recorded as an
:class:`LLMCall` — the finish reason, the token counts, a refusal if there was one — and the error
raised for an empty answer carries that record in its message, so the reason lands in the result
file the runner writes rather than in a shrug. ``finish_reason="length"`` means the budget ran out
before any answer was written; anything else means the budget is not the problem.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# The budget can be raised for a run without editing this file, because the value that is right
# depends on what the diagnosis above says and on what the run is worth paying for. It is bounded:
# a reasoning model bills every token it thinks with, and an experiment that loops over a matrix
# is exactly where an unbounded budget turns into a bill nobody decided on.
MAX_OUTPUT_TOKENS_VARIABLE = "PLAN_REPAIR_MAX_OUTPUT_TOKENS"
MAX_OUTPUT_TOKENS_CEILING = 32768


class LLMError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


@dataclass(frozen=True)
class LLMCall:
    """What one call to the model cost and how it ended.

    Kept as data rather than a log line because the interesting question is comparative — whether
    the calls that came back empty are the ones that spent their whole budget thinking — and that
    cannot be asked of prose.
    """

    model: str
    budget: int
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    refusal: str | None = None
    content_characters: int = 0

    @property
    def out_of_budget(self) -> bool:
        """Whether the model was cut off rather than choosing to stop."""
        return self.finish_reason == "length"

    def describe(self) -> str:
        """One line naming why an answer looks the way it does."""
        parts = [f"finish_reason={self.finish_reason or 'unreported'}"]
        if self.completion_tokens is not None:
            spent = f"{self.completion_tokens} completion token(s)"
            if self.reasoning_tokens is not None:
                spent += f", {self.reasoning_tokens} of them reasoning"
            parts.append(spent)
        parts.append(f"budget {self.budget}")
        if self.refusal:
            parts.append(f"refusal: {self.refusal}")
        if self.out_of_budget:
            parts.append("the budget ran out before an answer was written")
        return ", ".join(parts)


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

    ``max_output_tokens`` defaults to :data:`DEFAULT_MAX_OUTPUT_TOKENS`, or to whatever
    ``PLAN_REPAIR_MAX_OUTPUT_TOKENS`` says when it is set. The environment is the seam because a
    budget is a property of the run, not of the code: the same matrix is worth 4096 tokens a call
    while the answers fit and worth more once :attr:`last_call` says they did not. **A result file
    does not record which budget produced it** — a run at a different budget belongs in its own
    ``--out`` directory, or the aggregate cannot tell the two apart.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = (
            budget_from_environment() if max_output_tokens is None else max_output_tokens
        )
        self.timeout = timeout
        # What the last call cost and how it ended, for whoever wants to ask why an answer was
        # empty. Overwritten per call: this is a diagnostic, not a ledger.
        self.last_call: LLMCall | None = None
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
        self.last_call = self._record(response, content)
        if not content or not content.strip():
            raise LLMError(f"{self.model} returned an empty response — {self.last_call.describe()}")
        text: str = content
        return text

    def _record(self, response: Any, content: str | None) -> LLMCall:
        """Read the accounting off a response, tolerating a shape that does not carry it.

        Every field is optional on purpose. A record that raises while explaining why an answer
        was empty would replace the diagnosis with a second failure.
        """
        choice = _first(response, "choices")
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return LLMCall(
            model=self.model,
            budget=self.max_output_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            refusal=getattr(getattr(choice, "message", None), "refusal", None),
            content_characters=len(content or ""),
        )

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


def _first(response: Any, name: str) -> Any:
    items = getattr(response, name, None) or []
    return items[0] if items else None


def budget_from_environment() -> int:
    """The completion budget for this run.

    Refuses a value it cannot read and a value above :data:`MAX_OUTPUT_TOKENS_CEILING`, rather
    than falling back to the default: a budget that was asked for and quietly not applied would
    make the next run's empty answers unexplainable all over again.
    """
    raw = os.environ.get(MAX_OUTPUT_TOKENS_VARIABLE)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_OUTPUT_TOKENS
    try:
        budget = int(raw)
    except ValueError as exc:
        raise LLMError(
            f"{MAX_OUTPUT_TOKENS_VARIABLE} must be a positive integer; got {raw!r}"
        ) from exc
    if budget < 1:
        raise LLMError(f"{MAX_OUTPUT_TOKENS_VARIABLE} must be a positive integer; got {raw!r}")
    if budget > MAX_OUTPUT_TOKENS_CEILING:
        raise LLMError(
            f"{MAX_OUTPUT_TOKENS_VARIABLE}={budget} is above the ceiling of "
            f"{MAX_OUTPUT_TOKENS_CEILING}; a reasoning model bills every token it thinks with, so "
            "raise the ceiling deliberately rather than by environment"
        )
    return budget


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "MAX_OUTPUT_TOKENS_CEILING",
    "MAX_OUTPUT_TOKENS_VARIABLE",
    "LLMCall",
    "LLMClient",
    "LLMError",
    "OpenAIClient",
    "ScriptedLLMClient",
    "budget_from_environment",
]
