"""The client seam: what it promises, and what it refuses to do without a key."""

import pytest

from plan_repair.repair import LLMClient, LLMError, OpenAIClient, ScriptedLLMClient
from plan_repair.repair.llm_client import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL


def test_both_clients_satisfy_the_port():
    assert isinstance(ScriptedLLMClient([]), LLMClient)
    assert isinstance(OpenAIClient(), LLMClient)


def test_the_scripted_client_answers_in_order_and_records_the_calls():
    client = ScriptedLLMClient(["first", "second"])

    assert client.complete(system="s", user="a") == "first"
    assert client.complete(system="s", user="b") == "second"
    assert [call["user"] for call in client.calls] == ["a", "b"]
    assert client.call_count == 2


def test_the_scripted_client_raises_a_scripted_exception():
    client = ScriptedLLMClient([LLMError("rate limited")])

    with pytest.raises(LLMError, match="rate limited"):
        client.complete(system="s", user="u")


def test_running_out_of_scripted_answers_is_an_error():
    """A repairer calling more often than expected must not pass quietly."""
    client = ScriptedLLMClient(["only one"])
    client.complete(system="s", user="u")

    with pytest.raises(AssertionError, match="ran out of responses"):
        client.complete(system="s", user="u")


def test_a_missing_key_is_a_clear_error_not_a_silent_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        OpenAIClient().complete(system="s", user="u")


def test_the_key_is_only_ever_read_from_the_environment():
    """There is no constructor parameter through which a key could reach the source."""
    import inspect

    parameters = inspect.signature(OpenAIClient.__init__).parameters

    assert "api_key" not in parameters
    assert set(parameters) == {"self", "model", "temperature", "max_output_tokens", "timeout"}


def test_the_defaults_bound_cost():
    client = OpenAIClient()

    assert client.model == DEFAULT_MODEL
    assert client.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_temperature_is_omitted_by_default():
    """The default model rejects any temperature but its own, so none is sent."""
    assert OpenAIClient().temperature is None
    assert OpenAIClient(temperature=0.0).temperature == 0.0
