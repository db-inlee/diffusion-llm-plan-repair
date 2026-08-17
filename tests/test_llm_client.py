"""The client seam: what it promises, and what it refuses to do without a key.

The second half of this file is about an answer that came back empty. That happened on six of
sixteen ``ar_local`` cases and the call itself succeeded — HTTP 200, a response object, an empty
string in it — so the only way to tell "the model ran out of budget" from "the model refused" is
to read the finish reason. These tests feed the client responses shaped like the ones the SDK
returns, including responses missing the accounting fields, because a diagnosis that raises while
explaining an empty answer is worse than none.
"""

from types import SimpleNamespace

import pytest

from plan_repair.repair import LLMClient, LLMError, OpenAIClient, ScriptedLLMClient
from plan_repair.repair.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS_CEILING,
    MAX_OUTPUT_TOKENS_VARIABLE,
    LLMCall,
    budget_from_environment,
)


def response(
    content="{}",
    *,
    finish_reason="stop",
    prompt_tokens=1200,
    completion_tokens=900,
    reasoning_tokens=None,
    refusal=None,
    usage=True,
):
    """A response shaped like the SDK's, with the parts a diagnosis reads."""
    details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
        usage=(
            SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                completion_tokens_details=details,
            )
            if usage
            else None
        ),
    )


class RecordingAPI:
    """Stands in for the SDK client, keeping the request it was handed."""

    def __init__(self, answer):
        self._answer = answer
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self._answer


def client_returning(answer, **arguments):
    client = OpenAIClient(**arguments)
    api = RecordingAPI(answer)
    client._client = api  # the key check is exercised on its own below
    return client, api


@pytest.fixture(autouse=True)
def _no_budget_in_the_environment(monkeypatch):
    """The default is what the code says, not what the shell that ran the tests says."""
    monkeypatch.delenv(MAX_OUTPUT_TOKENS_VARIABLE, raising=False)


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


# --- why an answer came back empty ----------------------------------------------------------------


def test_an_empty_answer_says_the_budget_ran_out_when_that_is_what_happened():
    client, _ = client_returning(
        response("", finish_reason="length", completion_tokens=4096, reasoning_tokens=4096)
    )

    with pytest.raises(LLMError) as raised:
        client.complete(system="s", user="u")

    message = str(raised.value)
    assert "finish_reason=length" in message
    assert "4096 completion token(s), 4096 of them reasoning" in message
    assert "budget 4096" in message
    assert "the budget ran out before an answer was written" in message


def test_an_empty_answer_for_another_reason_does_not_blame_the_budget():
    client, _ = client_returning(
        response("", finish_reason="content_filter", completion_tokens=12, refusal="I cannot")
    )

    with pytest.raises(LLMError) as raised:
        client.complete(system="s", user="u")

    message = str(raised.value)
    assert "finish_reason=content_filter" in message
    assert "refusal: I cannot" in message
    assert "budget ran out" not in message


def test_whitespace_is_as_empty_as_nothing_and_is_diagnosed_the_same_way():
    client, _ = client_returning(response("   \n ", finish_reason="length"))

    with pytest.raises(LLMError, match="finish_reason=length"):
        client.complete(system="s", user="u")


def test_a_response_without_accounting_is_still_diagnosed():
    """A shape missing usage must not turn the diagnosis into a second failure."""
    client, _ = client_returning(response("", finish_reason=None, usage=False))

    with pytest.raises(LLMError) as raised:
        client.complete(system="s", user="u")

    assert "finish_reason=unreported" in str(raised.value)
    assert client.last_call is not None
    assert client.last_call.completion_tokens is None


def test_a_call_that_answered_is_recorded_too():
    """Sizing a budget needs the successful calls, not only the truncated ones."""
    client, _ = client_returning(
        response("{}", completion_tokens=1500, reasoning_tokens=600), max_output_tokens=4096
    )

    assert client.complete(system="s", user="u") == "{}"
    assert client.last_call == LLMCall(
        model=DEFAULT_MODEL,
        budget=4096,
        finish_reason="stop",
        prompt_tokens=1200,
        completion_tokens=1500,
        reasoning_tokens=600,
        refusal=None,
        content_characters=2,
    )
    assert client.last_call.out_of_budget is False


def test_a_failed_call_leaves_the_previous_record_alone_rather_than_inventing_one():
    client, api = client_returning(response("{}"))
    client.complete(system="s", user="u")

    def explode(**_request):
        raise RuntimeError("connection reset")

    api.chat.completions.create = explode
    with pytest.raises(LLMError, match="call failed"):
        client.complete(system="s", user="u")

    assert client.last_call is not None and client.last_call.finish_reason == "stop"


# --- the completion budget ------------------------------------------------------------------------


def test_the_budget_reaches_the_request_as_the_parameter_this_model_takes():
    client, api = client_returning(response(), max_output_tokens=16384)

    client.complete(system="s", user="u")

    assert api.requests[0]["max_completion_tokens"] == 16384
    assert "max_tokens" not in api.requests[0]
    assert "temperature" not in api.requests[0]


def test_the_budget_can_be_raised_for_a_run_without_editing_the_source(monkeypatch):
    monkeypatch.setenv(MAX_OUTPUT_TOKENS_VARIABLE, "16384")

    assert budget_from_environment() == 16384
    assert OpenAIClient().max_output_tokens == 16384


def test_an_explicit_budget_still_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(MAX_OUTPUT_TOKENS_VARIABLE, "16384")

    assert OpenAIClient(max_output_tokens=2048).max_output_tokens == 2048


@pytest.mark.parametrize("value", ["", "   "])
def test_an_unset_budget_is_the_default(monkeypatch, value):
    monkeypatch.setenv(MAX_OUTPUT_TOKENS_VARIABLE, value)

    assert budget_from_environment() == DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("value", ["lots", "0", "-1"])
def test_a_budget_that_cannot_be_read_is_refused_rather_than_ignored(monkeypatch, value):
    """Silently falling back would make the next run's empty answers unexplainable again."""
    monkeypatch.setenv(MAX_OUTPUT_TOKENS_VARIABLE, value)

    with pytest.raises(LLMError, match="must be a positive integer"):
        budget_from_environment()


def test_the_budget_has_a_ceiling_because_a_reasoning_model_bills_what_it_thinks(monkeypatch):
    monkeypatch.setenv(MAX_OUTPUT_TOKENS_VARIABLE, str(MAX_OUTPUT_TOKENS_CEILING + 1))

    with pytest.raises(LLMError, match="above the ceiling"):
        budget_from_environment()


def test_the_diagnosis_reaches_the_failure_the_runner_writes_down():
    """The point of the record: a result file says why the cell has no repair in it.

    The repairer turns an :class:`LLMError` into a recorded failure whose detail is the error's
    message, and the runner writes that into the result file. Nothing in the repair path changed
    for this — the message it was already copying now has the reason in it.
    """
    from plan_repair.corruption import UNKNOWN_MODE, inject_broken_dependency
    from plan_repair.data import DATA_PIPELINE_A, load_reference
    from plan_repair.repair import API_FAILURE, ARLocalRepairer
    from plan_repair.validation import validate_plan

    task, plan = load_reference(DATA_PIPELINE_A)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    broken = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE).broken_plan
    client, _ = client_returning(
        response("", finish_reason="length", completion_tokens=4096, reasoning_tokens=4096)
    )

    repairer = ARLocalRepairer(client)
    repairer.repair(broken, validate_plan(broken, task), task)

    assert [failure.kind for failure in repairer.failures] == [API_FAILURE]
    assert "finish_reason=length" in repairer.failures[0].detail
