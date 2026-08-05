"""The experiment runner, driven end to end on a mock backend.

The runner is what will be left alone for hours on a rented GPU, so the two things that decide
whether that is survivable are checked here: a case that finishes is written down immediately, and
a rerun does only what is missing. Both are exercised by actually running the script, because a
resume that works in theory is not a resume.

The backend is a mock throughout — no model, no GPU, no network.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
SCRIPT = REPOSITORY / "scripts" / "run_diffusion_experiment.py"


def run(*arguments, out: Path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "oracle", "--out", str(out), *arguments],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def results_in(out: Path):
    return sorted(path.name for path in out.glob("*.json"))


def test_the_script_runs_a_case_and_writes_it_down(tmp_path):
    run("--model", "llada", "--domain", "domain_b", "--corruption", "wrong_tool", out=tmp_path)

    assert results_in(tmp_path) == ["llada__domain_b__wrong_tool.json"]


def test_a_result_carries_what_a_measurement_needs(tmp_path):
    run(
        "--model",
        "dream",
        "--domain",
        "domain_a",
        "--corruption",
        "broken_dependency",
        out=tmp_path,
    )

    result = json.loads((tmp_path / "dream__domain_a__broken_dependency.json").read_text())

    assert result["case"] == {
        "model": "dream",
        "domain": "domain_a",
        "corruption": "broken_dependency",
    }
    assert result["backend"]["backend"] == "oracle"
    assert result["damaged_step_ids"] == ["dedupe"]
    assert "dedupe" in result["masked_step_ids"]
    assert result["solved"] is True
    assert result["collateral_total"] == 0
    assert result["failures"] == []
    assert set(result["score"]) >= {
        "collateral_modified",
        "collateral_renamed",
        "collateral_removed",
    }
    assert result["repaired_plan"]["steps"]
    assert result["elapsed_seconds"] >= 0


def test_the_full_matrix_is_two_models_by_two_domains_by_every_corruption(tmp_path):
    run(out=tmp_path)

    names = results_in(tmp_path)

    assert len(names) == 32
    assert len({name.split("__")[0] for name in names}) == 2
    assert len({name.split("__")[1] for name in names}) == 2
    assert len({name.split("__")[2] for name in names}) == 8


# --- resume ---------------------------------------------------------------------------------------


def test_a_finished_case_is_not_run_again(tmp_path):
    run("--model", "llada", "--domain", "domain_b", out=tmp_path)
    marker = tmp_path / "llada__domain_b__wrong_tool.json"
    marker.write_text('{"sentinel": true}', encoding="utf-8")

    second = run("--model", "llada", "--domain", "domain_b", out=tmp_path)

    assert json.loads(marker.read_text()) == {"sentinel": True}
    assert "8 already done, 0 to run" in second.stdout


def test_an_interrupted_run_continues_where_it_stopped(tmp_path):
    run(out=tmp_path)
    for path in sorted(tmp_path.glob("*.json"))[:20]:
        path.unlink()

    resumed = run(out=tmp_path)

    assert "32 case(s): 12 already done, 20 to run" in resumed.stdout
    assert len(results_in(tmp_path)) == 32


def test_the_progress_bar_starts_from_what_is_already_done(tmp_path):
    """A bar that restarts at zero after a crash misreports how much is left."""
    run(out=tmp_path)
    for path in sorted(tmp_path.glob("*.json"))[:4]:
        path.unlink()

    resumed = run(out=tmp_path)

    assert "28/32" in resumed.stderr  # the bar opens at 28 of 32, not 0 of 4
    assert "32/32" in resumed.stderr


def test_force_redoes_everything(tmp_path):
    run("--model", "llada", "--domain", "domain_b", out=tmp_path)
    marker = tmp_path / "llada__domain_b__wrong_tool.json"
    marker.write_text('{"sentinel": true}', encoding="utf-8")

    run("--model", "llada", "--domain", "domain_b", "--force", out=tmp_path)

    assert json.loads(marker.read_text()) != {"sentinel": True}


def test_limit_stops_after_the_given_number_of_new_cases(tmp_path):
    run("--limit", "3", out=tmp_path)

    assert len(results_in(tmp_path)) == 3


# --- the case matrix, without running anything ---------------------------------------------------


@pytest.fixture
def runner():
    sys.path.insert(0, str(REPOSITORY / "scripts"))
    import run_diffusion_experiment

    return run_diffusion_experiment


def test_every_corruption_is_reachable_in_both_domains(runner):
    """The targets are picked from the graph, so one matrix covers two unrelated pipelines."""
    from plan_repair.data import load_reference

    for domain in runner.DOMAINS.values():
        task, plan = load_reference(domain)
        for name, corrupt in runner.CORRUPTIONS.items():
            corruption = corrupt(task, plan)
            assert corruption.broken_plan != plan, f"{domain}/{name} changed nothing"


def test_the_case_list_is_the_product_of_the_choices(runner):
    cases = runner.enumerate_cases(["llada"], ["domain_a", "domain_b"], ["wrong_tool"])

    assert [case.key for case in cases] == [
        "llada__domain_a__wrong_tool",
        "llada__domain_b__wrong_tool",
    ]


def test_a_mock_backend_needs_no_tokenizer(runner):
    """Which is what lets the whole flow be checked off-GPU."""
    assert runner.build_tokenizer("llada", "oracle") is None
    assert runner.build_tokenizer("dream", "echo") is None
