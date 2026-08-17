"""The aggregator, on result trees built for the purpose.

Four things went wrong in the field, and each one was silent — which is why they are tested rather
than left to be noticed again:

* a remeasurement run wrote a JSON array and the loader crashed on it,
* two results for the same case were reduced to whichever file path sorted last, and the older
  measurement won,
* a result for a corruption the script did not list vanished from the tables without a word,
* an API call that returned nothing was scored as a repair that changed nothing.

The fixtures here are the smallest results that can show each of those, written to a temporary
directory. No model, no network, no reading of the real ``results/`` tree — a test that depends on
measurements on disk would start failing the day someone takes another one.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
SCRIPT = REPOSITORY / "scripts" / "aggregate_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_results"] = module
    spec.loader.exec_module(module)
    return module


agg = _load_module()


def result(
    model="llada",
    domain="domain_a",
    corruption="broken_dependency",
    *,
    solved=False,
    collateral=0,
    failures=(),
    masked=("dedupe",),
    **extra,
):
    """A result file's payload, with only the fields the aggregator reads."""
    payload = {
        "case": {"model": model, "domain": domain, "corruption": corruption},
        "failures": [{"repairer": model, "kind": kind, "detail": ""} for kind in failures],
        "masked_step_ids": list(masked),
        "solved": solved,
        "collateral_total": collateral,
        "score": {
            "errors_remaining": 0 if solved else 2,
            "collateral_modified": collateral,
            "collateral_renamed": 0,
            "collateral_removed": 0,
            "spurious_added": 0,
            "error_types_remaining": [] if solved else ["unknown_dependency"],
        },
    }
    payload.update(extra)
    return payload


def write(directory: Path, name: str, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def cell_of(cells, repairer, domain, corruption):
    return next(
        cell
        for cell in cells
        if (cell.repairer, cell.domain, cell.corruption) == (repairer, domain, corruption)
    )


# --- a file may hold one result or many ----------------------------------------------------------


def test_a_remeasurement_array_is_read_as_the_results_it_holds(tmp_path):
    write(
        tmp_path / "remeasure",
        "bd2_remeasure.json",
        [
            result(domain="domain_a", solved=True),
            result(domain="domain_b", solved=True),
        ],
    )

    load = agg.load_results(tmp_path)

    assert load.payloads == 2
    assert {measurement.domain for measurement in load.measurements} == {"domain_a", "domain_b"}
    assert [str(measurement.source).endswith("[0]") for measurement in load.measurements].count(
        True
    ) == 1


def test_a_single_result_file_still_loads(tmp_path):
    write(tmp_path, "llada__domain_a__broken_dependency.json", result())

    load = agg.load_results(tmp_path)

    cell = cell_of(agg.build_cells(load), "llada", "domain_a", "broken_dependency")

    assert load.payloads == 1
    assert load.measurements[0].source.index is None
    assert cell.outcome == agg.UNSOLVED


def test_a_file_that_is_not_json_is_reported_rather_than_ignored(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    load = agg.load_results(tmp_path)

    assert [path.name for path in load.unreadable] == ["broken.json"]


# --- two results for one case are never reduced to one -------------------------------------------


def test_two_measurements_of_one_case_are_both_kept(tmp_path):
    """The field case: an older run said unsolved, the newer run with a narrowed mask solved it."""
    write(tmp_path / "c1", "c1_remeasure.json", [result(solved=False, masked=("e_news", "dedupe"))])
    write(
        tmp_path / "bd2",
        "bd2_remeasure.json",
        [result(solved=True, masked=("dedupe",), snap=False, snap_dependencies=True)],
    )

    cells = agg.build_cells(agg.load_results(tmp_path))
    cell = cell_of(cells, "llada", "domain_a", "broken_dependency")

    assert cell.outcome == agg.AMBIGUOUS
    assert len(cell.measurements) == 2
    solved = [m for m in cell.measurements if m.payload["solved"]]
    assert len(solved) == 1
    assert solved[0].variant.snap_dependencies is True


def test_the_newer_measurement_is_visible_even_though_its_directory_sorts_first(tmp_path):
    """``bd2`` sorts before ``c1``; under the old key the older result won on that alone."""
    write(tmp_path / "bd2", "r.json", [result(solved=True, snap=False, snap_dependencies=True)])
    write(tmp_path / "c1", "r.json", [result(solved=False)])

    cells = agg.build_cells(agg.load_results(tmp_path))
    table = agg.ambiguity_table(cells)

    assert "snap=off deps=on" in table
    assert "| True |" in table  # the solved run is named in the table, not overwritten


def test_results_that_nothing_can_tell_apart_are_reported_as_such(tmp_path):
    """Same case, same recorded switches, different files: no rule here can order them."""
    write(tmp_path / "b_phase", "r.json", result(solved=False))
    write(tmp_path / "c1", "r.json", [result(solved=False)])

    load = agg.load_results(tmp_path)
    collisions = load.collisions()

    assert len(collisions) == 1
    key, group = collisions[0]
    assert key[:3] == ("llada", "domain_a", "broken_dependency")
    assert len(group) == 2
    assert "b_phase" in agg.collision_table(load) and "c1" in agg.collision_table(load)


def test_a_switch_recorded_as_off_is_not_the_same_as_a_switch_not_recorded(tmp_path):
    write(tmp_path / "before", "r.json", result())
    write(tmp_path / "after", "r.json", result(snap=False, snap_dependencies=False))

    load = agg.load_results(tmp_path)

    assert load.collisions() == []
    assert len({measurement.variant_key for measurement in load.measurements}) == 2


def test_one_measurement_per_cell_reads_exactly_as_before(tmp_path):
    write(tmp_path, "a.json", result(corruption="wrong_tool", solved=True, masked=("dedupe",)))

    cells = agg.build_cells(agg.load_results(tmp_path))
    cell = cell_of(cells, "llada", "domain_a", "wrong_tool")

    assert cell.outcome == agg.SOLVED
    assert cell.result is not None and cell.score["errors_remaining"] == 0


# --- what the tables cannot show is counted ------------------------------------------------------


def test_a_corruption_the_matrix_does_not_know_is_named_rather_than_dropped(tmp_path):
    write(tmp_path, "c3.json", [result(corruption="wrong_tool_length_matched", solved=True)])

    load = agg.load_results(tmp_path)
    cells = agg.build_cells(load)

    assert [reason for _, reason in load.unplaced()] == ["unknown corruption"]
    assert sum(len(cell.measurements) for cell in cells) == 0
    assert "wrong_tool_length_matched" in agg.accounting(load, cells)


def test_the_accounting_says_how_many_were_loaded_and_how_many_were_placed(tmp_path):
    write(tmp_path, "one.json", result())
    write(tmp_path, "two.json", [result(corruption="wrong_tool_length_matched")])

    load = agg.load_results(tmp_path)
    report = agg.accounting(load, agg.build_cells(load))

    assert "2 file(s) read, 2 result(s) in them." in report
    assert "1 placed in the matrix, 1 not placed" in report


def test_a_payload_without_a_case_is_counted(tmp_path):
    write(tmp_path, "stray.json", {"score": {}})

    load = agg.load_results(tmp_path)

    assert len(load.without_case) == 1
    assert "no case recorded" in agg.accounting(load, agg.build_cells(load))


# --- an answer that never came is not a measurement -----------------------------------------------


def test_an_empty_api_response_is_not_counted_as_a_repair(tmp_path):
    write(
        tmp_path,
        "ar.json",
        result(model="ar_local", corruption="step_deletion", failures=("api",)),
    )

    cells = agg.build_cells(agg.load_results(tmp_path))
    cell = cell_of(cells, "ar_local", "domain_a", "step_deletion")

    assert cell.outcome == agg.NOT_ANSWERED
    assert cell.outcome not in agg.MEASURED


def test_a_backend_failure_is_still_told_apart_from_a_silent_model(tmp_path):
    write(tmp_path, "b.json", result(corruption="wrong_tool", failures=("backend",)))
    write(tmp_path, "a.json", result(model="ar_local", failures=("api",)))

    cells = agg.build_cells(agg.load_results(tmp_path))

    assert cell_of(cells, "llada", "domain_a", "wrong_tool").outcome == agg.NOT_RUN
    assert cell_of(cells, "ar_local", "domain_a", "broken_dependency").outcome == agg.NOT_ANSWERED


def test_a_cell_with_no_answer_stays_out_of_the_collateral_totals(tmp_path):
    write(tmp_path, "answered.json", result(model="ar_local", collateral=1))
    write(
        tmp_path,
        "silent.json",
        result(model="ar_local", corruption="dependency_cycle", failures=("api",)),
    )

    cells = agg.build_cells(agg.load_results(tmp_path))
    summary = agg.summary_table(cells)
    row = next(line for line in summary.splitlines() if line.startswith("| ar_local "))

    assert "| 1/16 |" in row  # one measured cell, not two
    assert "| 1/0/0 |" in row


def test_the_measurement_rate_names_the_corruptions_the_model_went_silent_on(tmp_path):
    write(
        tmp_path,
        "cycle.json",
        result(model="ar_local", corruption="dependency_cycle", failures=("api",)),
    )
    write(
        tmp_path,
        "deletion.json",
        result(model="ar_local", corruption="step_deletion", failures=("api",)),
    )
    write(tmp_path, "tool.json", result(model="ar_local", corruption="wrong_tool", solved=True))

    cells = agg.build_cells(agg.load_results(tmp_path))
    report = agg.measurement_rate(cells)

    assert "ar_local       measured 1/3 of the cells with a result on disk" in report
    assert "no answer from the model: dependency_cycle/a, step_deletion/a" in report


# --- the script itself ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["single", "array"])
def test_the_script_runs_over_a_results_tree_and_prints_the_accounting(tmp_path, shape):
    payload = result(solved=True) if shape == "single" else [result(solved=True)]
    write(tmp_path / "run", "r.json", payload)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "1 file(s) read, 1 result(s) in them." in completed.stdout
    assert "## Cells with more than one measurement" in completed.stdout


def test_an_empty_tree_is_not_an_error(tmp_path):
    load = agg.load_results(tmp_path)
    cells = agg.build_cells(load)

    assert all(cell.outcome == agg.MISSING for cell in cells)
    assert "0 file(s) read" in agg.accounting(load, cells)
