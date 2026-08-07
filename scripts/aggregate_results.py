#!/usr/bin/env python
"""Read the result files and print the tables the comparison is made of.

    python scripts/aggregate_results.py --results results

Nothing here is typed by hand: every number in the write-up comes out of this script, so a claim
about the run can be checked by running it again.

**A file is not always a measurement.** Some cases never reached the model — the weights failed to
load, or the validator flagged no step so there was nothing to mask and the backend was never
called. Those are reported as their own outcomes rather than folded in as failures, because
counting a machine that ran out of memory as a repairer that could not repair would overstate
what was measured. :func:`classify` is where that line is drawn.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPAIRERS = ["deterministic", "ar_full", "ar_local", "llada", "dream"]
CORRUPTIONS = [
    "broken_dependency",
    "dependency_cycle",
    "wrong_tool",
    "duplicate_step",
    "step_deletion",
    "wrong_ordering",
    "missing_stop_condition",
    "drop_required_step",
]
DOMAINS = ["domain_a", "domain_b"]

# Outcomes a cell can have. The first three are measurements of a repairer; the last three are
# not, and are kept apart so they cannot be read as failures to repair.
SOLVED = "solved"
UNSOLVED = "unsolved"
PARSE_FAILED = "parse_failed"
NOT_RUN = "not_run"
NOTHING_MASKED = "nothing_masked"
MISSING = "missing"

MEASURED = {SOLVED, UNSOLVED, PARSE_FAILED}


@dataclass(frozen=True)
class Cell:
    repairer: str
    domain: str
    corruption: str
    outcome: str
    result: dict[str, Any] | None = None

    @property
    def score(self) -> dict[str, Any]:
        score: dict[str, Any] = (self.result or {}).get("score", {})
        return score

    @property
    def collateral(self) -> tuple[int, int, int, int]:
        score = self.score
        return (
            score.get("collateral_modified", 0),
            score.get("collateral_renamed", 0),
            score.get("collateral_removed", 0),
            score.get("spurious_added", 0),
        )

    @property
    def collateral_total(self) -> int:
        modified, renamed, removed, _ = self.collateral
        return modified + renamed + removed


def load_results(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        case = payload.get("case")
        if not case:
            continue
        results[(case["model"], case["domain"], case["corruption"])] = payload
    return results


def classify(result: dict[str, Any] | None) -> str:
    """What this cell actually tells us.

    A backend failure means the model never ran — that is a fact about the machine, not about the
    repairer. An empty mask means the validator named no step, so a step-level repairer had
    nothing to work on and was never called.
    """
    if result is None:
        return MISSING
    kinds = {failure["kind"] for failure in result.get("failures", [])}
    if {"backend", "alignment"} & kinds:
        return NOT_RUN
    if result.get("solved"):
        return SOLVED
    diagnostics = result.get("diagnostics")
    if diagnostics is not None and not diagnostics.get("parsed", True):
        return PARSE_FAILED
    if result["case"]["model"] in ("llada", "dream") and not result.get("masked_step_ids"):
        return NOTHING_MASKED
    return UNSOLVED


def build_cells(results: dict[tuple[str, str, str], dict[str, Any]]) -> list[Cell]:
    return [
        Cell(
            repairer,
            domain,
            corruption,
            classify(results.get((repairer, domain, corruption))),
            results.get((repairer, domain, corruption)),
        )
        for repairer in REPAIRERS
        for domain in DOMAINS
        for corruption in CORRUPTIONS
    ]


def parse_failure_cause(result: dict[str, Any]) -> str:
    """Why the text did not parse, as far as the recorded output can say."""
    diagnostics = result.get("diagnostics") or {}
    raw = diagnostics.get("raw_text", "")
    if diagnostics.get("mask_tokens_remaining", 0) > 0:
        return "denoising unfinished"
    failure = diagnostics.get("parse_failure") or {}
    excerpt = failure.get("excerpt", "")
    if _looks_degenerate(excerpt) or _looks_degenerate(raw[-400:]):
        return "degeneration"
    if "schema" in failure.get("message", ""):
        return "schema"
    return "format/boundary"


def _looks_degenerate(text: str) -> bool:
    """A run of the same short token repeated is how these models come apart."""
    words = text.split()
    if len(words) < 6:
        return False
    counts = Counter(words)
    word, count = counts.most_common(1)[0]
    return count >= 4 and len(word) <= 12


# --- tables --------------------------------------------------------------------------------------

MARKS = {
    SOLVED: "solved",
    UNSOLVED: "unsolved",
    PARSE_FAILED: "parse-fail",
    NOT_RUN: "N/A load",
    NOTHING_MASKED: "no mask",
    MISSING: "N/A missing",
}


def matrix_table(cells: list[Cell]) -> str:
    by_key = {(cell.repairer, cell.domain, cell.corruption): cell for cell in cells}
    lines = [
        "| corruption | domain | " + " | ".join(REPAIRERS) + " |",
        "|---|---|" + "---|" * len(REPAIRERS),
    ]
    for corruption in CORRUPTIONS:
        for domain in DOMAINS:
            row = [corruption, domain[-1]]
            for repairer in REPAIRERS:
                cell = by_key[(repairer, domain, corruption)]
                mark = MARKS[cell.outcome]
                if cell.outcome in MEASURED:
                    errors = cell.score.get("errors_remaining", "?")
                    mark = f"{mark} ({errors}e, c={cell.collateral_total})"
                row.append(mark)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summary_table(cells: list[Cell]) -> str:
    lines = [
        "| repairer | measured | solved | unsolved | parse-fail | N/A load | no mask | missing "
        "| collateral mod/ren/rem | added |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for repairer in REPAIRERS:
        own = [cell for cell in cells if cell.repairer == repairer]
        counts = Counter(cell.outcome for cell in own)
        measured = [cell for cell in own if cell.outcome in MEASURED]
        modified = sum(cell.collateral[0] for cell in measured)
        renamed = sum(cell.collateral[1] for cell in measured)
        removed = sum(cell.collateral[2] for cell in measured)
        added = sum(cell.collateral[3] for cell in measured)
        lines.append(
            f"| {repairer} | {len(measured)}/{len(own)} | {counts[SOLVED]} | {counts[UNSOLVED]} "
            f"| {counts[PARSE_FAILED]} | {counts[NOT_RUN]} | {counts[NOTHING_MASKED]} "
            f"| {counts[MISSING]} | {modified}/{renamed}/{removed} | {added} |"
        )
    return "\n".join(lines)


def remaining_error_types(cells: list[Cell]) -> str:
    lines = ["| repairer | error types left on unsolved cases (count) |", "|---|---|"]
    for repairer in REPAIRERS:
        counter: Counter[str] = Counter()
        for cell in cells:
            if cell.repairer != repairer or cell.outcome not in MEASURED:
                continue
            counter.update(cell.score.get("error_types_remaining", []))
        listing = ", ".join(f"{name} ({count})" for name, count in counter.most_common()) or "—"
        lines.append(f"| {repairer} | {listing} |")
    return "\n".join(lines)


def parse_failure_table(cells: list[Cell]) -> str:
    lines = [
        "| repairer | domain | corruption | cause | mask tokens left |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        if cell.outcome != PARSE_FAILED or cell.result is None:
            continue
        diagnostics = cell.result.get("diagnostics") or {}
        lines.append(
            f"| {cell.repairer} | {cell.domain[-1]} | {cell.corruption} "
            f"| {parse_failure_cause(cell.result)} "
            f"| {diagnostics.get('mask_tokens_remaining', '?')} |"
        )
    return "\n".join(lines) if len(lines) > 2 else "(no parse failures)"


def collateral_comparison(cells: list[Cell]) -> str:
    """Where the corruption was the same and more than one repairer was measured."""
    lines = ["| domain | corruption | " + " | ".join(REPAIRERS) + " |", "|---|---|" + "---|" * 5]
    by_key = {(cell.repairer, cell.domain, cell.corruption): cell for cell in cells}
    for corruption in CORRUPTIONS:
        for domain in DOMAINS:
            row = [domain[-1], corruption]
            for repairer in REPAIRERS:
                cell = by_key[(repairer, domain, corruption)]
                if cell.outcome not in MEASURED:
                    row.append("—")
                    continue
                modified, renamed, removed, added = cell.collateral
                row.append(f"{modified}/{renamed}/{removed} (+{added})")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cross_analysis(cells: list[Cell]) -> str:
    measured = [cell for cell in cells if cell.outcome in MEASURED]
    lines = []

    solved = Counter(cell.repairer for cell in cells if cell.outcome == SOLVED)
    lines.append("solved, out of cases actually measured:")
    for repairer in REPAIRERS:
        own = [cell for cell in measured if cell.repairer == repairer]
        lines.append(f"  {repairer:<14} {solved[repairer]}/{len(own)}")

    lines.append("")
    lines.append("collateral (healthy steps disturbed), summed over measured cases:")
    for repairer in REPAIRERS:
        own = [cell for cell in measured if cell.repairer == repairer]
        if not own:
            lines.append(f"  {repairer:<14} — (nothing measured)")
            continue
        total = sum(cell.collateral_total for cell in own)
        lines.append(
            f"  {repairer:<14} {total:>3} over {len(own)} case(s)   (mean {total / len(own):.2f})"
        )

    lines.append("")
    lines.append("renamed, the category predicted but never yet observed:")
    renamed = [cell for cell in measured if cell.collateral[1] > 0]
    lines.append(
        "  " + (", ".join(f"{c.repairer}/{c.domain}/{c.corruption}" for c in renamed) or "none")
    )

    lines.append("")
    lines.append("missing_operation left behind (the cost of regenerating a whole step):")
    for repairer in REPAIRERS:
        hits = [
            cell
            for cell in measured
            if cell.repairer == repairer
            and "missing_operation" in cell.score.get("error_types_remaining", [])
        ]
        lines.append(f"  {repairer:<14} {len(hits)} case(s)")

    lines.append("")
    lines.append("diffusion parsing:")
    for repairer in ("llada", "dream"):
        own = [cell for cell in cells if cell.repairer == repairer]
        reached = [cell for cell in own if cell.outcome in MEASURED]
        parsed = [cell for cell in reached if cell.outcome != PARSE_FAILED]
        share = f"{len(parsed)}/{len(reached)}" if reached else "0/0"
        lines.append(f"  {repairer:<14} parsed {share} of the cases that reached the model")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    results = load_results(arguments.results)
    cells = build_cells(results)

    print(f"# Aggregate over {arguments.results}\n")
    print(f"{len(results)} result file(s); {len(cells)} cells in the matrix.\n")
    print("## Matrix\n")
    print(matrix_table(cells))
    print("\n## By repairer\n")
    print(summary_table(cells))
    print("\n## Errors left behind\n")
    print(remaining_error_types(cells))
    print("\n## Parse failures\n")
    print(parse_failure_table(cells))
    print("\n## Collateral, cell by cell (modified/renamed/removed (+added))\n")
    print(collateral_comparison(cells))
    print("\n## Cross-analysis\n")
    print("```")
    print(cross_analysis(cells))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
