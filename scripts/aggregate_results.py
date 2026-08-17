#!/usr/bin/env python
"""Read the result files and print the tables the comparison is made of.

    python scripts/aggregate_results.py --results results

Nothing here is typed by hand: every number in the write-up comes out of this script, so a claim
about the run can be checked by running it again.

**A file is not always a measurement.** Some cases never reached the model — the weights failed to
load, or the validator flagged no step so there was nothing to mask and the backend was never
called, or the model answered with nothing at all. Those are reported as their own outcomes rather
than folded in as failures, because counting a machine that ran out of memory as a repairer that
could not repair would overstate what was measured. :func:`classify` is where that line is drawn,
and an empty API response is on the not-measured side of it: a repairer that was never told
anything cannot be scored on what it left behind.

**A cell is not always one measurement.** The same case has been measured more than once — before
and after the mask was narrowed, with and without a post-processing switch. Those results carry
the switches they were run with (``snap``, ``snap_dependencies``) and are kept apart by them, but
a change to the masking code left no mark in the file, so two measurements taken generations apart
can be indistinguishable from their contents alone. This script does not guess which of those is
the one that counts. It reports every measurement it found and marks the cell as ambiguous, which
is the honest reading: an aggregate that silently keeps whichever file sorts last is not a
measurement of anything.

**Nothing is dropped in silence.** What was loaded, what was placed in the matrix, and what was
neither is counted and printed. A result for a corruption this script does not know about used to
vanish without a word; now it is named.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
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

# Outcomes a cell can have. The first three are measurements of a repairer; the rest are not, and
# are kept apart so they cannot be read as failures to repair.
SOLVED = "solved"
UNSOLVED = "unsolved"
PARSE_FAILED = "parse_failed"
NOT_RUN = "not_run"
NOT_ANSWERED = "not_answered"
NOTHING_MASKED = "nothing_masked"
MISSING = "missing"
AMBIGUOUS = "ambiguous"

MEASURED = {SOLVED, UNSOLVED, PARSE_FAILED}

# Failure kinds that mean the model never ran, or ran and said nothing. Neither is a repair.
INFRASTRUCTURE_FAILURES = {"backend", "alignment"}
NO_ANSWER_FAILURES = {"api"}

# A cell of the matrix, and a cell together with the switches a result reports about itself.
CellKey = tuple[str, str, str]
VariantKey = tuple[str, str, str, bool | None, bool | None]


@dataclass(frozen=True)
class Source:
    """Where a measurement was read from — a file, and an index if the file held several."""

    path: Path
    index: int | None = None

    def __str__(self) -> str:
        return str(self.path) if self.index is None else f"{self.path}[{self.index}]"


@dataclass(frozen=True)
class Variant:
    """The run settings a result file reports about itself.

    Only what the file says. ``None`` means the field is absent, which is not the same as off: it
    marks a result written before that switch existed, and pretending otherwise would merge
    measurements that were taken under different code.
    """

    snap: bool | None = None
    snap_dependencies: bool | None = None

    @property
    def label(self) -> str:
        return f"snap={_flag(self.snap)} deps={_flag(self.snap_dependencies)}"


def _flag(value: bool | None) -> str:
    if value is None:
        return "—"
    return "on" if value else "off"


@dataclass(frozen=True)
class Measurement:
    """One repair, as one result file recorded it."""

    source: Source
    repairer: str
    domain: str
    corruption: str
    variant: Variant
    payload: dict[str, Any]

    @property
    def cell_key(self) -> CellKey:
        return (self.repairer, self.domain, self.corruption)

    @property
    def variant_key(self) -> VariantKey:
        """What has to differ for two results to be different measurements rather than a clash."""
        return (*self.cell_key, self.variant.snap, self.variant.snap_dependencies)

    @property
    def outcome(self) -> str:
        return classify(self.payload)

    @property
    def score(self) -> dict[str, Any]:
        score: dict[str, Any] = self.payload.get("score", {})
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


@dataclass
class Load:
    """Everything the load pass saw, including what it could not use."""

    measurements: list[Measurement] = field(default_factory=list)
    files_read: int = 0
    unreadable: list[Path] = field(default_factory=list)
    payloads: int = 0
    without_case: list[Source] = field(default_factory=list)

    def unplaced(self) -> list[tuple[Measurement, str]]:
        """Measurements the matrix has no cell for, and why."""
        rejected = []
        for measurement in self.measurements:
            reasons = [
                name
                for name, value, known in (
                    ("repairer", measurement.repairer, REPAIRERS),
                    ("domain", measurement.domain, DOMAINS),
                    ("corruption", measurement.corruption, CORRUPTIONS),
                )
                if value not in known
            ]
            if reasons:
                rejected.append((measurement, ", ".join(f"unknown {name}" for name in reasons)))
        return rejected

    def collisions(self) -> list[tuple[VariantKey, list[Measurement]]]:
        """Measurements that agree on everything their files record, so nothing can order them.

        These are the ones the old three-part key overwrote in silence: same case, same switches,
        different files. Whichever sorted last won, and it was not necessarily the newer one.
        """
        grouped: dict[VariantKey, list[Measurement]] = {}
        for measurement in self.measurements:
            grouped.setdefault(measurement.variant_key, []).append(measurement)
        return [(key, group) for key, group in grouped.items() if len(group) > 1]


@dataclass(frozen=True)
class Cell:
    repairer: str
    domain: str
    corruption: str
    outcome: str
    measurements: tuple[Measurement, ...] = ()

    @property
    def measurement(self) -> Measurement | None:
        """The one measurement this cell stands on, if there is exactly one."""
        return self.measurements[0] if len(self.measurements) == 1 else None

    @property
    def result(self) -> dict[str, Any] | None:
        found = self.measurement
        return found.payload if found else None

    @property
    def score(self) -> dict[str, Any]:
        found = self.measurement
        return found.score if found else {}

    @property
    def collateral(self) -> tuple[int, int, int, int]:
        found = self.measurement
        return found.collateral if found else (0, 0, 0, 0)

    @property
    def collateral_total(self) -> int:
        modified, renamed, removed, _ = self.collateral
        return modified + renamed + removed


def load_results(root: Path) -> Load:
    """Read every result under ``root``, keeping each one whole.

    A file holds either one result or a list of them — the remeasurement runs wrote arrays, and
    their elements have the same shape a single run writes. Both are flattened to measurements
    here, so no branch downstream has to know which file it came from.
    """
    load = Load()
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            load.unreadable.append(path)
            continue
        load.files_read += 1
        entries = payload if isinstance(payload, list) else [payload]
        for index, entry in enumerate(entries):
            source = Source(path, index if isinstance(payload, list) else None)
            load.payloads += 1
            if not isinstance(entry, dict):
                load.without_case.append(source)
                continue
            case = entry.get("case")
            if not case:
                load.without_case.append(source)
                continue
            load.measurements.append(
                Measurement(
                    source=source,
                    repairer=case["model"],
                    domain=case["domain"],
                    corruption=case["corruption"],
                    variant=Variant(entry.get("snap"), entry.get("snap_dependencies")),
                    payload=entry,
                )
            )
    return load


def classify(result: dict[str, Any] | None) -> str:
    """What this cell actually tells us.

    A backend failure means the model never ran — that is a fact about the machine, not about the
    repairer. An empty API response means it ran and returned nothing, which is a fact about the
    call: there is no repair in it to score, and counting its untouched plan as "no collateral"
    would credit a repairer for damage it had no opportunity to do. An empty mask means the
    validator named no step, so a step-level repairer had nothing to work on and was never called.
    """
    if result is None:
        return MISSING
    kinds = {failure["kind"] for failure in result.get("failures", [])}
    if INFRASTRUCTURE_FAILURES & kinds:
        return NOT_RUN
    if NO_ANSWER_FAILURES & kinds:
        return NOT_ANSWERED
    if result.get("solved"):
        return SOLVED
    diagnostics = result.get("diagnostics")
    if diagnostics is not None and not diagnostics.get("parsed", True):
        return PARSE_FAILED
    if result["case"]["model"] in ("llada", "dream") and not result.get("masked_step_ids"):
        return NOTHING_MASKED
    return UNSOLVED


def build_cells(load: Load) -> list[Cell]:
    """One cell per matrix position, carrying every measurement that claims it.

    A cell with more than one measurement is reported as ambiguous rather than resolved. The files
    do not record when they were written or which version of the masking code produced them, so
    any rule for picking a winner here would be invented by this script — and the rule that was in
    place before, "whichever path sorts last", was inventing one badly.
    """
    by_cell: dict[CellKey, list[Measurement]] = {}
    for measurement in load.measurements:
        by_cell.setdefault(measurement.cell_key, []).append(measurement)

    cells = []
    for repairer in REPAIRERS:
        for domain in DOMAINS:
            for corruption in CORRUPTIONS:
                found = by_cell.get((repairer, domain, corruption), [])
                if not found:
                    outcome = MISSING
                elif len(found) > 1:
                    outcome = AMBIGUOUS
                else:
                    outcome = classify(found[0].payload)
                cells.append(Cell(repairer, domain, corruption, outcome, tuple(found)))
    return cells


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
    NOT_ANSWERED: "N/A no answer",
    NOTHING_MASKED: "no mask",
    MISSING: "N/A missing",
    AMBIGUOUS: "ambiguous",
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
                elif cell.outcome == AMBIGUOUS:
                    mark = f"{mark} ({len(cell.measurements)} measurements)"
                row.append(mark)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summary_table(cells: list[Cell]) -> str:
    lines = [
        "| repairer | measured | solved | unsolved | parse-fail | N/A load | N/A no answer "
        "| no mask | missing | ambiguous | collateral mod/ren/rem | added |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
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
            f"| {counts[PARSE_FAILED]} | {counts[NOT_RUN]} | {counts[NOT_ANSWERED]} "
            f"| {counts[NOTHING_MASKED]} | {counts[MISSING]} | {counts[AMBIGUOUS]} "
            f"| {modified}/{renamed}/{removed} | {added} |"
        )
    return "\n".join(lines)


def ambiguity_table(cells: list[Cell]) -> str:
    """Every measurement of a cell that has more than one.

    This is what the old key hid. The table is the point of the fix: a reader can see that the
    narrowed-mask run of a case solved it even though an older run of the same case did not.
    """
    lines = [
        "| repairer | domain | corruption | variant | outcome | solved | coll | masked steps "
        "| source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        if cell.outcome != AMBIGUOUS:
            continue
        for measurement in cell.measurements:
            masked = measurement.payload.get("masked_step_ids") or []
            lines.append(
                f"| {cell.repairer} | {cell.domain[-1]} | {cell.corruption} "
                f"| {measurement.variant.label} | {MARKS[measurement.outcome]} "
                f"| {bool(measurement.payload.get('solved'))} | {measurement.collateral_total} "
                f"| {len(masked)} | {measurement.source} |"
            )
    return "\n".join(lines) if len(lines) > 2 else "(every cell has at most one measurement)"


def collision_table(load: Load) -> str:
    """Results nothing in their own contents can tell apart. The old key overwrote these."""
    collisions = load.collisions()
    if not collisions:
        return "(no two results claim the same case with the same recorded settings)"
    lines = ["| repairer | domain | corruption | variant | sources |", "|---|---|---|---|---|"]
    for key, group in sorted(collisions, key=lambda item: str(item[0])):
        repairer, domain, corruption = key[0], key[1], key[2]
        sources = ", ".join(str(measurement.source) for measurement in group)
        lines.append(
            f"| {repairer} | {domain[-1]} | {corruption} | {group[0].variant.label} | {sources} |"
        )
    return "\n".join(lines)


def measurement_rate(cells: list[Cell]) -> str:
    """How much of the matrix each repairer actually answered, and where it did not.

    Reported per repairer *and* per corruption because the gaps are not spread evenly: an API that
    returns nothing does so on the cases with the most to write, so a missing cell is evidence
    about which corruptions are hard, not a random hole.
    """
    lines = []
    for repairer in REPAIRERS:
        own = [cell for cell in cells if cell.repairer == repairer]
        present = [cell for cell in own if cell.outcome != MISSING]
        measured = [cell for cell in own if cell.outcome in MEASURED]
        share = f"{len(measured)}/{len(present)}" if present else "0/0"
        lines.append(f"  {repairer:<14} measured {share} of the cells with a result on disk")
        for outcome, what in ((NOT_ANSWERED, "no answer from the model"), (NOT_RUN, "never ran")):
            gaps = sorted(
                f"{cell.corruption}/{cell.domain[-1]}" for cell in own if cell.outcome == outcome
            )
            if gaps:
                lines.append(f"      {what}: {', '.join(gaps)}")
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


def accounting(load: Load, cells: list[Cell]) -> str:
    """Where every loaded result ended up. A result that reaches no table is named here."""
    placed = sum(len(cell.measurements) for cell in cells)
    unplaced = load.unplaced()
    collisions = load.collisions()
    lines = [
        f"{load.files_read} file(s) read, {load.payloads} result(s) in them.",
        f"{placed} placed in the matrix, {len(unplaced)} not placed, "
        f"{len(load.without_case)} without a case to place them by.",
        f"{sum(len(group) for _, group in collisions)} result(s) in "
        f"{len(collisions)} group(s) that nothing in their contents can tell apart.",
    ]
    if load.unreadable:
        lines.append(f"unreadable file(s): {', '.join(str(path) for path in load.unreadable)}")
    for measurement, reason in unplaced:
        lines.append(
            f"  not placed ({reason}): {measurement.repairer}/{measurement.domain}"
            f"/{measurement.corruption} from {measurement.source}"
        )
    for source in load.without_case:
        lines.append(f"  no case recorded: {source}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    load = load_results(arguments.results)
    cells = build_cells(load)

    print(f"# Aggregate over {arguments.results}\n")
    print(accounting(load, cells))
    print(f"\n{len(cells)} cells in the matrix.\n")
    print("## Matrix\n")
    print(matrix_table(cells))
    print("\n## Cells with more than one measurement\n")
    print(ambiguity_table(cells))
    print("\n## Results that cannot be told apart\n")
    print(collision_table(load))
    print("\n## By repairer\n")
    print(summary_table(cells))
    print("\n## How much was measured\n")
    print("```")
    print(measurement_rate(cells))
    print("```")
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
