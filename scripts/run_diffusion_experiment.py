#!/usr/bin/env python
"""Run the diffusion repairers over the corruption matrix and score what comes back.

Written to be run on a GPU machine after cloning the repository:

    pip install -e '.[gpu]'
    python scripts/run_diffusion_experiment.py --model llada --out results/llada

and to be interruptible. Each case writes its own result file the moment it finishes, and a rerun
skips whatever is already on disk — so a session that dies halfway costs only the case it was in
the middle of. The progress bar counts the skipped cases as done rather than restarting at zero,
because a bar that lies about how much is left is worse than none.

Locally, where there is no GPU, the same script runs end to end against the mock backends:

    python scripts/run_diffusion_experiment.py --backend oracle --out /tmp/dry

which is how the flow is checked without a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:  # so a clone runs without installing
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from plan_repair.corruption import (  # noqa: E402
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_drop_required_step,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference  # noqa: E402
from plan_repair.repair import (  # noqa: E402
    ARFullRepairer,
    ARLocalRepairer,
    ByteOffsetTokenizer,
    DeterministicRepairer,
    EchoBackend,
    HuggingFaceTokenizer,
    OpenAIClient,
    OracleBackend,
    TorchDLLMBackend,
    repair_and_score,
)
from plan_repair.repair.diagnostics import summarise  # noqa: E402
from plan_repair.repair.diffusion import (  # noqa: E402
    DREAM_MASK_TOKEN_ID,
    DREAM_MODEL,
    LLADA_MASK_TOKEN_ID,
    LLADA_MODEL,
    DreamRepairer,
    LLaDARepairer,
)
from plan_repair.schema import AgentPlan, AgentTask, CorruptionResult  # noqa: E402

MODELS = {
    "llada": (LLaDARepairer, LLADA_MODEL, LLADA_MASK_TOKEN_ID, "fast"),
    "dream": (DreamRepairer, DREAM_MODEL, DREAM_MASK_TOKEN_ID, "byte"),
}
# Repairers without a diffusion backend, run through the same matrix for comparability.
BASELINES = ("deterministic", "ar_full", "ar_local")
DOMAINS = {"domain_a": DATA_PIPELINE_A, "domain_b": DATA_PIPELINE_B}

# The targets are chosen structurally, so the same matrix applies to both pipelines.
CORRUPTIONS: dict[str, Callable[[AgentTask, AgentPlan], CorruptionResult]] = {
    "broken_dependency": lambda task, plan: inject_broken_dependency(
        plan, step_id=_fan_in(plan), mode=UNKNOWN_MODE
    ),
    "dependency_cycle": lambda task, plan: inject_broken_dependency(
        plan, step_id=_first_source(plan), mode=CYCLE_MODE, cycle_with=plan.steps[-1].id
    ),
    "wrong_tool": lambda task, plan: inject_wrong_tool(plan, step_id=_fan_in(plan)),
    "duplicate_step": lambda task, plan: inject_duplicate_step(plan, step_id=_chain(plan)),
    "step_deletion": lambda task, plan: inject_step_deletion(plan, step_id=_fan_in(plan)),
    "wrong_ordering": lambda task, plan: inject_wrong_ordering(plan, step_id=_fan_in(plan)),
    "missing_stop_condition": lambda task, plan: inject_missing_stop_condition(plan),
    "drop_required_step": lambda task, plan: inject_drop_required_step(
        plan, task, requirement=task.required_evidence[0]
    ),
}


def _fan_in(plan: AgentPlan) -> str:
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def _chain(plan: AgentPlan) -> str:
    return next(step.id for step in plan.steps if len(step.input_from) == 1)


def _first_source(plan: AgentPlan) -> str:
    consumed = {dep for step in plan.steps for dep in step.input_from}
    return next(step.id for step in plan.steps if not step.input_from and step.id in consumed)


@dataclass(frozen=True)
class Case:
    model: str
    domain: str
    corruption: str

    @property
    def key(self) -> str:
        return f"{self.model}__{self.domain}__{self.corruption}"

    def __str__(self) -> str:
        return f"{self.model}/{self.domain}/{self.corruption}"


def enumerate_cases(models: list[str], domains: list[str], corruptions: list[str]) -> list[Case]:
    return [
        Case(model, domain, corruption)
        for model in models
        for domain in domains
        for corruption in corruptions
    ]


def build_backend(
    kind: str, model: str, reference: AgentPlan, steps: int, temperature: float
) -> Any:
    """The backend under test, or a mock standing in for it."""
    if kind == "oracle":
        return OracleBackend(reference)
    if kind == "echo":
        return EchoBackend()
    _, model_id, mask_id, _ = MODELS[model]
    return TorchDLLMBackend(model_id, mask_id, name=model, steps=steps, temperature=temperature)


def build_tokenizer(model: str, backend_kind: str) -> Any:
    """LLaDA reports offsets itself; Dream's are reconstructed from its byte vocabulary."""
    if backend_kind != "torch":
        return None
    from transformers import AutoTokenizer

    _, model_id, _, offsets = MODELS[model]
    load: Any = AutoTokenizer.from_pretrained
    raw = load(model_id, trust_remote_code=True)
    return (
        HuggingFaceTokenizer(raw, model) if offsets == "fast" else ByteOffsetTokenizer(raw, model)
    )


def build_repairer(
    case: Case, backend_kind: str, plan: AgentPlan, steps: int, temperature: float
) -> tuple[Any, Any]:
    """The repairer for this case, and the backend behind it if it has one.

    The rule-based and autoregressive repairers are here so that every repairer in the comparison
    writes the same result shape from the same corruption matrix — a table assembled from files
    produced by different scripts would be a table of differences between scripts.
    """
    if case.model == "deterministic":
        return DeterministicRepairer(), None
    if case.model in ("ar_full", "ar_local"):
        client = OpenAIClient()
        builder = ARFullRepairer if case.model == "ar_full" else ARLocalRepairer
        return builder(client), None
    backend = build_backend(backend_kind, case.model, plan, steps, temperature)
    return MODELS[case.model][0](backend, build_tokenizer(case.model, backend_kind)), backend


def run_case(case: Case, backend_kind: str, steps: int, temperature: float) -> dict[str, Any]:
    task, plan = load_reference(DOMAINS[case.domain])
    corruption = CORRUPTIONS[case.corruption](task, plan)
    damaged = [step_id for error in corruption.injected for step_id in error.damaged_step_ids]

    repairer, backend = build_repairer(case, backend_kind, plan, steps, temperature)

    started = time.monotonic()
    repaired, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=damaged,
    )
    elapsed = time.monotonic() - started

    return {
        "case": {"model": case.model, "domain": case.domain, "corruption": case.corruption},
        "backend": (
            getattr(backend, "settings", lambda: {"backend": backend_kind})()
            if backend is not None
            else {"backend": case.model}
        ),
        "damaged_step_ids": damaged,
        "masked_step_ids": (
            mask.masked_step_ids if (mask := getattr(repairer, "last_mask", None)) else []
        ),
        "masked_token_count": (
            alignment.masked_token_count
            if (alignment := getattr(repairer, "last_alignment", None))
            else None
        ),
        "failures": [failure.model_dump() for failure in getattr(repairer, "failures", [])],
        # What the model produced, before anything was parsed. Without this a parse failure has
        # no explanation, only a line number.
        "diagnostics": (
            found.model_dump() if (found := getattr(repairer, "last_diagnostics", None)) else None
        ),
        "backend_diagnostics": (backend.diagnostics() if hasattr(backend, "diagnostics") else None),
        "score": score.model_dump(),
        "solved": score.solved,
        "collateral_total": score.collateral_total,
        "elapsed_seconds": round(elapsed, 3),
        "repaired_plan": repaired.model_dump(),
    }


def progress(cases: list[Case], done: int, description: str) -> Iterator[Case]:
    """A bar over every case, starting at however many were already finished."""
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:  # pragma: no cover - tqdm ships with transformers
        yield from cases
        return
    bar = tqdm(total=len(cases) + done, initial=done, desc=description, unit="case")
    try:
        for case in cases:
            bar.set_postfix_str(str(case))
            yield case
            bar.update(1)
    finally:
        bar.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=[*MODELS, *BASELINES, "all", "all-repairers"],
        default="all",
        help="'all' is the diffusion matrix; 'all-repairers' adds the baselines",
    )
    parser.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    parser.add_argument("--corruption", choices=[*CORRUPTIONS, "all"], default="all")
    parser.add_argument(
        "--backend",
        choices=["torch", "oracle", "echo"],
        default="torch",
        help="torch runs the real model; oracle and echo are mocks for checking the flow",
    )
    parser.add_argument("--steps", type=int, default=64, help="denoising passes")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 keeps it greedy")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=None, help="stop after N new cases")
    parser.add_argument("--force", action="store_true", help="redo cases already on disk")
    arguments = parser.parse_args(argv)

    if arguments.model == "all":
        models = list(MODELS)
    elif arguments.model == "all-repairers":
        models = [*MODELS, *BASELINES]
    else:
        models = [arguments.model]
    domains = list(DOMAINS) if arguments.domain == "all" else [arguments.domain]
    corruptions = list(CORRUPTIONS) if arguments.corruption == "all" else [arguments.corruption]
    cases = enumerate_cases(models, domains, corruptions)

    arguments.out.mkdir(parents=True, exist_ok=True)
    finished = [case for case in cases if (arguments.out / f"{case.key}.json").exists()]
    pending = cases if arguments.force else [case for case in cases if case not in finished]
    if arguments.limit is not None:
        pending = pending[: arguments.limit]

    print(
        f"{len(cases)} case(s): {len(finished)} already done, {len(pending)} to run "
        f"[backend={arguments.backend}, steps={arguments.steps}, "
        f"temperature={arguments.temperature}]"
    )

    failed = 0
    for case in progress(
        pending, done=len(finished) if not arguments.force else 0, description="repairing"
    ):
        destination = arguments.out / f"{case.key}.json"
        try:
            result = run_case(case, arguments.backend, arguments.steps, arguments.temperature)
        except Exception as exc:  # a case that blows up must not take the batch with it
            failed += 1
            print(f"\n  {case}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    written = sorted(arguments.out.glob("*.json"))
    print(
        f"wrote {len(written)} result file(s) to {arguments.out}"
        + (f", {failed} case(s) errored" if failed else "")
    )
    report_diagnostics(written)
    return 1 if failed else 0


def report_diagnostics(written: list[Path]) -> None:
    """Say what the batch's parse failures have in common, if anything.

    The first fork when reading a run: failures that still carry mask tokens point at the
    denoising budget, failures without them point at the format or at the mask boundaries.
    """
    results = []
    for path in written:
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summary = summarise(results)
    if not summary["with_diagnostics"]:
        return
    print(
        f"parse failures: {summary['parse_failures']}/{summary['with_diagnostics']}"
        f" — of those, {summary['parse_failures_with_mask_tokens_left']} still had mask tokens"
        f" in the parsed text"
    )
    if summary["parse_failures"]:
        share = summary["share_of_failures_still_masked"]
        print(
            "  a high share points at the denoising budget (--steps); a low one at the format "
            f"or the mask boundaries  [share={share}]"
        )


if __name__ == "__main__":
    raise SystemExit(main())
