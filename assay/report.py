"""Turn runs into answers: format one run, diff two runs, gate a run for CI.

`compare` is the workhorse — per-scorer deltas plus the specific cases that
regressed, so a drop in the mean points you at the rows that caused it. `gate`
turns a run (optionally vs a baseline) into a pass/fail with reasons and an exit
code, which is what a CI step actually needs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from assay.types import RunResult


# --- single-run formatting ------------------------------------------------


def _fmt(v: Optional[float], pct: bool = False) -> str:
    if v is None:
        return "  —  "
    return f"{v * 100:5.1f}%" if pct else f"{v:6.3f}"


def format_run(run: RunResult) -> str:
    s = run.summary()
    p50 = s["latency_p50_s"]
    p50_str = "—" if p50 is None else f"{p50:.2f}s"
    lines = [
        f"Run {run.run_id}",
        f"  system={run.system}  dataset={s['dataset']}  n={s['n_cases']}",
        f"  errors={_fmt(s['error_rate'], pct=True)}  "
        f"cost=${s['total_cost_usd']:.4f}  p50={p50_str}",
        "",
        f"  {'scorer':<28}{'mean':>8}{'pass':>9}{'n':>5}",
        f"  {'-' * 28}{'-' * 8}{'-' * 9}{'-' * 5}",
    ]
    for name, agg in s["scorers"].items():
        lines.append(
            f"  {name:<28}{_fmt(agg['mean']):>8}"
            f"{_fmt(agg['pass_rate'], pct=True):>9}{agg['n']:>5}"
        )
    return "\n".join(lines)


# --- comparison of two runs -----------------------------------------------


class ScorerDelta(BaseModel):
    scorer: str
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    delta: Optional[float]
    regressed: bool


class CaseRegression(BaseModel):
    case_id: str
    scorer: str
    baseline: float
    candidate: float
    detail: str = ""


class Comparison(BaseModel):
    baseline_run: str
    candidate_run: str
    dataset_matches: bool
    scorer_deltas: list[ScorerDelta]
    case_regressions: list[CaseRegression]

    @property
    def has_regression(self) -> bool:
        return any(d.regressed for d in self.scorer_deltas)


def compare(
    baseline: RunResult,
    candidate: RunResult,
    *,
    min_delta: float = 0.02,
    case_epsilon: float = 0.001,
) -> Comparison:
    """Diff candidate against baseline.

    A scorer 'regresses' if its mean drops by more than `min_delta`. A case
    regresses if its per-scorer score fell by more than `case_epsilon` — those
    are the exact rows to go look at.
    """
    dataset_matches = (
        baseline.dataset == candidate.dataset
        and baseline.dataset_version == candidate.dataset_version
    )

    scorers = sorted(set(baseline.scorer_names()) | set(candidate.scorer_names()))
    deltas: list[ScorerDelta] = []
    for name in scorers:
        b, c = baseline.mean(name), candidate.mean(name)
        d = (c - b) if (b is not None and c is not None) else None
        deltas.append(
            ScorerDelta(
                scorer=name,
                baseline_mean=b,
                candidate_mean=c,
                delta=d,
                regressed=d is not None and d < -min_delta,
            )
        )

    # Per-case regressions, matched by case id.
    base_by_id = {r.case.id: r for r in baseline.results}
    regressions: list[CaseRegression] = []
    for cr in candidate.results:
        br = base_by_id.get(cr.case.id)
        if not br:
            continue
        for cs in cr.scores:
            bs = br.score_for(cs.scorer)
            if bs is None:
                continue
            if cs.value < bs.value - case_epsilon:
                regressions.append(
                    CaseRegression(
                        case_id=cr.case.id,
                        scorer=cs.scorer,
                        baseline=bs.value,
                        candidate=cs.value,
                        detail=cs.detail[:160],
                    )
                )
    regressions.sort(key=lambda r: r.candidate - r.baseline)  # worst first

    return Comparison(
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        dataset_matches=dataset_matches,
        scorer_deltas=deltas,
        case_regressions=regressions,
    )


def format_comparison(cmp: Comparison, top_cases: int = 10) -> str:
    lines = [
        f"Compare  baseline={cmp.baseline_run}",
        f"         candidate={cmp.candidate_run}",
    ]
    if not cmp.dataset_matches:
        lines.append("  ⚠  datasets differ — deltas may not be comparable")
    lines += [
        "",
        f"  {'scorer':<28}{'base':>8}{'cand':>8}{'Δ':>9}",
        f"  {'-' * 28}{'-' * 8}{'-' * 8}{'-' * 9}",
    ]
    for d in cmp.scorer_deltas:
        arrow = "  ↓" if d.regressed else ("  ↑" if (d.delta or 0) > 0.02 else "")
        lines.append(
            f"  {d.scorer:<28}{_fmt(d.baseline_mean):>8}{_fmt(d.candidate_mean):>8}"
            f"{('' if d.delta is None else f'{d.delta:+.3f}'):>9}{arrow}"
        )
    if cmp.case_regressions:
        lines += ["", f"  Regressed cases ({len(cmp.case_regressions)}):"]
        for cr in cmp.case_regressions[:top_cases]:
            lines.append(
                f"    {cr.case_id:<22} {cr.scorer:<20} "
                f"{cr.baseline:.2f} → {cr.candidate:.2f}  {cr.detail}"
            )
        if len(cmp.case_regressions) > top_cases:
            lines.append(f"    …and {len(cmp.case_regressions) - top_cases} more")
    else:
        lines += ["", "  No per-case regressions."]
    return "\n".join(lines)


# --- gate (CI) ------------------------------------------------------------


class GateResult(BaseModel):
    passed: bool
    reasons: list[str]

    def __str__(self) -> str:
        head = "GATE PASS" if self.passed else "GATE FAIL"
        return "\n".join([head, *(f"  - {r}" for r in self.reasons)])


def gate(
    run: RunResult,
    *,
    thresholds: Optional[dict[str, float]] = None,
    max_error_rate: Optional[float] = None,
    baseline: Optional[RunResult] = None,
    min_delta: float = 0.02,
) -> GateResult:
    """Pass/fail a run for CI.

    - `thresholds`: {scorer_name: min_mean} the candidate must meet.
    - `max_error_rate`: hard cap on SUT error rate.
    - `baseline`: if given, fail on any scorer regression beyond `min_delta`.
    """
    reasons: list[str] = []
    passed = True

    if max_error_rate is not None and run.error_rate() > max_error_rate:
        passed = False
        reasons.append(
            f"error_rate {run.error_rate():.1%} > cap {max_error_rate:.1%}"
        )

    for name, floor in (thresholds or {}).items():
        got = run.mean(name)
        if got is None:
            passed = False
            reasons.append(f"{name}: no scores (expected ≥ {floor:.3f})")
        elif got < floor:
            passed = False
            reasons.append(f"{name}: mean {got:.3f} < floor {floor:.3f}")
        else:
            reasons.append(f"{name}: mean {got:.3f} ≥ floor {floor:.3f} ✓")

    if baseline is not None:
        cmp = compare(baseline, run, min_delta=min_delta)
        for d in cmp.scorer_deltas:
            if d.regressed:
                passed = False
                reasons.append(
                    f"{d.scorer}: regressed {d.delta:+.3f} vs baseline "
                    f"(> {min_delta:.3f})"
                )

    if not reasons:
        reasons.append("no gate criteria configured — passing by default")
    return GateResult(passed=passed, reasons=reasons)
