"""LLM-as-judge scorers.

For quality that regex can't reach — is the summary faithful to the source, is
the tone right, is the answer actually correct — a model grades the output
against a rubric. Two guards against the usual failure modes of LLM judges:

  1. Structured output. The judge must return a JSON verdict (reasoning + score
     + pass), enforced by the API's structured-output mode. No prose to parse,
     no "as an AI…" preamble to strip.

  2. Position-bias mitigation for pairwise. Models favor whichever candidate is
     shown first. PairwiseJudge runs both orderings and only counts a win the
     judge holds regardless of order.

The judge model defaults to Claude Opus 4.8. Token usage and an estimated cost
land on each Score's metadata so the eval's own bill is visible, not hidden.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from pydantic import BaseModel, Field

from assay.scorers.base import make_score
from assay.types import Case, Prediction, Score

# Judge-model price table ($ per 1M tokens), input/output. Used only to estimate
# the eval's own cost — kept small and explicit rather than fetched live.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_client_lock = threading.Lock()
_shared_client: Any = None


def _client() -> Any:
    """Lazily construct one shared Anthropic client. Import is deferred so the
    rest of assay works with no SDK installed and no key set."""
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            import anthropic  # noqa: PLC0415 — deferred on purpose

            _shared_client = anthropic.Anthropic()
        return _shared_client


def _estimate_cost(model: str, usage: Any) -> Optional[float]:
    price = _PRICES.get(model)
    if not price or usage is None:
        return None
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    return (in_tok * price[0] + out_tok * price[1]) / 1_000_000


def _render(value: Any, limit: int = 6000) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


class Rubric(BaseModel):
    """What the judge grades against. `criteria` is prose telling the judge what
    a good output looks like; keep it concrete and gradeable."""

    name: str = "quality"
    criteria: str
    scale_min: int = 1
    scale_max: int = 5
    pass_at: float = 0.6  # normalized [0,1] threshold for `passed`

    def normalize(self, raw: int | float) -> float:
        span = self.scale_max - self.scale_min
        if span <= 0:
            return 1.0 if raw >= self.scale_max else 0.0
        clamped = min(max(float(raw), self.scale_min), self.scale_max)
        return (clamped - self.scale_min) / span


class _Verdict(BaseModel):
    reasoning: str = Field(description="Brief justification grounded in the output.")
    score: int = Field(description="Integer score on the stated scale.")
    meets_bar: bool = Field(description="Whether the output clears the rubric's bar.")


class _Choice(BaseModel):
    reasoning: str = Field(description="Why the chosen candidate is better.")
    winner: str = Field(description='Exactly one of "A", "B", or "tie".')


_JUDGE_SYSTEM = (
    "You are a rigorous, impartial evaluator. Grade the candidate output strictly "
    "against the criteria and any reference provided. Reward substantiated, faithful, "
    "on-target answers; penalize claims unsupported by the input or reference, "
    "hallucinated specifics, and off-criteria padding. Judge only what is asked."
)


class LLMJudge:
    """Pointwise rubric scoring. Returns a Score in [0,1] with the judge's
    reasoning and token cost attached."""

    def __init__(
        self,
        rubric: Rubric,
        name: Optional[str] = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        thinking: bool = False,
        effort: Optional[str] = None,
    ) -> None:
        self.rubric = rubric
        self.name = name or f"judge:{rubric.name}"
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort

    def _prompt(self, case: Case, prediction: Prediction) -> str:
        parts = [
            "## Task input",
            _render(case.input),
            "\n## Grading criteria",
            self.rubric.criteria,
            f"\n## Scale\nScore from {self.rubric.scale_min} (worst) to "
            f"{self.rubric.scale_max} (best).",
        ]
        if case.expected:
            parts += ["\n## Reference (ground truth)", _render(case.expected)]
        parts += ["\n## Candidate output to grade", _render(prediction.output)]
        return "\n".join(parts)

    def _call(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, Optional[float]]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = _client().messages.parse(**kwargs)
        cost = _estimate_cost(self.model, getattr(resp, "usage", None))
        return resp.parsed_output, cost

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        verdict, cost = self._call(self._prompt(case, prediction), _Verdict)
        assert isinstance(verdict, _Verdict)
        value = self.rubric.normalize(verdict.score)
        return make_score(
            self.name,
            value,
            passed=value >= self.rubric.pass_at,
            label=f"{verdict.score}/{self.rubric.scale_max}",
            detail=verdict.reasoning,
            metadata={
                "raw_score": verdict.score,
                "meets_bar": verdict.meets_bar,
                "judge_model": self.model,
                "judge_cost_usd": cost,
            },
        )


class PairwiseJudge:
    """Prefer the candidate over a baseline, order-swapped to cancel position
    bias. The baseline output comes from `case.expected[baseline_key]`.

    Score: 1.0 candidate wins both orderings, 0.5 split/tie, 0.0 loses both."""

    def __init__(
        self,
        criteria: str,
        baseline_key: str = "baseline",
        name: str = "pairwise",
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        self.criteria = criteria
        self.baseline_key = baseline_key
        self.name = name
        self.model = model
        self.max_tokens = max_tokens

    def _ask(self, case: Case, a: Any, b: Any) -> tuple[str, str, Optional[float]]:
        prompt = "\n".join(
            [
                "## Task input",
                _render(case.input),
                "\n## Criteria for the better answer",
                self.criteria,
                "\n## Candidate A",
                _render(a),
                "\n## Candidate B",
                _render(b),
                '\nDecide which better satisfies the criteria. Answer "A", "B", or "tie".',
            ]
        )
        resp = _client().messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=_Choice,
        )
        choice = resp.parsed_output
        assert isinstance(choice, _Choice)
        cost = _estimate_cost(self.model, getattr(resp, "usage", None))
        return choice.winner.strip().lower(), choice.reasoning, cost

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        baseline = case.expected.get(self.baseline_key)
        if baseline is None:
            return make_score(self.name, 0.0, passed=None, detail="no baseline to compare")

        cand = prediction.output
        # Ordering 1: candidate is A. Ordering 2: candidate is B (swapped).
        w1, r1, c1 = self._ask(case, cand, baseline)
        w2, r2, c2 = self._ask(case, baseline, cand)
        cand_wins = (w1 == "a") + (w2 == "b")
        base_wins = (w1 == "b") + (w2 == "a")

        if cand_wins == 2:
            value, verdict = 1.0, "candidate wins both orderings"
        elif base_wins == 2:
            value, verdict = 0.0, "baseline wins both orderings"
        else:
            value, verdict = 0.5, "split or tie (position-sensitive)"

        cost = sum(c for c in (c1, c2) if c is not None) or None
        return make_score(
            self.name,
            value,
            passed=value >= 0.5,
            label=verdict,
            detail=f"order1={w1} ({r1[:120]}); order2={w2} ({r2[:120]})",
            metadata={"judge_model": self.model, "judge_cost_usd": cost},
        )
