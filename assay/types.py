"""Core data model. Everything that flows through assay is one of these."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Case(BaseModel):
    """One evaluation example.

    `input` is handed to the System. `expected` (optional) holds reference data
    the scorers compare against — a golden output, a set of required facts, a
    numeric target, whatever the scorers you attach happen to read.
    """

    id: str
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """What a System produced for one Case.

    `output` is the structured (or raw) result scorers read. `error` is set
    instead when the System blew up — scorers should treat that as a failure,
    not crash on it.
    """

    output: Any = None
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class Score(BaseModel):
    """One scorer's verdict on one prediction.

    `value` is normalized to [0, 1] so heterogeneous scorers aggregate cleanly.
    `passed` is the boolean view used for pass-rate and CI gating; when a scorer
    doesn't set it, the runner derives it from `value` and the scorer threshold.
    """

    scorer: str
    value: float = 0.0
    passed: Optional[bool] = None
    label: Optional[str] = None
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case: Case
    prediction: Prediction
    scores: list[Score] = Field(default_factory=list)

    def score_for(self, scorer: str) -> Optional[Score]:
        return next((s for s in self.scores if s.scorer == scorer), None)


class RunResult(BaseModel):
    """The artifact of one evaluation run. Serializable to JSON, diffable."""

    run_id: str
    system: str
    dataset: str
    dataset_version: str = "unversioned"
    created_at: str = Field(default_factory=_utcnow)
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[CaseResult] = Field(default_factory=list)

    # -- aggregation -------------------------------------------------------

    def scorer_names(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            for s in r.scores:
                if s.scorer not in seen:
                    seen.append(s.scorer)
        return seen

    def mean(self, scorer: str) -> Optional[float]:
        vals = [s.value for r in self.results if (s := r.score_for(scorer))]
        return statistics.fmean(vals) if vals else None

    def pass_rate(self, scorer: str) -> Optional[float]:
        flags = [
            bool(s.passed)
            for r in self.results
            if (s := r.score_for(scorer)) and s.passed is not None
        ]
        return statistics.fmean([float(f) for f in flags]) if flags else None

    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        errs = sum(1 for r in self.results if not r.prediction.ok)
        return errs / len(self.results)

    def total_cost(self) -> float:
        return sum(r.prediction.cost_usd or 0.0 for r in self.results)

    def latency_p50(self) -> Optional[float]:
        lat = sorted(
            r.prediction.latency_s for r in self.results if r.prediction.latency_s is not None
        )
        return statistics.median(lat) if lat else None

    def summary(self) -> dict[str, Any]:
        """A compact per-scorer rollup — the numbers a human actually reads."""
        by_scorer: dict[str, dict[str, Any]] = {}
        for name in self.scorer_names():
            by_scorer[name] = {
                "mean": self.mean(name),
                "pass_rate": self.pass_rate(name),
                "n": sum(1 for r in self.results if r.score_for(name)),
            }
        return {
            "run_id": self.run_id,
            "system": self.system,
            "dataset": f"{self.dataset}@{self.dataset_version}",
            "n_cases": len(self.results),
            "error_rate": self.error_rate(),
            "total_cost_usd": self.total_cost(),
            "latency_p50_s": self.latency_p50(),
            "scorers": by_scorer,
        }

    def by_tag(self, scorer: str) -> dict[str, float]:
        """Mean of one scorer broken out per tag — where slices hide."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            s = r.score_for(scorer)
            if not s:
                continue
            for tag in r.case.tags or ["<untagged>"]:
                buckets[tag].append(s.value)
        return {tag: statistics.fmean(v) for tag, v in buckets.items()}
