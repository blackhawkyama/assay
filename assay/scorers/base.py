"""Scorer protocol and shared plumbing."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from assay.types import Case, Prediction, Score


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, case: Case, prediction: Prediction) -> Score: ...


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted path against nested dicts/lists.

    Supports keys and integer list indices: "summary.risks.0.severity".
    Returns `default` on any miss rather than raising — scorers decide what a
    missing value means.
    """
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        try:
            if isinstance(cur, dict):
                cur = cur[part]
            elif isinstance(cur, (list, tuple)):
                cur = cur[int(part)]
            else:
                return default
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    return cur


def make_score(
    scorer: str,
    value: float,
    *,
    passed: Optional[bool] = None,
    threshold: Optional[float] = None,
    label: Optional[str] = None,
    detail: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> Score:
    """Construct a Score, deriving `passed` from a threshold when not given."""
    if passed is None and threshold is not None:
        passed = value >= threshold
    return Score(
        scorer=scorer,
        value=float(value),
        passed=passed,
        label=label,
        detail=detail,
        metadata=metadata or {},
    )
