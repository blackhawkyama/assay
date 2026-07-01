"""assay — a small, honest LLM-evaluation framework.

Ship a change to an LLM system, then know whether it got better or worse.

The spine:
    Dataset  → versioned golden cases (inputs + optional expectations)
    System   → the thing under test (a callable, a script, an API)
    Scorer   → grades one prediction (deterministic checks or an LLM judge)
    Runner   → runs a System over a Dataset with Scorers, captures cost/latency
    Report   → aggregates, diffs against a baseline, flags regressions, gates CI
"""

from assay.types import (
    Case,
    CaseResult,
    Prediction,
    RunResult,
    Score,
)

__all__ = [
    "Case",
    "Prediction",
    "Score",
    "CaseResult",
    "RunResult",
]

__version__ = "0.1.0"
