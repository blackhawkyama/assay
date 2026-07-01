"""Eval spec for an external property-inspection report engine.

    REPORT_ENGINE_DIR=/path/to/engine/API assay run adapters/report_engine/spec.py

Points assay at a local checkout of the engine (the directory containing
`scripts/regen_full.py`), runs the golden inspection set through it, and grades
each report. Nothing deploys.

Scorers, and what they need:
  - no_error / risk_score bounds / faithfulness  → need only ambient credentials
  - required_coverage / no_fabrication           → need ground-truth labels in
    golden.yaml (ported from the engine's own hand-verified eval fixtures)

Requires the engine's `.env` (LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY) — the
same setup its `regen_full.py` already uses. The faithfulness judge additionally
calls the Anthropic API directly and reads the ambient credentials.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `adapters.report_engine.adapter` importable no matter how the spec is run.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adapters.report_engine.adapter import (  # noqa: E402
    NoFabrication,
    ReportEngineSystem,
    RequiredCoverage,
)
from assay.dataset import Dataset  # noqa: E402
from assay.scorers import LatencyUnder, LLMJudge, NoError, NumericBounds, Rubric  # noqa: E402

_HERE = Path(__file__).resolve().parent


def build():
    engine_dir = os.environ.get("REPORT_ENGINE_DIR")
    if not engine_dir:
        raise SystemExit(
            "set REPORT_ENGINE_DIR to the engine's API directory "
            "(the one containing scripts/regen_full.py)"
        )
    system = ReportEngineSystem(engine_dir)
    dataset = Dataset.from_yaml(_HERE / "golden.yaml")

    faithfulness = LLMJudge(
        Rubric(
            name="exec_summary_faithfulness",
            criteria=(
                "Grade the executive summary against the extracted risks. Every "
                "claim in the summary must be supported by a risk item; penalize "
                "invented defects, inflated severities, and figures not present in "
                "the report. Reward a concise, accurate synthesis."
            ),
            scale_min=1,
            scale_max=5,
            pass_at=0.6,
        ),
        name="faithfulness",
    )

    scorers = [
        NoError(),
        RequiredCoverage(),                         # needs golden labels
        NoFabrication(),                            # needs golden labels
        NumericBounds("risk_score", name="risk_score_bounds",
                      lo=0, hi=100),                # sanity window; tighten per case
        faithfulness,                               # LLM judge on exec_summary
        LatencyUnder(budget_s=180.0),
    ]
    return system, dataset, scorers
