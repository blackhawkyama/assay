"""Adapter: an external document→report engine as an assay System.

A worked example of adapting a real, script-driven system to assay without
rewriting it. The target here is a property-inspection report engine: it turns
inspection/permit PDFs into a structured report. Locally that pipeline is driven
by a `regen_full.py <job-spec.json>` script that writes `out/<name>.report.json`.

This adapter runs that same script per case and maps the report into a flat
`prediction.output` that assay scorers read:

    {
      "risk_score":       float,          # report.property_risk_score.overall_score
      "risk_level":       "low|medium|…", # report.property_risk_score.risk_level
      "repair_total_usd": float,          # …property_risk_score.details.repair_total_usd
      "risks":            [ {..}, ... ],  # report.risks.items
      "risk_text":        "…",            # all risk titles/descriptions, joined
      "exec_summary":     "…",            # report.executive_summary text
      "permits":          [ {..}, ... ],  # report.permits.items
      "n_risks":          int,
      "report":           {..},           # the full report, for bespoke scorers
    }

A `Case.input` for this system is the engine's job spec (the JSON the script
consumes). Nothing here deploys — it runs entirely against a local checkout of
the engine and its own `.env`, exactly like running the script by hand. Point it
at that checkout with the `REPORT_ENGINE_DIR` environment variable (see spec.py).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from assay.scorers.base import make_score
from assay.types import Case, Prediction, Score


def _text_of(node: Any) -> str:
    """Best-effort flatten of a report node to searchable text."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_text_of(v) for v in node.values())
    if isinstance(node, list):
        return " ".join(_text_of(v) for v in node)
    return "" if node is None else str(node)


def map_report(doc: dict) -> dict:
    """Project a raw report.json into the flat shape scorers read."""
    report = doc.get("report", doc)  # tolerate {report:{...}} or bare report
    risks = (report.get("risks") or {}).get("items") or []
    permits = (report.get("permits") or {}).get("items") or []
    score_node = report.get("property_risk_score") or {}
    details = score_node.get("details") or {}
    return {
        "risk_score": score_node.get("overall_score"),
        "risk_level": score_node.get("risk_level"),
        "repair_total_usd": details.get("repair_total_usd"),
        "risks": risks,
        "risk_text": _text_of(risks),
        "exec_summary": _text_of(report.get("executive_summary")),
        "permits": permits,
        "n_risks": len(risks),
        "report": report,
    }


class ReportEngineSystem:
    """Run the engine's `regen_full.py` per case and return the mapped report.

    `engine_dir` is the path to the engine's API directory (the one containing
    `scripts/regen_full.py`). The system uses that checkout's virtualenv python
    and `.env`, so all model/API config comes from the engine — this adapter adds
    no credentials of its own.
    """

    def __init__(
        self,
        engine_dir: str | Path,
        name: str = "report-engine-v2",
        python: str | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        self.dir = Path(engine_dir).resolve()
        self.name = name
        self.python = python or str(self.dir / ".venv" / "bin" / "python")
        self.timeout_s = timeout_s
        if not (self.dir / "scripts" / "regen_full.py").exists():
            raise FileNotFoundError(f"scripts/regen_full.py not found under {self.dir}")

    def predict(self, case: Case) -> Prediction:
        spec = dict(case.input)
        name = spec.get("name") or case.id
        spec["name"] = name
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".job.json", dir=self.dir, delete=False
        ) as fh:
            json.dump(spec, fh)
            spec_path = fh.name
        try:
            proc = subprocess.run(
                [self.python, "scripts/regen_full.py", spec_path],
                cwd=str(self.dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                return Prediction(
                    error=f"regen_full exit {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[-500:]}",
                    latency_s=elapsed,
                )
            report_path = self.dir / "out" / f"{name}.report.json"
            if not report_path.exists():
                return Prediction(
                    error=f"report not written: {report_path}", latency_s=elapsed
                )
            doc = json.loads(report_path.read_text())
            return Prediction(
                output=map_report(doc),
                latency_s=elapsed,
                raw={"report_path": str(report_path)},
            )
        except subprocess.TimeoutExpired:
            return Prediction(
                error=f"timeout after {self.timeout_s}s",
                latency_s=time.perf_counter() - start,
            )
        finally:
            Path(spec_path).unlink(missing_ok=True)


# --- domain scorers (the anti-hallucination checks) ------------------------
#
# These encode the ground-truth semantics an inspection report must satisfy: it
# must surface the genuinely-flagged defects (`required`) and must NOT invent
# unmarked ones (`forbidden`). Ground truth lives on `case.expected` and is
# ported from the engine's own hand-verified eval fixtures.


class NoFabrication:
    """None of the `forbidden` findings (known false positives — unmarked boxes,
    misread comments) may appear in the report's risk text. Score is the fraction
    correctly absent; any fabrication fails the case."""

    def __init__(self, forbidden_key: str = "forbidden", name: str = "no_fabrication") -> None:
        self.key = forbidden_key
        self.name = name

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        forbidden = case.expected.get(self.key, [])
        if not forbidden:
            return make_score(self.name, 1.0, passed=True, detail="no forbidden set")
        hay = str(prediction.output.get("risk_text", "")).casefold()
        fabricated = [f for f in forbidden if f.casefold() in hay]
        frac_ok = 1.0 - len(fabricated) / len(forbidden)
        return make_score(
            self.name,
            frac_ok,
            passed=not fabricated,
            detail="" if not fabricated else f"fabricated: {fabricated}",
        )


class RequiredCoverage:
    """Every genuinely-flagged defect in `required` must appear in the report's
    risk text. Score is the fraction covered."""

    def __init__(self, required_key: str = "required", name: str = "required_coverage") -> None:
        self.key = required_key
        self.name = name

    def score(self, case: Case, prediction: Prediction) -> Score:
        if not prediction.ok:
            return make_score(self.name, 0.0, passed=False, detail="prediction errored")
        required = case.expected.get(self.key, [])
        if not required:
            return make_score(self.name, 1.0, passed=True, detail="no required set")
        hay = str(prediction.output.get("risk_text", "")).casefold()
        missing = [r for r in required if r.casefold() not in hay]
        frac = 1.0 - len(missing) / len(required)
        return make_score(
            self.name,
            frac,
            passed=not missing,
            detail="" if not missing else f"omitted: {missing}",
        )
