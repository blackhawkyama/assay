"""Run a System over a Dataset with a set of Scorers, and persist the result.

Cases run concurrently (systems and judges are I/O-bound). A scorer that raises
is caught and recorded as a failed Score rather than killing the run — one flaky
judge call shouldn't discard the other 99 cases' worth of signal.
"""

from __future__ import annotations

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from assay.dataset import Dataset
from assay.scorers.base import Scorer
from assay.systems import System
from assay.types import Case, CaseResult, RunResult, Score


def _run_id(system: str, dataset: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{system}-{dataset}".replace("/", "_").replace(" ", "_")
    return f"{slug}-{stamp}-{uuid.uuid4().hex[:6]}"


def _score_one(scorers: Sequence[Scorer], case: Case, prediction) -> list[Score]:
    out: list[Score] = []
    for scorer in scorers:
        try:
            out.append(scorer.score(case, prediction))
        except Exception as exc:  # noqa: BLE001 — a scorer fault is data, not a crash
            out.append(
                Score(
                    scorer=getattr(scorer, "name", "scorer"),
                    value=0.0,
                    passed=None,
                    detail=f"scorer error: {type(exc).__name__}: {exc}"[:300],
                )
            )
    return out


def run(
    system: System,
    dataset: Dataset,
    scorers: Sequence[Scorer],
    *,
    concurrency: int = 8,
    out_dir: str | Path = "runs",
    save: bool = True,
    progress: bool = True,
    config: Optional[dict] = None,
) -> RunResult:
    run_id = _run_id(system.name, dataset.name)

    def work(case: Case) -> CaseResult:
        prediction = system.predict(case)
        scores = _score_one(scorers, case, prediction)
        return CaseResult(case=case, prediction=prediction, scores=scores)

    results: list[CaseResult] = []
    total = len(dataset)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, c): c for c in dataset}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if progress:
                print(f"\r  {i}/{total} cases", end="", file=sys.stderr, flush=True)
    if progress and total:
        print(file=sys.stderr)

    # Restore dataset order (as_completed yields out of order).
    order = {c.id: i for i, c in enumerate(dataset)}
    results.sort(key=lambda r: order.get(r.case.id, 0))

    run_result = RunResult(
        run_id=run_id,
        system=system.name,
        dataset=dataset.name,
        dataset_version=dataset.version,
        config={
            "scorers": [getattr(s, "name", "scorer") for s in scorers],
            "concurrency": concurrency,
            **(config or {}),
        },
        results=results,
    )

    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{run_id}.json"
        path.write_text(run_result.model_dump_json(indent=2))
        if progress:
            print(f"  saved → {path}", file=sys.stderr)

    return run_result


def load_run(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text())


def latest_run(out_dir: str | Path = "runs", system: Optional[str] = None) -> Optional[RunResult]:
    """Most recent saved run, optionally filtered to one system. Handy as the
    implicit baseline for `compare`/`gate`."""
    out = Path(out_dir)
    if not out.exists():
        return None
    runs = sorted(out.glob("*.json"))
    for path in reversed(runs):
        r = load_run(path)
        if system is None or r.system == system:
            return r
    return None
