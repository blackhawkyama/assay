"""Command line: run an eval spec, show a run, compare two, gate for CI.

An eval *spec* is a Python file that wires the pieces together and exposes either
module-level `system`, `dataset`, `scorers`, or a `build()` returning that trio.
The CLI imports it and drives the run — the spec owns the wiring, the CLI owns
the plumbing (concurrency, persistence, exit codes).

    assay run     examples/toy_spec.py
    assay show    runs/<id>.json
    assay compare runs/<base>.json runs/<cand>.json
    assay gate    runs/<cand>.json --threshold judge:quality=0.7 --max-error-rate 0.0
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from assay import report
from assay.dataset import Dataset
from assay.runner import load_run, run as run_eval
from assay.scorers.base import Scorer
from assay.systems import System


def _load_spec(path: str) -> tuple[System, Dataset, list[Scorer]]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"spec not found: {path}")
    spec = importlib.util.spec_from_file_location(f"assay_spec_{p.stem}", p)
    if spec is None or spec.loader is None:
        sys.exit(f"could not import spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "build"):
        system, dataset, scorers = module.build()
    else:
        try:
            system = module.system
            dataset = module.dataset
            scorers = module.scorers
        except AttributeError:
            sys.exit(
                f"{path} must define `build()` returning (system, dataset, scorers) "
                "or module-level `system`, `dataset`, `scorers`"
            )
    return system, dataset, list(scorers)


def _parse_thresholds(items: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"bad --threshold {item!r}; expected name=value")
        name, val = item.rsplit("=", 1)
        out[name] = float(val)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    system, dataset, scorers = _load_spec(args.spec)
    result = run_eval(
        system,
        dataset,
        scorers,
        concurrency=args.concurrency,
        out_dir=args.out,
        save=not args.no_save,
    )
    print(report.format_run(result))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    print(report.format_run(load_run(args.run)))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base = load_run(args.baseline)
    cand = load_run(args.candidate)
    print(report.format_comparison(report.compare(base, cand, min_delta=args.min_delta)))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    cand = load_run(args.candidate)
    baseline = load_run(args.baseline) if args.baseline else None
    result = report.gate(
        cand,
        thresholds=_parse_thresholds(args.threshold),
        max_error_rate=args.max_error_rate,
        baseline=baseline,
        min_delta=args.min_delta,
    )
    print(report.format_run(cand))
    print()
    print(result)
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="assay", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an eval spec")
    r.add_argument("spec", help="path to a .py eval spec")
    r.add_argument("-c", "--concurrency", type=int, default=8)
    r.add_argument("-o", "--out", default="runs", help="output dir for run artifacts")
    r.add_argument("--no-save", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="print a saved run")
    s.add_argument("run", help="path to a runs/<id>.json")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("compare", help="diff two runs")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--min-delta", type=float, default=0.02)
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("gate", help="pass/fail a run for CI (exit 1 on fail)")
    g.add_argument("candidate")
    g.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=VAL",
        help="minimum mean for a scorer; repeatable",
    )
    g.add_argument("--max-error-rate", type=float, default=None)
    g.add_argument("--baseline", default=None, help="fail on regression vs this run")
    g.add_argument("--min-delta", type=float, default=0.02)
    g.set_defaults(func=cmd_gate)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
