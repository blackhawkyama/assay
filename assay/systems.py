"""Systems under test — the thing whose quality you're measuring.

A System takes a Case and returns a Prediction. It never raises for an ordinary
failure: it catches, records the error on the Prediction, and lets scorers treat
it as a miss. That keeps one bad case from killing a whole run.

Three ready-made shapes cover most needs:
  - CallableSystem:   wrap any `fn(input: dict) -> output`.
  - SubprocessSystem: shell out to a script/binary (e.g. a report-gen script).
  - stub/echo systems live in examples/ for tests and demos.

For anything bespoke, subclass System and implement `predict`.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from assay.types import Case, Prediction


@runtime_checkable
class System(Protocol):
    name: str

    def predict(self, case: Case) -> Prediction: ...


class CallableSystem:
    """Wrap a plain function. The function sees `case.input`; its return value
    becomes `prediction.output`. Timing and error capture are handled here."""

    def __init__(self, fn: Callable[[dict[str, Any]], Any], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def predict(self, case: Case) -> Prediction:
        start = time.perf_counter()
        try:
            output = self._fn(case.input)
            return Prediction(output=output, latency_s=time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 — a SUT failure is data, not a crash
            return Prediction(
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - start,
            )


class SubprocessSystem:
    """Run an external command per case and parse its output.

    `build_cmd(case)` returns the argv list. By default the case input is sent
    on stdin as JSON and the process is expected to print JSON to stdout; supply
    `parse` to read a file it wrote, pull a field, etc. This is the bridge to
    systems that already exist as scripts — you evaluate them without a rewrite.
    """

    def __init__(
        self,
        build_cmd: Callable[[Case], list[str]],
        name: str = "subprocess",
        parse: Optional[Callable[[subprocess.CompletedProcess], Any]] = None,
        send_input_json: bool = True,
        cwd: Optional[str | Path] = None,
        timeout_s: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.build_cmd = build_cmd
        self.name = name
        self._parse = parse or (lambda p: json.loads(p.stdout or "null"))
        self._send_input_json = send_input_json
        self._cwd = str(cwd) if cwd else None
        self._timeout = timeout_s
        self._env = env

    def predict(self, case: Case) -> Prediction:
        cmd = self.build_cmd(case)
        stdin = json.dumps(case.input) if self._send_input_json else None
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=self._timeout,
                env=self._env,
            )
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                return Prediction(
                    error=f"exit {proc.returncode}: {(proc.stderr or '').strip()[:500]}",
                    latency_s=elapsed,
                    raw={"stdout": proc.stdout, "stderr": proc.stderr},
                )
            return Prediction(
                output=self._parse(proc),
                latency_s=elapsed,
                raw={"cmd": cmd},
            )
        except subprocess.TimeoutExpired:
            return Prediction(
                error=f"timeout after {self._timeout}s",
                latency_s=time.perf_counter() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return Prediction(
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - start,
            )
