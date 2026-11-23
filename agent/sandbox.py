"""Layer 2: run checked code in a separate, limited Python process.

What the sandbox process gets:   a temporary folder with a copy of the data, the code,
                                and nothing else - no API keys, no network, time,
                                CPU and memory limits.
What comes back:                 JSON only (a table, a chart or a value), or an error.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.io as pio

from agent.code_check import check_code

RUNNER = Path(__file__).with_name("sandbox_runner.py")
TIMEOUT_SECONDS = 30
MAX_OUTPUT_BYTES = 5_000_000


@dataclass
class SandboxResult:
    ok: bool
    kind: str = ""            # "figure", "table" or "value"
    value: object = None
    error: str = ""


def run_code(code: str, df: pd.DataFrame) -> SandboxResult:
    check = check_code(code)
    if not check.ok:
        return SandboxResult(ok=False, error="Code check failed: " + " ".join(check.problems))

    with tempfile.TemporaryDirectory(prefix="eda_sandbox_") as tmp:
        work = Path(tmp)
        df.to_parquet(work / "data.parquet")
        (work / "code.py").write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(work)],
                check=False, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                cwd=work,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},  # no API keys or other secrets
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, error=f"Timed out after {TIMEOUT_SECONDS} seconds.")

    if len(proc.stdout) > MAX_OUTPUT_BYTES:
        return SandboxResult(ok=False, error="The result was too large.")
    try:
        reply = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        detail = (proc.stderr or "no output").strip().splitlines()[-1:] or ["no output"]
        return SandboxResult(ok=False, error=f"The sandbox crashed: {detail[0][:300]}")

    if not reply["ok"]:
        return SandboxResult(ok=False, error=reply["error"])
    kind, value = reply["type"], reply["value"]
    if kind == "figure":
        value = pio.from_json(value)
    elif kind == "table":
        value = pd.read_json(StringIO(value), orient="split")
    return SandboxResult(ok=True, kind=kind, value=value)