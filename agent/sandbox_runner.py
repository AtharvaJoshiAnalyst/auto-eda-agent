"""Runs INSIDE the separate sandbox process. Never imported by the app.

Usage: python -I sandbox_runner.py <work_dir>
The work directory holds data.parquet and code.py. The answer is printed to
stdout as JSON - never pickle, because the code that produced it is untrusted.
"""

import json
import resource
import socket
import sys
from pathlib import Path

MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB (address space; charts need ~1.5 GB)
CPU_LIMIT_SECONDS = 20
MAX_TABLE_ROWS = 200


def apply_limits() -> None:
    # CPU time works on Linux and macOS; the memory limit is only enforced on Linux.
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        pass  # macOS does not support RLIMIT_AS


def block_network() -> None:
    def refuse(*args, **kwargs):
        raise PermissionError("Network access is disabled in the sandbox.")
    socket.socket = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def safe_import_factory(allowed):
    real_import = __import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in allowed:
            raise ImportError(f"Importing '{name}' is not allowed in the sandbox.")
        return real_import(name, globals, locals, fromlist, level)
    return safe_import


def to_json_result(value):
    import pandas as pd
    import plotly.graph_objects as go

    if isinstance(value, go.Figure):
        return {"type": "figure", "value": value.to_json()}
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        shown = value.head(MAX_TABLE_ROWS).reset_index()
        return {"type": "table", "value": shown.to_json(orient="split", date_format="iso", default_handler=str),
                "rows": len(value)}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return {"type": "value", "value": value}
    if hasattr(value, "item"):  # numpy number
        return {"type": "value", "value": value.item()}
    return {"type": "value", "value": str(value)[:2000]}


def main() -> None:
    work = Path(sys.argv[1])
    # Load the libraries and the data BEFORE locking things down.
    import math
    import statistics

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.read_parquet(work / "data.parquet")
    code = (work / "code.py").read_text()

    apply_limits()
    block_network()

    allowed = {"pandas", "numpy", "plotly", "plotly.express", "plotly.graph_objects",
            "math", "statistics", "datetime"}
    import builtins
    safe_builtins = {k: v for k, v in vars(builtins).items()
                    if k not in {"open", "eval", "exec", "compile", "input", "breakpoint",
                                "globals", "locals", "vars", "getattr", "setattr", "delattr",
                                "exit", "quit", "help", "memoryview"}}
    safe_builtins["__import__"] = safe_import_factory(allowed)

    namespace = {"__builtins__": safe_builtins, "df": df, "pd": pd, "np": np, "px": px, "go": go,
                "math": math, "statistics": statistics}
    try:
        exec(compile(code, "<llm_code>", "exec"), namespace)  # noqa: S102 - this IS the sandbox
        if "result" not in namespace:
            raise NameError("The code did not set 'result'.")
        print(json.dumps({"ok": True, **to_json_result(namespace["result"])}))
    except Exception as e:  # noqa: BLE001 - report any failure back to the agent
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:1000]}))


if __name__ == "__main__":
    main()