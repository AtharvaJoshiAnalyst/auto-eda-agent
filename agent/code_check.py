"""Layer 1 of the sandbox: inspect LLM-written code BEFORE it runs.

The code is parsed into a syntax tree (AST) and walked node by node.
Anything not clearly needed for data analysis is refused. This is a filter,
not the security boundary - the separate process in sandbox.py is.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

MAX_CODE_CHARS = 4000

# Modules the code may import. Everything else is refused (deny by default).
ALLOWED_IMPORTS = {"pandas", "numpy", "plotly", "plotly.express", "plotly.graph_objects",
                "math", "statistics", "datetime"}

# Built-in functions that can run code, touch files or reach into Python's internals.
BLOCKED_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
                "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property"}

# Attributes that lead to files, the network, other processes or other code.
BLOCKED_ATTRIBUTES = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "io", "builtins", "importlib",
    "ctypes", "pickle", "marshal", "urllib", "http", "requests", "system", "popen", "urlopen",
    "open", "eval", "exec", "query", "lib", "ctypeslib", "testing", "load", "save", "savez",
    "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "memmap", "show",
}
# pandas/numpy/plotly methods starting with these read or write files...
BLOCKED_PREFIXES = ("read_", "to_", "write_", "from_")
# ...except these, which only convert data in memory.
SAFE_EXCEPTIONS = {"to_dict", "to_list", "to_frame", "to_numpy", "to_period", "to_timestamp",
                "to_datetime", "to_numeric", "to_string", "to_timedelta", "from_dict", "from_records"}


@dataclass
class CheckResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check_code(code: str) -> CheckResult:
    if len(code) > MAX_CODE_CHARS:
        return CheckResult(False, [f"Code is longer than {MAX_CODE_CHARS} characters."])
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CheckResult(False, [f"Syntax error on line {e.lineno}: {e.msg}"])

    problems = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    problems.append(f"Line {line}: importing '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS:
                problems.append(f"Line {line}: importing from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            problems.append(f"Line {line}: '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                problems.append(f"Line {line}: private attribute '.{attr}' is not allowed.")
            elif attr in BLOCKED_ATTRIBUTES:
                problems.append(f"Line {line}: '.{attr}' is not allowed.")
            elif attr.startswith(BLOCKED_PREFIXES) and attr not in SAFE_EXCEPTIONS:
                problems.append(f"Line {line}: '.{attr}' could read or write files and is not allowed.")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete)):
            problems.append(f"Line {line}: '{type(node).__name__}' statements are not allowed.")

    assigns_result = any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
        for n in ast.walk(tree)
    )
    if not assigns_result:
        problems.append("The code must store its answer in a variable called 'result'.")

    return CheckResult(not problems, problems)