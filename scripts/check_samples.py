"""Run every file in samples/ through the loader, router, profiler and tools.

Usage (from the project root):  uv run python -m scripts.check_samples
"""

from pathlib import Path

from agent.tools import ToolError, run_tool
from core.loaders import UnsupportedFileError, load_file
from core.profiler import profile_dataframe
from core.router import choose_path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def first_column(profile: dict, kind: str) -> str | None:
    return next((c["name"] for c in profile["column_profiles"] if c["kind"] == kind), None)


for path in sorted(SAMPLES.iterdir()):
    print(f"\n=== {path.name} ===")
    try:
        loaded = load_file(path.name, path.read_bytes())
    except UnsupportedFileError as e:
        print(f"  rejected: {e}")
        continue

    print(f"  path: {choose_path(loaded)} | tables: {len(loaded.tables)} | text words: {len(loaded.text.split())}")
    for w in loaded.warnings:
        print(f"  loader warning: {w}")

    for i, df in enumerate(loaded.tables, start=1):
        profile = profile_dataframe(df)
        source = df.attrs.get("source", "")
        print(f"  table {i} {source}: {profile['rows']} rows x {profile['columns']} cols, "
              f"{len(profile['warnings'])} warnings")
        for w in profile["warnings"]:
            print(f"    - {w}")

        # try one tool per column kind so every tool gets exercised
        numeric = first_column(profile, "numeric")
        categorical = first_column(profile, "categorical")
        text = first_column(profile, "text")
        calls = [("histogram", {"column": numeric}) if numeric else None,
                 ("value_counts", {"column": categorical}) if categorical else None,
                 ("text_length", {"column": text}) if text else None,
                 ("correlation_heatmap", {})]
        for call in filter(None, calls):
            try:
                print(f"    tool {call[0]}: {run_tool(df, *call).facts}")
            except ToolError as e:
                print(f"    tool {call[0]}: skipped ({e})")