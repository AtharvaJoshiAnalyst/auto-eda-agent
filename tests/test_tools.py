import math

import pandas as pd
import pytest

from agent.tools import TOOLS, ToolError, run_tool


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5], "group": list("aabbb"),
                        "notes": ["ok", "very good indeed", "bad", "fine thanks", "great"],
                        "day": pd.date_range("2026-01-01", periods=5, freq="D")})

def test_each_tool_returns_a_chart_and_facts(df):
    calls = [("histogram", {"column": "x"}), ("value_counts", {"column": "group"}),
            ("correlation_heatmap", {}), ("box_by_category", {"numeric": "x", "category": "group"}),
            ("scatter", {"x": "x", "y": "y"}), ("text_length", {"column": "notes"}),
            ("scatter", {"x": "x", "y": "y"}), ("text_length", {"column": "notes"}),
            ("trend_over_time", {"date": "day", "value": "x"})]
    for name, args in calls:
        result = run_tool(df, name, args)
        assert result.figure is not None and result.facts, name


def test_correlation_pairs_have_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 9]})  # only one pair
    pairs = run_tool(df, "correlation_heatmap", {}).facts["strongest_pairs"]
    assert len(pairs) == 1
    assert not any(math.isnan(v) for v in pairs.values())


@pytest.mark.parametrize("name, args", [
    ("hack", {}),                            # unknown tool
    ("histogram", {"column": "nope"}),       # column doesn't exist
    ("histogram", {"column": "group"}),      # wrong column type
    ("histogram", {"col": "x"}),
    ("trend_over_time", {"date": "group", "value": "x"}),# not a date column
])
def test_bad_requests_are_blocked(df, name, args):
    with pytest.raises(ToolError):
        run_tool(df, name, args)
        
def test_scatter_handles_missing_values():
    # regression test: the sample size was based on all rows, not rows left after dropna()
    df = pd.DataFrame({"a": [1, 2, 3, 4, None, None], "b": [2, None, 6, 8, 10, 12]})
    result = run_tool(df, "scatter", {"x": "a", "y": "b"})
    assert result.figure is not None


def test_unexpected_errors_become_tool_errors(monkeypatch):
    # if a tool crashes for a reason we didn't foresee, the app gets a ToolError, not a crash
    def broken(df, column):
        raise ValueError("something unexpected")

    monkeypatch.setitem(TOOLS["histogram"], "fn", broken)
    with pytest.raises(ToolError, match="something unexpected"):
        run_tool(pd.DataFrame({"x": [1, 2]}), "histogram", {"column": "x"})