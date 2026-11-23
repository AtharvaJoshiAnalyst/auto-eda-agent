"""Layer 2 tests: code runs in a separate, limited process; normal analysis still works."""

import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import pytest

from agent.sandbox import run_code


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["N", "S", "N", "E"], "revenue": [100.0, 80.0, 120.0, 50.0]})


# ---------- normal analysis still works ----------

def test_table_result(df):
    r = run_code("result = df.groupby('region')['revenue'].sum().sort_values()", df)
    assert r.ok and r.kind == "table"
    assert r.value["revenue"].sum() == 350


def test_chart_result(df):
    r = run_code("import plotly.express as px\nresult = px.bar(df, x='region', y='revenue')", df)
    assert r.ok and r.kind == "figure" and isinstance(r.value, go.Figure)


def test_single_value_and_lambda(df):
    r = run_code("result = df['revenue'].apply(lambda v: v * 2).max()", df)
    assert r.ok and r.value == 240


def test_runtime_error_is_reported(df):
    r = run_code("result = df['profit'].sum()", df)
    assert not r.ok and "KeyError" in r.error


# ---------- attacks stopped by the process ----------

def test_infinite_loop_times_out(df, monkeypatch):
    monkeypatch.setattr("agent.sandbox.TIMEOUT_SECONDS", 3)
    r = run_code("while True:\n    pass\nresult = 1", df)
    assert not r.ok and "Timed out" in r.error


def test_secrets_are_not_passed_to_the_sandbox(df, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    seen = {}
    real_run = subprocess.run

    def spy(*args, **kwargs):  # record the environment the sandbox receives
        seen.update(kwargs["env"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr("agent.sandbox.subprocess.run", spy)
    assert run_code("result = 1", df).ok
    assert "LLM_API_KEY" not in seen


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce memory limits")
def test_runaway_memory_is_stopped(df):
    r = run_code("result = [0] * (600 * 1024 * 1024)", df)  # about 5 GB
    assert not r.ok and "MemoryError" in r.error