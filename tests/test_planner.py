import json

import pandas as pd
import pytest

from agent.planner import MAX_STEPS, build_messages, make_plan
from core.llm import LLMError
from core.profiler import profile_dataframe


class FakeLLM:
    """Stands in for the real LLM: returns a fixed reply, or raises an error."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def chat_json(self, system: str, user: str) -> dict:
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def profile():
    df = pd.DataFrame({
        "row_id": range(8),
        "price": [10, 12, 9, 30, 11, 10, 13, 12],
        "units": [1, 3, 2, 8, 1, 2, 3, 2],
        "region": ["N", "S", "N", "E", "N", "S", "E", "N"],
        "comment": ["good", "bad item", "ok", "great value here", "meh", "fine", "love it", "no"],
    })
    return profile_dataframe(df)


def step(tool, **args):
    return {"tool": tool, "args": args, "reason": "test"}


def test_valid_plan_is_kept(profile):
    llm = FakeLLM({"steps": [step("histogram", column="price"), step("value_counts", column="region")]})
    result = make_plan(profile, llm)
    assert result.source == "llm"
    assert [s.tool for s in result.steps] == ["histogram", "value_counts"]
    assert result.rejected == []


@pytest.mark.parametrize("bad_step, reason", [
    (step("delete_everything"), "unknown tool"),
    (step("histogram", column="profit"), "does not exist"),       # hallucinated column
    (step("histogram", column="region"), "needs numeric"),        # wrong kind
    (step("histogram", column="row_id"), "ID column"),            # ID columns are off limits
    (step("histogram", col="price"), "expected arguments"),       # wrong argument name
])
def test_bad_steps_are_rejected_but_good_ones_survive(profile, bad_step, reason):
    llm = FakeLLM({"steps": [bad_step, step("histogram", column="price")]})
    result = make_plan(profile, llm)
    assert [s.args for s in result.steps] == [{"column": "price"}]
    assert reason in result.rejected[0]


def test_duplicates_and_too_many_steps_are_trimmed(profile):
    many = [step("histogram", column="price")] * 2 + [
        step("histogram", column="units"), step("value_counts", column="region"),
        step("correlation_heatmap"), step("text_length", column="comment"),
        step("scatter", x="price", y="units"), step("box_by_category", numeric="price", category="region"),
    ]
    result = make_plan(profile, FakeLLM({"steps": many}))
    assert len(result.steps) == MAX_STEPS
    assert any("duplicate" in r for r in result.rejected)


@pytest.mark.parametrize("llm", [
    FakeLLM(error=LLMError("connection refused")),     # LLM down
    FakeLLM({"wrong_key": []}),                        # wrong JSON shape
    FakeLLM({"steps": [step("histogram", column="nope")]}),  # nothing valid
])
def test_falls_back_to_rules_when_llm_fails(profile, llm):
    result = make_plan(profile, llm)
    assert result.source == "fallback"
    assert result.steps                     # the user still gets an analysis
    assert all(s.args.get("column") != "row_id" for s in result.steps)


def test_prompt_contains_no_raw_values(profile):
    _system, user = build_messages(profile)
    assert "great value here" not in user   # data values never reach the LLM
    assert "comment" in user                # column names do
    json.loads(user)