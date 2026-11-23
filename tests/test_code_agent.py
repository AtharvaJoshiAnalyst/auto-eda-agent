"""The code agent's retry loop, tested with a scripted fake LLM (no real model needed)."""

import json

import pandas as pd
import pytest

from agent.code_agent import MAX_ATTEMPTS, answer_question
from core.llm import LLMError
from core.profiler import profile_dataframe


class ScriptedLLM:
    """Returns pre-written replies in order and records every prompt it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat_json(self, system: str, user: str) -> dict:
        self.prompts.append(json.loads(user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def code(text):
    return {"code": text, "explanation": "test"}


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South", "North", "East"] * 5,
                         "revenue": [100.0, 80.0, 120.0, 50.0] * 5,
                        "comment": [f"customer note number {i}" for i in range(20)]})


@pytest.fixture
def profile(df):
    return profile_dataframe(df)


def test_answers_on_first_try(df, profile):
    llm = ScriptedLLM([code("result = df.groupby('region')['revenue'].sum()")])
    answer = answer_question("Revenue by region?", df, profile, llm)
    assert answer.ok and answer.attempts == 1
    assert answer.result.kind == "table"


def test_fixes_its_own_error_on_retry(df, profile):
    llm = ScriptedLLM([code("result = df['Revenue'].sum()"),      # wrong capital letter
                    code("result = df['revenue'].sum()")])
    answer = answer_question("Total revenue?", df, profile, llm)
    assert answer.ok and answer.attempts == 2 and answer.result.value == 1750
    # the second prompt contained the first error, so the LLM could fix it
    assert "KeyError" in llm.prompts[1]["previous_error"]


def test_blocked_code_is_sent_back_too(df, profile):
    llm = ScriptedLLM([code("import os\nresult = os.getcwd()"), code("result = len(df)")])
    answer = answer_question("How many rows?", df, profile, llm)
    assert answer.ok and answer.attempts == 2
    assert "not allowed" in llm.prompts[1]["previous_error"]


def test_gives_up_after_max_attempts(df, profile):
    llm = ScriptedLLM([code("result = df['nope'].sum()")] * MAX_ATTEMPTS)
    answer = answer_question("?", df, profile, llm)
    assert not answer.ok
    assert answer.attempts == MAX_ATTEMPTS and len(answer.history) == MAX_ATTEMPTS
    assert "KeyError" in answer.error


def test_llm_unavailable_fails_gracefully(df, profile):
    answer = answer_question("?", df, profile, ScriptedLLM([LLMError("connection refused")]))
    assert not answer.ok and "LLM unavailable" in answer.error


def test_prompt_has_category_values_but_no_free_text(df, profile):
    llm = ScriptedLLM([code("result = 1")])
    answer_question("?", df, profile, llm)
    prompt = json.dumps(llm.prompts[0])
    assert "North" in prompt                       # category values help the LLM filter correctly
    assert "customer note number" not in prompt    # free text never leaves the app
    
def test_line_breaks_sent_as_text_are_repaired(df, profile):
    # a small model once sent "\n" as two characters, so every attempt was a syntax error
    escaped = "total = df['revenue'].sum()\\nresult = total / 2"
    llm = ScriptedLLM([code(escaped)])
    answer = answer_question("Half the revenue?", df, profile, llm)
    assert answer.ok and answer.attempts == 1 and answer.result.value == 875