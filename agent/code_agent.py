"""The code agent: answers a user's question by writing, checking and running code.

A small LangGraph graph:

    START -> write_code -> run_code -> (worked?) -> END
                ^                         |
                +------ no, try again ----+   (at most MAX_ATTEMPTS in total)

When code fails, the exact error goes back to the LLM so it can fix its own code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.sandbox import SandboxResult, run_code
from core.llm import LLMError

MAX_ATTEMPTS = 3  # the first try plus two retries
MAX_LABEL_CHARS = 30  # longer "categories" are probably free text

SYSTEM_PROMPT = """You are a careful data analyst who answers questions by writing short Python code.

The data is already loaded in a pandas DataFrame called `df`.
These are already available: pd (pandas), np (numpy), px (plotly.express), go (plotly.graph_objects).

Rules:
- Store the answer in a variable called `result`: a DataFrame, a Series, a single number,
or a plotly figure (use a figure when a chart answers the question better).
- Use only the columns listed. Their exact names matter.
-- For averages, totals or counts by group, calculate them first (e.g. df.groupby(...).mean())
and chart the result. Never chart raw rows when the question asks for a summary.
- Do not read or write files, use the network, or import anything other than pandas, numpy,
plotly, math, statistics or datetime.
- Do not call fig.show(), and do not use try/except or names starting with an underscore.
- Keep the code short and readable.

Reply with JSON only: {"code": "<python code>", "explanation": "<one sentence for a business user>"}"""


class AgentState(TypedDict):
    question: str
    columns: list[dict]
    code: str
    explanation: str
    error: str
    attempts: int
    history: list[dict]
    result: SandboxResult | None


@dataclass
class CodeAnswer:
    ok: bool
    code: str = ""
    explanation: str = ""
    result: SandboxResult | None = None
    attempts: int = 0
    history: list[dict] = field(default_factory=list)
    error: str = ""


def describe_columns(profile: dict) -> list[dict]:
    """Column names and kinds, plus example values of genuine category columns only.

    Values are shared only when they repeat (at most half as many distinct values as rows)
    and are short labels - so free text, names and IDs never leave the app, even in small
    tables where the profiler may call a text column "categorical".
    """
    columns = []
    for c in profile["column_profiles"]:
        info = {"name": c["name"], "kind": c["kind"]}
        values = list(c.get("top_values", {}))[:10]
        repeats = c["unique"] <= 0.5 * profile["rows"]
        short = all(len(v) <= MAX_LABEL_CHARS for v in values)
        if c["kind"] == "categorical" and repeats and short:
            info["values"] = values
        columns.append(info)
    return columns

def clean_code(code: str) -> str:
    """Fix a common small-model mistake: line breaks sent as the two characters \\ and n."""
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\n", "\n")
    return code.strip()

def build_user_message(state: AgentState) -> str:
    message = {"question": state["question"], "columns": state["columns"]}
    if state["error"]:
        message["previous_code"] = state["code"]
        message["previous_error"] = state["error"]
        message["instruction"] = "The previous code failed. Fix it and return the full corrected code."
    return json.dumps(message, indent=1)


def build_graph(df: pd.DataFrame, llm):
    def write_code(state: AgentState) -> dict:
        try:
            reply = llm.chat_json(SYSTEM_PROMPT, build_user_message(state))
        except LLMError as e:
            return {"error": f"LLM unavailable: {e}", "attempts": MAX_ATTEMPTS, "code": ""}
        return {"code": clean_code(str(reply.get("code", ""))),
                "explanation": str(reply.get("explanation", "")), "attempts": state["attempts"] + 1}

    def execute(state: AgentState) -> dict:
        if not state["code"]:
            return {"result": None}  # nothing to run (e.g. the LLM was unavailable)
        result = run_code(state["code"], df)
        attempt = {"attempt": state["attempts"], "code": state["code"],
                "ok": result.ok, "error": result.error}
        return {"result": result, "error": "" if result.ok else result.error,
                "history": state["history"] + [attempt]}

    def next_step(state: AgentState) -> str:
        if state["result"] is not None and state["result"].ok:
            return "done"
        return "retry" if state["attempts"] < MAX_ATTEMPTS else "done"

    graph = StateGraph(AgentState)
    graph.add_node("write_code", write_code)
    graph.add_node("run_code", execute)
    graph.add_edge(START, "write_code")
    graph.add_edge("write_code", "run_code")
    graph.add_conditional_edges("run_code", next_step, {"retry": "write_code", "done": END})
    return graph.compile()


def answer_question(question: str, df: pd.DataFrame, profile: dict, llm) -> CodeAnswer:
    graph = build_graph(df, llm)
    start: AgentState = {"question": question, "columns": describe_columns(profile), "code": "",
                        "explanation": "", "error": "", "attempts": 0, "history": [], "result": None}
    final = graph.invoke(start)
    result = final["result"]
    ok = result is not None and result.ok
    return CodeAnswer(ok=ok, code=final["code"], explanation=final["explanation"], result=result,
                    attempts=final["attempts"], history=final["history"],
                    error="" if ok else final["error"] or "No answer was produced.")