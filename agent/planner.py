"""The planner: the LLM reads the data profile and picks analyses from the tool menu.

The LLM only ever sees a compact profile (column names, types, missing %),
never the raw rows. Whatever it returns is validated before anything runs,
and if the LLM is unavailable or useless we fall back to a sensible default plan.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from agent.tools import TOOLS
from core.llm import LLMError

MAX_STEPS = 6

# which column kinds (from the profiler) each tool parameter kind accepts
ACCEPTS = {"numeric": {"numeric"}, "categorical": {"categorical", "boolean"}, "text": {"text"},
        "datetime": {"datetime"}}

class Step(BaseModel):
    tool: str
    args: dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class Plan(BaseModel):
    steps: list[Step]


class PlanResult(BaseModel):
    steps: list[Step]
    source: str                      # "llm" or "fallback"
    rejected: list[str] = Field(default_factory=list)


# ---------- prompt ----------

SYSTEM_PROMPT = """You are a senior data analyst planning a first-pass exploratory analysis.
You receive a JSON profile of a dataset (column names, kinds and quality figures, no raw data)
and a menu of analysis tools. Choose up to {max_steps} tools that would be most informative.

Rules:
- Use only tools from the menu, with exactly the argument names shown.
- Every argument value must be a column name from the profile whose kind matches the argument.
- Do not analyse ID columns (columns flagged as is_sequence).
- Do not repeat the same tool with the same arguments.
- Prefer a mix: distributions of key numeric columns, frequent categories, relationships.

Reply with JSON only, in exactly this shape:
{{"steps": [{{"tool": "histogram", "args": {{"column": "price"}}, "reason": "why this helps"}}]}}"""


def compact_profile(profile: dict) -> dict:
    """Keep only what the planner needs. No example values (data minimisation)."""
    return {
        "rows": profile["rows"],
        "columns": [
            {k: c[k] for k in ("name", "kind", "missing_pct", "unique", "is_sequence") if k in c}
            for c in profile["column_profiles"]
        ],
        "warnings": profile["warnings"],
    }


def tool_menu() -> list[dict]:
    return [{"tool": name, "args": spec["params"], "description": spec["description"]}
            for name, spec in TOOLS.items()]


def build_messages(profile: dict) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(max_steps=MAX_STEPS)
    user = json.dumps({"tools": tool_menu(), "profile": compact_profile(profile)}, indent=1)
    return system, user


# ---------- validation ----------

def validate_steps(steps: list[Step], profile: dict) -> tuple[list[Step], list[str]]:
    """Keep only steps that are safe and sensible to run; explain every rejection."""
    kinds = {c["name"]: c["kind"] for c in profile["column_profiles"]}
    ids = {c["name"] for c in profile["column_profiles"] if c.get("is_sequence")}
    kept, rejected, seen = [], [], set()

    for step in steps:
        label = f"{step.tool}({step.args})"
        if step.tool not in TOOLS:
            rejected.append(f"{label}: unknown tool")
            continue
        params = TOOLS[step.tool]["params"]
        if set(step.args) != set(params):
            rejected.append(f"{label}: expected arguments {sorted(params)}")
            continue
        problem = None
        for arg, column in step.args.items():
            if column not in kinds:
                problem = f"column '{column}' does not exist"
            elif column in ids:
                problem = f"'{column}' is an ID column"
            elif kinds[column] not in ACCEPTS[params[arg]]:
                problem = f"'{column}' is {kinds[column]}, but '{arg}' needs {params[arg]}"
            if problem:
                break
        if problem:
            rejected.append(f"{label}: {problem}")
            continue
        key = (step.tool, tuple(sorted(step.args.items())))
        if key in seen:
            rejected.append(f"{label}: duplicate")
            continue
        seen.add(key)
        kept.append(step)

    if len(kept) > MAX_STEPS:
        rejected.append(f"{len(kept) - MAX_STEPS} extra steps dropped (limit {MAX_STEPS})")
        kept = kept[:MAX_STEPS]
    return kept, rejected


# ---------- fallback ----------

def fallback_plan(profile: dict) -> list[Step]:
    """A simple rule-based plan, used when the LLM is unavailable or returns nothing usable."""
    cols = [c for c in profile["column_profiles"] if not c.get("is_sequence")]
    numeric = [c["name"] for c in cols if c["kind"] == "numeric"]
    categorical = [c["name"] for c in cols if c["kind"] in ACCEPTS["categorical"]]
    text = [c["name"] for c in cols if c["kind"] == "text"]
    dates = [c["name"] for c in cols if c["kind"] == "datetime"]

    steps = [Step(tool="histogram", args={"column": n}, reason="Distribution") for n in numeric[:2]]
    steps += [Step(tool="value_counts", args={"column": c}, reason="Frequent values") for c in categorical[:1]]
    if len(numeric) >= 2:
        steps.append(Step(tool="correlation_heatmap", reason="Relationships between numeric columns"))
    if numeric and categorical:
        steps.append(Step(tool="box_by_category", args={"numeric": numeric[0], "category": categorical[0]},
                        reason="Compare groups"))
    if dates and numeric:
        steps.append(Step(tool="trend_over_time", args={"date": dates[0], "value": numeric[0]},
                        reason="Change over time"))
    steps += [Step(tool="text_length", args={"column": t}, reason="Length of free text") for t in text[:1]]
    return steps[:MAX_STEPS]


# ---------- main entry point ----------

def make_plan(profile: dict, llm) -> PlanResult:
    """Ask the LLM for a plan, validate it, and fall back to rules if needed."""
    try:
        raw = llm.chat_json(*build_messages(profile))
        steps, rejected = validate_steps(Plan.model_validate(raw).steps, profile)
    except (LLMError, ValidationError) as e:
        return PlanResult(steps=fallback_plan(profile), source="fallback",
                        rejected=[f"LLM plan unusable: {str(e)[:200]}"])
    if not steps:
        return PlanResult(steps=fallback_plan(profile), source="fallback",
                        rejected=rejected + ["No valid steps from the LLM"])
    return PlanResult(steps=steps, source="llm", rejected=rejected)