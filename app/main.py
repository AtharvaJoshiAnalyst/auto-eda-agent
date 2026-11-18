import json

import pandas as pd
import streamlit as st

from agent.planner import PlanResult, fallback_plan, make_plan
from agent.tools import ToolError, run_tool
from core.llm import LLMClient, LLMError
from core.loaders import SUPPORTED_TYPES, UnsupportedFileError, load_file
from core.profiler import profile_dataframe
from core.router import choose_path

st.set_page_config(page_title="Auto-EDA Agent", page_icon="📊", layout="wide")
st.title("📊 Auto-EDA Agent")
st.caption("Upload a CSV, Excel, PDF or Word file and get an instant first-pass analysis.")


# ---------- cached helpers ----------

@st.cache_data(show_spinner="Reading file...")
def cached_load(name: str, data: bytes):
    return load_file(name, data)


@st.cache_resource
def get_llm():
    """Create the LLM client once. Returns None if it isn't configured."""
    try:
        return LLMClient()
    except LLMError:
        return None


@st.cache_data(show_spinner="The agent is planning the analysis...")
def cached_plan(profile_json: str, use_ai: bool, _llm):
    # Cache key = profile + toggle. The leading underscore tells Streamlit not to hash
    # the client, so the same profile is only sent to the LLM once.
    profile = json.loads(profile_json)
    if not use_ai or _llm is None:
        return PlanResult(steps=fallback_plan(profile), source="fallback", rejected=["No LLM configured"])
    return make_plan(profile, _llm)


# ---------- sidebar ----------

llm = get_llm()
with st.sidebar:
    st.header("Agent settings")
    use_ai = st.toggle("Use AI planner", value=llm is not None, disabled=llm is None)
    if llm:
        st.caption(f"Provider: **{llm.config.provider}**  \nModel: `{llm.config.model}`")
    else:
        st.warning("No LLM configured, so the rule-based plan is used.")
    st.caption("The AI sees column names, types and quality figures only, never your data values.")


# ---------- page sections ----------

def show_profile(df: pd.DataFrame, profile: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Duplicate rows", profile["duplicate_rows"])
    c4.metric("Memory (MB)", profile["memory_mb"])

    if profile["warnings"]:
        with st.expander(f"⚠️ {len(profile['warnings'])} data-quality warnings", expanded=True):
            for w in profile["warnings"]:
                st.write(f"- {w}")

    with st.expander("Data preview and column profile"):
        st.dataframe(df.head(20))
        # columns mix numbers and text (e.g. "min" is a date for date columns), so show everything as text
        st.dataframe(pd.DataFrame(profile["column_profiles"]).astype(str))


def show_analysis(df: pd.DataFrame, profile: dict) -> None:
    st.markdown("#### 🤖 Agent analysis")
    plan = cached_plan(json.dumps(profile), use_ai, llm)

    label = "AI planner" if plan.source == "llm" else "rule-based fallback"
    st.caption(f"Plan from the **{label}** · {len(plan.steps)} analyses")
    if plan.rejected:
        with st.expander(f"🛡️ Validation notes ({len(plan.rejected)})"):
            for r in plan.rejected:
                st.write(f"- {r}")

    cols = st.columns(2)
    for i, step in enumerate(plan.steps):
        with cols[i % 2]:
            try:
                result = run_tool(df, step.tool, step.args)
            except ToolError as e:
                st.warning(f"{step.tool} skipped: {e}")
                continue
            st.plotly_chart(result.figure, use_container_width=True)
            st.caption(f"**Why:** {step.reason}")
            with st.expander("Key numbers"):
                st.json(result.facts)


# ---------- main flow ----------

uploaded = st.file_uploader("Upload a file", type=SUPPORTED_TYPES)

if uploaded:
    try:
        loaded = cached_load(uploaded.name, uploaded.getvalue())
    except UnsupportedFileError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001 - show a message instead of crashing on corrupt files
        st.error(f"Could not read this file: {e}")
        st.exception(e)
        st.stop()

    path = choose_path(loaded)
    st.info(f"File type: **{loaded.file_type.upper()}** · Analysis path: **{path}** · "
            f"Tables found: **{len(loaded.tables)}**")
    for w in loaded.warnings:
        st.warning(w)

    for i, df in enumerate(loaded.tables, start=1):
        source = df.attrs.get("source", "")
        st.subheader(f"Table {i}" + (f" ({source})" if source else ""))
        profile = profile_dataframe(df)
        show_profile(df, profile)
        show_analysis(df, profile)
        st.divider()

    if path in ("text", "both"):
        st.subheader("Document text")
        st.caption(f"{len(loaded.text.split()):,} words. Summary and Q&A arrive on Day 5.")
        st.text_area("Preview", loaded.text[:3000], height=250)