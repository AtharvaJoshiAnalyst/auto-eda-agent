import pandas as pd
import streamlit as st

from core.loaders import SUPPORTED_TYPES, UnsupportedFileError, load_file
from core.profiler import profile_dataframe
from core.router import choose_path

st.set_page_config(page_title="Auto-EDA Agent", page_icon="📊", layout="wide")
st.title("📊 Auto-EDA Agent")
st.caption("Upload a CSV, Excel, PDF or Word file and get an instant first-pass analysis.")

@st.cache_data(show_spinner="Loading file...")
def cached_load(name: str, data: bytes):
    return load_file(name, data)

uploaded = st.file_uploader("Upload a file", type=SUPPORTED_TYPES)

def show_table(df: pd.DataFrame, title:str) -> None:
    st.subheader(title)
    profile = profile_dataframe(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Duplicate rows", profile["duplicate_rows"])
    c4.metric("Memory (MB)", profile["memory_mb"])

    if profile["warnings"]:
        with st.expander(f"⚠️ {len(profile['warnings'])} data-quality warnings", expanded=True):
            for w in profile["warnings"]:
                st.write(f"- {w}")

    st.dataframe(df.head(20))
    st.markdown("**Column profile**")
    st.dataframe(pd.DataFrame(profile["column_profiles"]))

if uploaded:
    try:
        loaded = cached_load(uploaded.name, uploaded.getvalue())
    except UnsupportedFileError as e:
        st.error(str(e))
        st.stop()
    except Exception as e: # noqa: BLE001 - show a message instead of crashing on corrupt files
        st.error(f"Could not read this file: {e}")
        st.exception(e)
        st.stop()

    path = choose_path(loaded)
    st.info(f"File type: **{loaded.file_type.upper()}** · Analysis path: **{path}** · Tables found: **{len(loaded.tables)}**")
    for w in loaded.warnings:
        st.warning(w)

    for i, df in enumerate(loaded.tables, start=1):
        source = df.attrs.get("source", "")
        show_table(df, f"Table {i}" + (f" ({source})" if source else ""))

    if path in ("text", "both"):
        st.subheader("Document text")
        st.caption(f"{len(loaded.text.split()):,} words. Summary and Q&A arrive on Day 5.")
        st.text_area("Preview", loaded.text[:3000], height=250)

