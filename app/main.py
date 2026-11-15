import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto-EDA Agent", page_icon="📊", layout="wide")
st.title("📊 Auto-EDA Agent")
st.caption("Upload a file and get an instant first-pass analysis.")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20))

    st.subheader("Missing values per column")
    st.bar_chart(df.isna().sum())
