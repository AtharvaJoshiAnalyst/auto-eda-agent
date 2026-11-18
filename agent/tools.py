""" Safe pre written analysis functions for the agent to use. """



from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


class ToolError(ValueError):
    """ Raised when the tool is given bad inputs. """
    
@dataclass
class ToolResult:
    text : str
    figure : Figure | None = None
    facts : dict | None = field(default_factory=dict)
    

# ------------------ input checks ----

def _need_column(df: pd.DataFrame, column:str) -> pd.Series:
    if column not in df.columns:
        raise ToolError(f"Column '{column}' not found in the dataframe.")
    return df[column]


def _need_numeric(df: pd.DataFrame, column:str) -> pd.Series:
    s = _need_column(df, column)
    if not pd.api.types.is_numeric_dtype(s):
        raise ToolError(f"Column '{column}' is not numeric.")
    return s

# ----- tools analysis ------


def histogram(df: pd.DataFrame, column: str) -> ToolResult:
    s = _need_numeric(df, column).dropna()
    fig = px.histogram(s, x=column, nbins=40, title=f"Distribution of {column}")
    facts = {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
            "skew": round(float(s.skew()), 3)}
    return ToolResult(f"Distribution of {column}", fig, facts)


def value_counts(df: pd.DataFrame, column: str) -> ToolResult:
    counts = _need_column(df, column).value_counts().head(15)
    fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": column, "y": "count"},
                title=f"Most common values of {column}")
    facts = {"top_values": {str(k): int(v) for k, v in counts.head(5).items()}}
    return ToolResult(f"Most common values of {column}", fig, facts)


def correlation_heatmap(df: pd.DataFrame) -> ToolResult:
    numeric = df.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise ToolError("Need at least two numeric columns for correlations.")
    corr = numeric.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation between numeric columns")
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)  # each pair once, no self-pairs
    pairs = corr.where(upper).stack().dropna().sort_values(key=abs, ascending=False)
    facts = {"strongest_pairs": {f"{a} ~ {b}": float(v) for (a, b), v in pairs.head(3).items()}}
    return ToolResult("Correlation between numeric columns", fig, facts)


def box_by_category(df: pd.DataFrame, numeric: str, category: str) -> ToolResult:
    _need_numeric(df, numeric)
    cats = _need_column(df, category)
    top = cats.value_counts().head(10).index  # keep the chart readable
    data = df[cats.isin(top)]
    fig = px.box(data, x=category, y=numeric, title=f"{numeric} by {category}")
    medians = data.groupby(category)[numeric].median().round(3)
    facts = {"median_by_group": {str(k): float(v) for k, v in medians.items()}}
    return ToolResult(f"{numeric} by {category}", fig, facts)


def scatter(df: pd.DataFrame, x: str, y: str) -> ToolResult:
    _need_numeric(df, x)
    _need_numeric(df, y)
    data = df[[x, y]].dropna()
    if len(data) < 2:
        raise ToolError(f"Not enough rows with both '{x}' and '{y}' filled in.")
    sample = data.sample(min(len(data), 5000), random_state=0)
    fig = px.scatter(sample, x=x, y=y, opacity=0.5, title=f"{y} vs {x}")
    facts = {"correlation": round(float(sample[x].corr(sample[y])), 3)}
    return ToolResult(f"{y} vs {x}", fig, facts)


def text_length(df: pd.DataFrame, column: str) -> ToolResult:
    words = _need_column(df, column).dropna().astype(str).str.split().str.len()
    fig = px.histogram(words, nbins=40, title=f"Word count of {column}", labels={"value": "words"})
    facts = {"median_words": float(words.median()), "max_words": int(words.max())}
    return ToolResult(f"Word count of {column}", fig, facts)

def trend_over_time(df: pd.DataFrame, date: str, value: str) -> ToolResult:
    dates = _need_column(df, date)
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ToolError(f"Column '{date}' is not a date column.")
    _need_numeric(df, value)
    span_days = (dates.max() - dates.min()).days
    period = "D" if span_days <= 60 else "W" if span_days <= 365 else "MS"  # day, week or month
    label = {"D": "day", "W": "week", "MS": "month"}[period]
    totals = df.set_index(date)[value].resample(period).sum()
    fig = px.line(x=totals.index, y=totals.values, markers=True,
                labels={"x": date, "y": f"total {value}"}, title=f"Total {value} per {label}")
    first, last = float(totals.iloc[0]), float(totals.iloc[-1])
    facts = {"period": label, "periods": len(totals), "first": round(first, 2), "last": round(last, 2),
            "peak_period": str(totals.idxmax().date()), "peak": round(float(totals.max()), 2)}
    return ToolResult(f"Total {value} per {label}", fig, facts)

# ---------- the registry the LLM chooses from ----------

TOOLS = {
    "histogram": {"fn": histogram, "params": {"column": "numeric"},
                "description": "Distribution of one numeric column."},
    "value_counts": {"fn": value_counts, "params": {"column": "categorical"},
                    "description": "Most frequent values of a categorical column."},
    "correlation_heatmap": {"fn": correlation_heatmap, "params": {},
                            "description": "Correlations between all numeric columns."},
    "box_by_category": {"fn": box_by_category, "params": {"numeric": "numeric", "category": "categorical"},
                        "description": "Compare a numeric column across groups of a categorical column."},
    "scatter": {"fn": scatter, "params": {"x": "numeric", "y": "numeric"},
                "description": "Relationship between two numeric columns."},
    "text_length": {"fn": text_length, "params": {"column": "text"},
                    "description": "Word-count distribution of a free-text column."},
    "trend_over_time": {"fn": trend_over_time, "params": {"date": "datetime", "value": "numeric"},
                    "description": "Total of a numeric column per day, week or month."},
}


def run_tool(df: pd.DataFrame, name: str, args: dict) -> ToolResult:
    """Run a tool by name after checking the name and argument names."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'.")
    expected = set(TOOLS[name]["params"])
    if set(args) != expected:
        raise ToolError(f"Tool '{name}' expects arguments {sorted(expected)}, got {sorted(args)}.")
    try:
        return TOOLS[name]["fn"](df, **args)
    except ToolError:
        raise
    except Exception as e:  # one failing chart must not crash the whole report
        raise ToolError(f"Tool '{name}' failed: {type(e).__name__}: {e}") from e

