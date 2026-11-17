"""Profile a DataFrame without any LLM.

The output is a plain dictionary of schema and summary statistics.
"""

from __future__ import annotations

import pandas as pd

high_missing = 0.5 # flag columns with more than 50% missing
high_Cardinality = 0.9 # flag columns with more than 90% unique values


def profile_dataframe(df: pd.DataFrame) -> dict:
    rows = len(df)
    profile = {
        "rows": rows,
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "column_profiles": [],
        "warnings": [],
    }
    
    
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        info = {
            "name": str(col),
            "kind": _column_kind(s),
            "dtype": str(s.dtype),
            "missing": missing,
            "missing_pct": round(missing / rows * 100, 2) if rows else 0.0,
            "unique": unique,
        }
        
        if info["kind"] == "numeric":
            desc = s.describe()
            info.update({k: _round(desc[k]) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]})
            info["outliers_iqr"] = _count_outliers(s)
        elif info["kind"] in ("categorical", "text"):
            top = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        elif info["kind"] == "datetime":
            info["min"], info["max"] = str(s.min()), str(s.max())
        profile["column_profiles"].append(info)
        profile["warnings"].extend(_column_warnings(info,rows))
        
    if profile["duplicate_rows"]:
        profile["warnings"].append(f" {profile['duplicate_rows']} duplicate rows found.")
    return profile


def _column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    non_null = s.dropna()
    if len(non_null) and non_null.nunique() <= max(20, 0.05 * len(non_null)):
        return "categorical"
    return "text"  


def _count_outliers(s: pd.Series) -> int:
    """ Count the number of outliers in a numeric series using the IQR method. """
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

def _column_warnings(info: dict, rows: int) -> list[str]:
    name, warnings = info["name"], []
    if rows == 0:
        return warnings
    if info["missing"] / rows > high_missing:
        warnings.append(f"Column '{name}' has {info['missing_pct']}% missing values.")
    if info["unique"] == 1:
        warnings.append(f"Column '{name}' has a single constant value and adds no information.")
    if info["kind"] == "text" and info["unique"] / rows > high_Cardinality:
        warnings.append(f"Column '{name}' has almost all uniques ({info['unique']} maybe free text or ID values).") 
    if info["kind"] == "numeric" and info["unique"] == rows:
        warnings.append(f"'{name}' has a different value in every row; it may be an ID or row index.")
    if info.get("outliers_iqr",0)  > 0:
        warnings.append(f"Column '{name}' has {info['outliers_iqr']} potential outliers (IQR method).")
    return warnings

def _round(x):
    return None if pd.isna(x) else round(float(x), 3)
