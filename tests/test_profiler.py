import json

import pandas as pd

from core.profiler import profile_dataframe


def test_basic_counts(sample_df):
    p = profile_dataframe(sample_df)
    assert p["rows"] == 6
    assert p["columns"] == 3
    age = next(c for c in p["column_profiles"] if c["name"] == "age")
    assert age["kind"] == "numeric"
    assert age["missing"] == 1


def test_flags_constant_column_and_outliers(sample_df):
    warnings = " ".join(profile_dataframe(sample_df)["warnings"])
    assert "'constant'" in warnings  # the column with a single value
    assert "'age'" in warnings       # 300 is an outlier


def test_flags_index_column():
    df = pd.DataFrame({"Unnamed: 0": [0, 1, 2, 3], "score": [5, 5, 3, 4]})
    warnings = " ".join(profile_dataframe(df)["warnings"])
    assert "'Unnamed: 0'" in warnings
    assert "'score'" not in warnings


def test_profile_is_json_serialisable(sample_df):
    # The profile is what the LLM will receive on Day 3, so it must be plain JSON.
    json.dumps(profile_dataframe(sample_df))