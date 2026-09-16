"""Cohort filtering and subject-level dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_and_split_operations(
    operations: pd.DataFrame,
    *,
    min_age: float = 16,
    min_operation_minutes: float = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply the legacy cohort criteria and create a leak-free subject split."""

    required = {
        "subject_id",
        "antype",
        "age",
        "op_duration",
        "or_duration",
        "cpb_duration",
        "an_duration",
        "icu_duration",
    }
    missing = required.difference(operations.columns)
    if missing:
        raise ValueError(f"operations is missing columns: {sorted(missing)}")

    result = operations.loc[
        operations["antype"].eq("General")
        & pd.to_numeric(operations["age"], errors="coerce").gt(min_age)
    ].copy()
    duration_columns = [
        "op_duration",
        "or_duration",
        "cpb_duration",
        "an_duration",
        "icu_duration",
    ]
    for column in duration_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result = result.loc[values.isna() | values.ge(0)]
    result = result.loc[pd.to_numeric(result["op_duration"], errors="coerce").ge(min_operation_minutes)]

    subjects = result["subject_id"].dropna().unique()
    shuffled = np.random.default_rng(seed).permutation(subjects)
    train_end = int(np.floor(len(shuffled) * 0.7))
    test_end = train_end + int(np.floor(len(shuffled) * 0.2))
    split_map = {
        **dict.fromkeys(shuffled[:train_end], 1),
        **dict.fromkeys(shuffled[train_end:test_end], 2),
        **dict.fromkeys(shuffled[test_end:], 3),
    }
    result["dataset"] = result["subject_id"].map(split_map).astype("Int8")
    return result.reset_index(drop=True)
