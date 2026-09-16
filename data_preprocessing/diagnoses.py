"""Efficient ICD annotation for surgical episodes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .icd import ICD_RULES, ICDRule, normalize_icd10, series_matches

REQUIRED_OPERATION_COLUMNS = {"op_id", "subject_id", "orin_time", "discharge_time"}
REQUIRED_DIAGNOSIS_COLUMNS = {"subject_id", "chart_time", "icd10_cm"}


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")


def prepare_diagnoses(
    diagnoses: pd.DataFrame,
    *,
    time_divisor: float = 60.0,
) -> pd.DataFrame:
    """Normalize ICD codes and put diagnosis time on the operation time scale.

    INSPIRE diagnosis ``chart_time`` is stored in minutes while the operation
    table uses hours. The original project divided diagnosis times by 60; that
    conversion remains the default and can be changed from the command line.
    """

    _require_columns(diagnoses, REQUIRED_DIAGNOSIS_COLUMNS, "diagnoses")
    if time_divisor <= 0:
        raise ValueError("time_divisor must be positive")
    result = diagnoses.loc[:, ["subject_id", "chart_time", "icd10_cm"]].copy()
    result["chart_time"] = pd.to_numeric(result["chart_time"], errors="coerce") / time_divisor
    result["icd_category"] = result["icd10_cm"].map(normalize_icd10).astype("string")
    result = result.dropna(subset=["subject_id", "icd_category"])
    return result.sort_values(["subject_id", "chart_time"], kind="stable").reset_index(drop=True)


def _window_has_diagnosis(
    operations: pd.DataFrame,
    diagnoses: pd.DataFrame,
    *,
    lower_col: str,
    upper_col: str,
) -> np.ndarray:
    """Test whether each operation has a diagnosis in an open time interval."""

    flags = np.zeros(len(operations), dtype=np.int8)
    if diagnoses.empty:
        return flags

    op_groups = operations.groupby("subject_id", sort=False).indices
    for subject_id, group in diagnoses.groupby("subject_id", sort=False):
        op_positions = op_groups.get(subject_id)
        if op_positions is None:
            continue
        times = np.sort(pd.to_numeric(group["chart_time"], errors="coerce").dropna().to_numpy())
        if not len(times):
            continue
        lower = pd.to_numeric(operations.iloc[op_positions][lower_col], errors="coerce").to_numpy()
        upper = pd.to_numeric(operations.iloc[op_positions][upper_col], errors="coerce").to_numpy()
        valid = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        left = np.searchsorted(times, lower, side="right")
        right = np.searchsorted(times, upper, side="left")
        flags[np.asarray(op_positions)[valid]] = (left[valid] < right[valid]).astype(np.int8)
    return flags


def annotate_diagnoses(
    operations: pd.DataFrame,
    diagnoses: pd.DataFrame,
    *,
    rules: Iterable[ICDRule] = ICD_RULES,
) -> pd.DataFrame:
    """Add preoperative comorbidity and postoperative diagnosis flags.

    Normal preoperative codes require ``chart_time < orin_time``. Codes in a
    rule's ``timeless`` subset count as preoperative regardless of diagnosis
    time. Postoperative flags use ``orin_time < chart_time < discharge_time``.
    """

    _require_columns(operations, REQUIRED_OPERATION_COLUMNS, "operations")
    _require_columns(
        diagnoses,
        REQUIRED_DIAGNOSIS_COLUMNS | {"icd_category"},
        "prepared diagnoses",
    )
    result = operations.reset_index(drop=True).copy()
    subject_ids = result["subject_id"]
    orin_time = pd.to_numeric(result["orin_time"], errors="coerce")

    for rule in rules:
        matches = series_matches(diagnoses["icd_category"], rule.ranges)
        rule_diagnoses = diagnoses.loc[matches]

        earliest = rule_diagnoses.groupby("subject_id", sort=False)["chart_time"].min()
        earliest_for_operation = subject_ids.map(earliest)
        preop = earliest_for_operation.lt(orin_time).fillna(False)

        if rule.timeless:
            timeless_subjects = diagnoses.loc[
                series_matches(diagnoses["icd_category"], rule.timeless), "subject_id"
            ].unique()
            preop |= subject_ids.isin(timeless_subjects)

        result[f"preop_{rule.name}"] = preop.astype(np.int8)
        result[f"postop_{rule.name}"] = _window_has_diagnosis(
            result,
            rule_diagnoses,
            lower_col="orin_time",
            upper_col="discharge_time",
        )

    return result
