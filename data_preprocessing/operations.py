"""Operation-table feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

PCS_SURGERY_SITE = {
    "08": "Head_and_Neck",
    "09": "Head_and_Neck",
    "0C": "Head_and_Neck",
    "0H": "Skin_and_Soft_Tissue",
    "0J": "Skin_and_Soft_Tissue",
    "0K": "Skin_and_Soft_Tissue",
    "0T": "Urinary_and_Reproductive_Systems",
    "0U": "Urinary_and_Reproductive_Systems",
    "0V": "Urinary_and_Reproductive_Systems",
    "10": "Urinary_and_Reproductive_Systems",
    "0D": "Gastrointestinal_System",
    "0F": "Hepatobiliary_System_and_Pancreas",
    "00": "Central_Nervous_System_and_Cranial_Nerves",
    "0G": "Endocrine_System",
    "0B": "Respiratory_System",
    "0P": "Bones_and_Joints",
    "0Q": "Bones_and_Joints",
    "0N": "Bones_and_Joints",
    "0S": "Bones_and_Joints",
    "0R": "Bones_and_Joints",
    "0L": "Bones_and_Joints",
    "0M": "Bones_and_Joints",
    "02": "Heart_and_Great_Vessels",
    "03": "Peripheral_Vascular_System",
    "04": "Peripheral_Vascular_System",
    "05": "Peripheral_Vascular_System",
    "06": "Peripheral_Vascular_System",
    "07": "Lymphatic_and_Hemic_Systems",
    "0W": "Other",
    "0Y": "Other",
    "0X": "Other",
    "01": "Other",
    "0E": "Other",
    "0A": "Other",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_operations(operations: pd.DataFrame) -> pd.DataFrame:
    """Derive operation-level variables without changing raw time columns."""

    required = {"op_id", "subject_id", "orin_time", "orout_time"}
    missing = required.difference(operations.columns)
    if missing:
        raise ValueError(f"operations is missing columns: {sorted(missing)}")

    result = operations.copy()
    height_m = _numeric(result, "height") / 100.0
    result["bmi"] = _numeric(result, "weight") / height_m.pow(2)

    duration_pairs = {
        "los": ("discharge_time", "admission_time"),
        "op_duration": ("opend_time", "opstart_time"),
        "or_duration": ("orout_time", "orin_time"),
        "an_duration": ("anend_time", "anstart_time"),
        "cpb_duration": ("cpboff_time", "cpbon_time"),
        "icu_duration": ("icuout_time", "icuin_time"),
    }
    for output, (end, start) in duration_pairs.items():
        result[output] = _numeric(result, end) - _numeric(result, start)

    if "sex" in result:
        raw_sex = result["sex"].astype("string").str.upper()
        numeric_sex = pd.to_numeric(result["sex"], errors="coerce")
        result["sex"] = np.select(
            [raw_sex.eq("F"), raw_sex.eq("M")],
            [0, 1],
            default=numeric_sex,
        )

    for column in ("op_duration", "or_duration", "an_duration", "icu_duration", "cpb_duration"):
        result[column] = result[column].fillna(0)

    result["have_icu"] = result["icu_duration"].gt(0).astype(np.int8)
    result["have_cpb"] = result["cpb_duration"].gt(0).astype(np.int8)

    death_delta = _numeric(result, "inhosp_death_time") - _numeric(result, "orout_time")
    result["death_30d"] = (
        _numeric(result, "inhosp_death_time").notna()
        & death_delta.ge(0)
        & death_delta.le(30 * 24 * 60)
    ).astype(np.int8)

    pcs = result.get("icd10_pcs", pd.Series(pd.NA, index=result.index)).astype("string")
    result["surgery_site"] = pcs.str[:2].map(PCS_SURGERY_SITE).fillna("Unknown")
    return result
