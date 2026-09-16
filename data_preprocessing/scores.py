"""SORT, Charlson, and RCRI score derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .icd import ICD_RULES, category_matches
from .outcomes import _measure_groups, _window_endpoint

TIMELESS_SPECS = tuple(spec for rule in ICD_RULES for spec in rule.timeless)


def _cci_weight(category: str) -> int:
    """Return the legacy project weight for one unique ICD category."""

    score = 0
    if category in {"I21", "I50", "I73", "F01", "I63", "M35", "K25", "K26", "K27", "E11", "J44", "K70"}:
        score = 1
    if category in {"I69", "N18"} or category_matches(category, ("C00-C99", "D00-D49")):
        score = 2
    if category in {"K74", "K75", "K76"}:
        score = 3
    if category_matches(category, ("C77-C80", "B20-B24")):
        score = 6
    return score


def _charlson_score(operations: pd.DataFrame, diagnoses: pd.DataFrame) -> np.ndarray:
    summary = (
        diagnoses.dropna(subset=["chart_time", "icd_category"])
        .groupby(["subject_id", "icd_category"], sort=False, as_index=False)["chart_time"]
        .min()
    )
    summary["weight"] = summary["icd_category"].map(_cci_weight).astype(np.int8)
    summary = summary.loc[summary["weight"].gt(0)].copy()
    summary["timeless"] = summary["icd_category"].map(
        lambda value: category_matches(value, TIMELESS_SPECS)
    )
    by_subject = {
        subject_id: group
        for subject_id, group in summary.groupby("subject_id", sort=False)
    }
    scores = np.zeros(len(operations), dtype=np.int16)
    for position, row in enumerate(operations.itertuples(index=False)):
        group = by_subject.get(row.subject_id)
        if group is None:
            continue
        eligible = group["timeless"] | group["chart_time"].lt(row.orin_time)
        scores[position] = int(group.loc[eligible, "weight"].sum())
    return scores


def _z85_history(operations: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.Series:
    z85 = diagnoses.loc[diagnoses["icd_category"].eq("Z85")]
    earliest = z85.groupby("subject_id", sort=False)["chart_time"].min()
    return operations["subject_id"].map(earliest).lt(
        pd.to_numeric(operations["orin_time"], errors="coerce")
    ).fillna(False)


def add_clinical_scores(
    operations: pd.DataFrame,
    diagnoses: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    """Add score columns using the same output names as the R pipeline."""

    result = operations.reset_index(drop=True).copy()
    asa = pd.to_numeric(result.get("asa"), errors="coerce")
    emop = pd.to_numeric(result.get("emop"), errors="coerce")
    duration = pd.to_numeric(result.get("op_duration"), errors="coerce")
    age = pd.to_numeric(result.get("age"), errors="coerce")
    pcs = result.get("icd10_pcs", pd.Series(pd.NA, index=result.index)).astype("string").str[:2]

    malignant = result.get("preop_malignant_neoplasm", 0)
    other_neoplasm = result.get("preop_other_neoplasm", 0)
    cancer_history = (
        pd.to_numeric(malignant, errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(other_neoplasm, errors="coerce").fillna(0).gt(0)
        | _z85_history(result, diagnoses)
    )
    sort_score = np.select(
        [asa.eq(3), asa.eq(4), asa.eq(5)],
        [1.411, 2.388, 4.081],
        default=0.0,
    )
    sort_score += np.where(emop.eq(1), 1.657, 0.0)
    sort_score += np.where(pcs.isin({"0D", "0F", "0G", "0W", "0X", "0Y", "02"}), 0.712, 0.0)
    sort_score += np.where(duration.gt(120), 0.381, 0.0)
    sort_score += np.where(cancer_history, 0.667, 0.0)
    sort_score += np.select([age.between(65, 79), age.ge(80)], [0.777, 1.591], default=0.0)
    result["sort_score"] = sort_score

    result["CCI_score"] = _charlson_score(result, diagnoses)

    ischemic = pd.to_numeric(result.get("preop_ischemic_heart_disease", 0), errors="coerce").fillna(0).gt(0)
    heart_failure = pd.to_numeric(result.get("preop_heart_failure", 0), errors="coerce").fillna(0).gt(0)
    stroke = (
        pd.to_numeric(result.get("preop_hemorrhagic_cerebrovascular_disease", 0), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(result.get("preop_ischemic_stroke", 0), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(result.get("preop_transient_ischemic_attack", 0), errors="coerce").fillna(0).gt(0)
    )
    diabetes = pd.to_numeric(result.get("preop_diabetes_mellitus", 0), errors="coerce").fillna(0).gt(0)

    creatinine = labs.loc[
        labs["item_name"].eq("creatinine") & pd.to_numeric(labs["value"], errors="coerce").gt(2)
    ]
    creatinine_groups = _measure_groups(creatinine, "creatinine")
    _, creatinine_value = _window_endpoint(
        result,
        creatinine_groups,
        lower_col="admission_time",
        upper_col="orin_time",
        endpoint="first",
    )
    high_risk_surgery = pcs.isin({"0D", "0F", "0G", "0T", "0U", "0V", "10", "0B", "00", "02", "03", "05"})
    result["RCRI_score"] = (
        ischemic.astype(np.int8)
        + heart_failure.astype(np.int8)
        + stroke.astype(np.int8)
        + diabetes.astype(np.int8)
        + np.isfinite(creatinine_value).astype(np.int8)
        + high_risk_surgery.astype(np.int8)
    )
    return result
