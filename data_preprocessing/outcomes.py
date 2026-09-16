"""Laboratory, intervention, and composite postoperative outcomes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _measure_groups(
    events: pd.DataFrame,
    item_name: str,
    *,
    value_scale: float = 1.0,
) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    subset = events.loc[events["item_name"].eq(item_name), ["subject_id", "chart_time", "value"]].copy()
    subset["chart_time"] = _as_numeric(subset, "chart_time")
    subset["value"] = _as_numeric(subset, "value") * value_scale
    subset = subset.dropna(subset=["subject_id", "chart_time", "value"])
    groups: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id, group in subset.sort_values("chart_time", kind="stable").groupby("subject_id", sort=False):
        groups[subject_id] = (group["chart_time"].to_numpy(), group["value"].to_numpy())
    return groups


def _window_endpoint(
    operations: pd.DataFrame,
    groups: dict[object, tuple[np.ndarray, np.ndarray]],
    *,
    lower_col: str | None,
    upper_col: str | None,
    endpoint: str,
    closed: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """Return event time and value at one endpoint of each operation window."""

    event_times = np.full(len(operations), np.nan)
    event_values = np.full(len(operations), np.nan)
    op_groups = operations.groupby("subject_id", sort=False).indices
    left_side = "left" if closed in {"both", "left"} else "right"
    right_side = "right" if closed in {"both", "right"} else "left"

    for subject_id, positions in op_groups.items():
        subject_events = groups.get(subject_id)
        if subject_events is None:
            continue
        times, values = subject_events
        op_rows = operations.iloc[positions]
        lower = (
            _as_numeric(op_rows, lower_col).to_numpy()
            if lower_col is not None
            else np.full(len(op_rows), -np.inf)
        )
        upper = (
            _as_numeric(op_rows, upper_col).to_numpy()
            if upper_col is not None
            else np.full(len(op_rows), np.inf)
        )
        left = np.searchsorted(times, lower, side=left_side)
        right = np.searchsorted(times, upper, side=right_side)
        valid = np.isfinite(lower) | np.isneginf(lower)
        valid &= np.isfinite(upper) | np.isposinf(upper)
        valid &= left < right
        chosen = left if endpoint == "first" else right - 1
        position_array = np.asarray(positions)
        event_times[position_array[valid]] = times[chosen[valid]]
        event_values[position_array[valid]] = values[chosen[valid]]
    return event_times, event_values


def _window_item_flags(
    operations: pd.DataFrame,
    events: pd.DataFrame,
    items: Iterable[str],
    *,
    lower_col: str,
    upper_col: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=operations.index)
    for item in items:
        groups = _measure_groups(events, item)
        _, values = _window_endpoint(
            operations,
            groups,
            lower_col=lower_col,
            upper_col=upper_col,
            endpoint="first",
        )
        output[item] = np.isfinite(values).astype(np.int8)
    return output


def _annotate_troponin(operations: pd.DataFrame, labs: pd.DataFrame) -> None:
    groups = _measure_groups(labs, "troponin_t", value_scale=1000.0)
    _, pre = _window_endpoint(
        operations,
        groups,
        lower_col="admission_time",
        upper_col="orin_time",
        endpoint="first",
    )
    _, post = _window_endpoint(
        operations,
        groups,
        lower_col="orout_time",
        upper_col="discharge_time",
        endpoint="last",
    )
    delta = post - pre
    pmi = np.where(np.isfinite(pre) & np.isfinite(post), (delta > 14).astype(float), np.nan)
    mins = np.where(
        np.isfinite(post),
        ((post >= 65) | (np.isfinite(pre) & (post >= 20) & (delta >= 5))).astype(float),
        np.nan,
    )
    operations["postop_cardiac_performance_index"] = pmi
    operations["postop_cardiac_function_index"] = mins


def _annotate_aki(operations: pd.DataFrame, labs: pd.DataFrame) -> None:
    groups = _measure_groups(labs.drop_duplicates(["subject_id", "chart_time"]), "creatinine")
    pre_time, pre = _window_endpoint(
        operations,
        groups,
        lower_col=None,
        upper_col="orin_time",
        endpoint="last",
    )
    post_time, post = _window_endpoint(
        operations,
        groups,
        lower_col="orout_time",
        upper_col=None,
        endpoint="first",
    )
    ratio = np.divide(post, pre, out=np.full(len(operations), np.nan), where=pre != 0)
    increase = post - pre
    hours = (post_time - pre_time) / 60.0
    stage = np.full(len(operations), np.nan)
    observed = np.isfinite(pre) & np.isfinite(post)
    stage[observed] = 0
    stage[observed & (((ratio >= 1.5) & (ratio < 2) & (hours <= 168)) | ((increase >= 0.3) & (hours <= 48)))] = 1
    stage[observed & (ratio >= 2) & (ratio < 3) & (hours <= 168)] = 2
    crrt = pd.to_numeric(
        operations.get("postop_continuous_renal_replacement_therapy", 0), errors="coerce"
    ).fillna(0).to_numpy()
    stage[observed & (((ratio >= 3) & (hours <= 168)) | (post >= 4) | (crrt == 1))] = 3
    operations["postop_acute_kidney_injury"] = stage
    operations["have_aki"] = np.where(np.isnan(stage), np.nan, (stage > 0).astype(float))


def _annotate_ali(operations: pd.DataFrame, labs: pd.DataFrame) -> None:
    groups = _measure_groups(labs, "alt")
    _, pre = _window_endpoint(
        operations,
        groups,
        lower_col=None,
        upper_col="opstart_time",
        endpoint="last",
    )
    _, post = _window_endpoint(
        operations,
        groups,
        lower_col="opend_time",
        upper_col=None,
        endpoint="first",
    )
    sex = pd.to_numeric(operations["sex"], errors="coerce").to_numpy()
    upper_limit = np.where(sex == 0, 35.0, 40.0)
    ratio = post / upper_limit
    observed = np.isfinite(pre) & np.isfinite(post)
    stage = np.full(len(operations), np.nan)
    stage[observed] = 0
    stage[observed & (ratio > 2) & (ratio <= 3)] = 1
    stage[observed & (ratio > 3) & (ratio <= 5)] = 2
    stage[observed & (ratio > 5)] = 4
    death = pd.to_numeric(operations["death_30d"], errors="coerce").fillna(0).to_numpy()
    stage[observed & (death == 1)] = 5
    operations["postop_acute_liver_injury"] = stage
    operations["have_ali"] = np.where(np.isnan(stage), np.nan, (stage > 0).astype(float))


def _annotate_unplanned_intubation(operations: pd.DataFrame, ward_vitals: pd.DataFrame) -> None:
    subset = ward_vitals.loc[
        ward_vitals["item_name"].eq("vent"), ["subject_id", "chart_time", "value"]
    ].copy()
    subset["chart_time"] = _as_numeric(subset, "chart_time")
    subset["value"] = _as_numeric(subset, "value")
    event_groups = {
        subject_id: group.sort_values("chart_time", kind="stable")
        for subject_id, group in subset.dropna(subset=["chart_time", "value"]).groupby(
            "subject_id", sort=False
        )
    }
    flags = np.zeros(len(operations), dtype=np.int8)
    first_times = np.full(len(operations), np.nan)
    for position, row in enumerate(operations.itertuples(index=False)):
        group = event_groups.get(row.subject_id)
        if group is None:
            continue
        times = group["chart_time"].to_numpy()
        values = group["value"].to_numpy()
        left = np.searchsorted(times, row.orout_time, side="right")
        right = np.searchsorted(times, row.discharge_time, side="left")
        if right - left < 2:
            continue
        changes = np.flatnonzero(np.diff(values[left:right]) == 1)
        if len(changes):
            flags[position] = 1
            first_times[position] = times[left + changes[0] + 1]
    operations["postop_unplanned_intubation"] = flags
    operations["postop_first_intubation_time"] = first_times


def _add_composite_outcomes(operations: pd.DataFrame) -> None:
    pulmonary = [
        "postop_respiratory_infection",
        "postop_acute_respiratory_distress_syndrome",
        "postop_pulmonary_edema",
        "postop_pleural_effusion",
        "postop_pneumothorax",
        "postop_unplanned_intubation",
    ]
    cardiac = [
        "postop_ischemic_heart_disease",
        "postop_pulmonary_heart_and_pulmonary_circulation_disease",
        "postop_pericardial_disease",
        "postop_valvular_heart_disease",
        "postop_valvular_endocardial_disease",
        "postop_chronic_rheumatic_heart_disease",
        "postop_cardiomyopathy",
        "postop_arrhythmia_and_conduction_disorder",
        "postop_heart_failure",
        "postop_cardiac_performance_index",
        "postop_cardiac_function_index",
    ]
    stroke = [
        "postop_hemorrhagic_cerebrovascular_disease",
        "postop_ischemic_stroke",
    ]
    operations["postop_lung_complications"] = operations.reindex(columns=pulmonary).fillna(0).gt(0).any(axis=1).astype(np.int8)
    operations["postop_cardiac_complications"] = operations.reindex(columns=cardiac).fillna(0).gt(0).any(axis=1).astype(np.int8)
    operations["postop_stroke"] = operations.reindex(columns=stroke).fillna(0).gt(0).any(axis=1).astype(np.int8)
    operations["postop_maces"] = (
        operations["postop_cardiac_complications"].eq(1) | operations["postop_stroke"].eq(1)
    ).astype(np.int8)


def annotate_outcomes(
    operations: pd.DataFrame,
    labs: pd.DataFrame,
    ward_vitals: pd.DataFrame,
) -> pd.DataFrame:
    """Add the project outcomes while preserving ICD annotations."""

    result = operations.reset_index(drop=True).copy()
    required_lab = {"subject_id", "chart_time", "item_name", "value"}
    required_ward = required_lab
    if missing := required_lab.difference(labs.columns):
        raise ValueError(f"labs is missing columns: {sorted(missing)}")
    if missing := required_ward.difference(ward_vitals.columns):
        raise ValueError(f"ward_vitals is missing columns: {sorted(missing)}")

    pre = _window_item_flags(
        result,
        ward_vitals,
        ("crrt", "ecmo", "vent"),
        lower_col="admission_time",
        upper_col="orin_time",
    )
    post = _window_item_flags(
        result,
        ward_vitals,
        ("crrt", "ecmo", "vent"),
        lower_col="orout_time",
        upper_col="discharge_time",
    )
    for item in ("crrt", "ecmo", "vent"):
        result[f"preop_{item}"] = pre[item].to_numpy()
    result["postop_continuous_renal_replacement_therapy"] = post["crrt"].to_numpy()
    result["postop_extracorporeal_membrane_oxygenation"] = post["ecmo"].to_numpy()
    result["postop_ventilation"] = post["vent"].to_numpy()

    _annotate_troponin(result, labs)
    _annotate_aki(result, labs)
    _annotate_ali(result, labs)
    _annotate_unplanned_intubation(result, ward_vitals)
    _add_composite_outcomes(result)
    return result
