"""Generate per-operation static and time-series model inputs."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .emr import (
    Stats,
    fill_frame,
    fill_last_values,
    get_mask,
    one_hot,
    resample_long,
    stats_long,
    stats_wide,
    z_parameters,
)

STATIC_OUTCOMES = (
    "death_30d",
    "have_icu",
    "have_aki",
    "have_ali",
    "postop_lung_complications",
    "postop_cardiac_complications",
    "postop_stroke",
)


@dataclass
class GenerationContext:
    operations: pd.DataFrame
    labs_by_subject: dict[object, pd.DataFrame]
    vitals_by_operation: dict[object, pd.DataFrame]
    ward_by_subject: dict[object, pd.DataFrame]
    dictionaries: dict[str, pd.DataFrame]
    stats: dict[str, Stats]


def load_variable_dictionaries(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    required = {"lab", "vital", "static", "ward_vital"}
    if missing := required.difference(sheets):
        raise ValueError(f"Variable dictionary is missing sheets: {sorted(missing)}")
    return {name: sheets[name].copy() for name in required}


def _simple_impute_by_split(
    operations: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    result = operations.copy()
    items = [item for item in dictionary["itemid"].astype(str) if item in result]
    types = dict(zip(dictionary["itemid"].astype(str), dictionary["value_type"].astype(str)))
    for positions in result.groupby("dataset", dropna=False, sort=False).indices.values():
        position_array = np.asarray(positions)
        for item in items:
            values = result.iloc[position_array][item]
            if types[item] in {"num", "bin", "ord"}:
                numeric = pd.to_numeric(values, errors="coerce")
                fill_value = numeric.median()
                if pd.isna(fill_value):
                    fill_value = 0
                result.loc[result.index[position_array], item] = numeric.fillna(fill_value).to_numpy()
            else:
                mode = values.dropna().mode()
                fill_value = mode.iloc[0] if not mode.empty else "unknown"
                result.loc[result.index[position_array], item] = values.fillna(fill_value).to_numpy()
    return result


def build_generation_context(
    operations: pd.DataFrame,
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    ward_vitals: pd.DataFrame,
    dictionaries: dict[str, pd.DataFrame],
) -> GenerationContext:
    train_subjects = operations.loc[operations["dataset"].eq(1), "subject_id"].unique()
    train_ops = operations.loc[operations["dataset"].eq(1), "op_id"].unique()
    training_labs = labs.loc[labs["subject_id"].isin(train_subjects)]
    training_vitals = vitals.loc[vitals["op_id"].isin(train_ops)]
    training_ward = ward_vitals.loc[ward_vitals["subject_id"].isin(train_subjects)]
    static_items = [item for item in dictionaries["static"]["itemid"].astype(str) if item in operations]
    training_static = operations.loc[operations["dataset"].eq(1), static_items]

    statistics = {
        "lab": stats_long(training_labs, dictionaries["lab"]),
        "vital": stats_long(training_vitals, dictionaries["vital"]),
        "ward_vital": stats_long(training_ward, dictionaries["ward_vital"]),
        "static": stats_wide(training_static, dictionaries["static"]),
    }
    imputed_operations = _simple_impute_by_split(operations, dictionaries["static"])
    return GenerationContext(
        operations=imputed_operations,
        labs_by_subject={key: value for key, value in labs.groupby("subject_id", sort=False)},
        vitals_by_operation={key: value for key, value in vitals.groupby("op_id", sort=False)},
        ward_by_subject={key: value for key, value in ward_vitals.groupby("subject_id", sort=False)},
        dictionaries=dictionaries,
        stats=statistics,
    )


def _time_points(frame: pd.DataFrame, start: float, end: float, fallback: float) -> np.ndarray:
    times = pd.to_numeric(frame.get("chart_time", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    times = np.sort(times[(times >= start) & (times <= end)])
    return times if len(times) else np.asarray([fallback], dtype=float)


def _add_dt(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["dt"] = np.r_[0.0, np.diff(pd.to_numeric(result["time"], errors="coerce"))] / 60.0
    for column in result.columns.difference(["time"]):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _derive_blood_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("nibp_sbp", "nibp_dbp", "art_sbp", "art_dbp"):
        if column not in result:
            result[column] = np.nan
    result["nibp_sbp"] = result["nibp_sbp"].fillna(result["art_sbp"])
    result["nibp_dbp"] = result["nibp_dbp"].fillna(result["art_dbp"])
    result["art_mbp"] = (result["art_sbp"] + 2 * result["art_dbp"]) / 3
    result["nibp_mbp"] = (result["nibp_sbp"] + 2 * result["nibp_dbp"]) / 3
    return result


def _trajectory_label(values: pd.Series, baseline: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy()
    reference = pd.to_numeric(baseline, errors="coerce").mean()
    if not np.isfinite(reference):
        return np.full(len(numeric), np.nan)
    return np.where(np.isnan(numeric), np.nan, np.where(numeric < 0.8 * reference, 1, np.where(numeric < 1.2 * reference, 0, 2)))


def generate_operation_data(row: pd.Series, context: GenerationContext) -> dict[str, pd.DataFrame]:
    op_id = row["op_id"]
    subject_id = row["subject_id"]
    lab = context.labs_by_subject.get(subject_id, pd.DataFrame(columns=["chart_time", "item_name", "value"]))
    vital = context.vitals_by_operation.get(op_id, pd.DataFrame(columns=["chart_time", "item_name", "value"]))
    ward = context.ward_by_subject.get(subject_id, pd.DataFrame(columns=["chart_time", "item_name", "value"]))

    admission = float(row["admission_time"])
    orin = float(row["orin_time"])
    orout = float(row["orout_time"])

    lab_frame = resample_long(
        lab,
        context.dictionaries["lab"],
        _time_points(lab, admission, orin, orin),
        time_window=12 * 60,
        direction="left",
        keep_empty_rows=False,
        keep_first=True,
    ).drop(columns="keep")
    lab_raw = lab_frame.copy()
    mask_lab = get_mask(lab_frame, lab_frame.columns[1:])
    lab_frame = _add_dt(fill_frame(lab_frame, context.dictionaries["lab"], context.stats["lab"]))

    ward_frame = resample_long(
        ward,
        context.dictionaries["ward_vital"],
        _time_points(ward, admission, orin, orin),
        time_window=5,
        direction="both",
        keep_empty_rows=False,
        keep_first=False,
    ).drop(columns="keep")
    ward_raw = ward_frame.copy()
    if {"nibp_sbp", "nibp_dbp"}.issubset(ward_frame):
        ward_frame["nibp_mbp"] = (ward_frame["nibp_sbp"] + 2 * ward_frame["nibp_dbp"]) / 3
    mask_ward = get_mask(ward_frame, ward_frame.columns[1:])
    ward_frame = fill_frame(ward_frame, context.dictionaries["ward_vital"], context.stats["ward_vital"])
    ward_frame = fill_last_values(ward_frame, mask_ward, context.dictionaries["ward_vital"])
    ward_frame = one_hot(
        ward_frame,
        context.dictionaries["ward_vital"],
        context.stats["ward_vital"],
        time_column="time",
    )
    ward_frame = _add_dt(ward_frame)

    intraoperative_times = np.arange(orin, orout + 0.001, 5.0)
    if not len(intraoperative_times):
        intraoperative_times = np.asarray([orin])
    vital_frame = resample_long(
        vital,
        context.dictionaries["vital"],
        intraoperative_times,
        time_window=5,
        direction="both",
        keep_empty_rows=True,
        keep_first=False,
    ).drop(columns="keep")
    vital_frame = _derive_blood_pressure(vital_frame)
    vital_raw = vital_frame.copy()
    mask_vital = get_mask(vital_frame, vital_frame.columns[1:])
    observed_vital = vital_frame.copy()
    vital_frame = fill_frame(vital_frame, context.dictionaries["vital"], context.stats["vital"])
    for column in vital_frame.columns.difference(["time"]):
        vital_frame[column] = pd.to_numeric(vital_frame[column], errors="coerce")

    y_mat = pd.DataFrame(
        {
            "mbp": _trajectory_label(observed_vital["nibp_mbp"], ward_frame.get("nibp_mbp", pd.Series(dtype=float))),
            "hr": _trajectory_label(observed_vital.get("hr", pd.Series(np.nan, index=observed_vital.index)), ward_frame.get("hr", pd.Series(dtype=float))),
        }
    )
    y_mask = y_mat.notna().astype(np.int8)
    y_mat = y_mat.fillna(0)

    static_items = [item for item in context.dictionaries["static"]["itemid"].astype(str) if item in row.index]
    static_frame = pd.DataFrame([{item: row[item] for item in static_items}])
    x_static = one_hot(
        static_frame,
        context.dictionaries["static"],
        context.stats["static"],
    )
    y_static = pd.DataFrame([{name: row.get(name, np.nan) for name in STATIC_OUTCOMES}])
    y_mask_static = y_static.notna().astype(np.int8)
    y_static = y_static.fillna(0)

    return {
        "lab.csv": lab_frame,
        "mask_lab.csv": mask_lab,
        "vit.csv": vital_frame,
        "mask_vit.csv": mask_vital,
        "ward_vit.csv": ward_frame,
        "mask_ward_vit.csv": mask_ward,
        "t_list.csv": pd.DataFrame({"time": intraoperative_times}),
        "x_s.csv": x_static,
        "y_mat.csv": y_mat,
        "y_mask.csv": y_mask,
        "y_static.csv": y_static,
        "y_mask1.csv": y_mask_static,
        "lab_raw.csv": lab_raw,
        "vit_raw.csv": vital_raw,
        "ward_vit_raw.csv": ward_raw,
    }


def _write_operation(row: pd.Series, context: GenerationContext, root: Path) -> None:
    folder = root / str(row["op_id"])
    folder.mkdir(parents=True, exist_ok=False)
    for filename, frame in generate_operation_data(row, context).items():
        frame.to_csv(folder / filename, index=False)


def generate_dataset(
    context: GenerationContext,
    output_root: Path,
    *,
    workers: int = 1,
    overwrite: bool = False,
) -> None:
    """Write one folder per operation and normalization parameter files."""

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    rows = [row for _, row in context.operations.iterrows()]
    if workers <= 1:
        for row in rows:
            _write_operation(row, context, output_root)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda row: _write_operation(row, context, output_root), rows))

    parameter_dir = output_root.parent / "param_folder"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    for name, include_dt in (("vital", False), ("static", False), ("lab", True), ("ward_vital", True)):
        z_parameters(
            context.dictionaries[name],
            context.stats[name],
            include_dt=include_dt,
        ).to_csv(parameter_dir / f"z_param_{name}.csv", index=False)
