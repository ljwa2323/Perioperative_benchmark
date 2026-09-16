"""Compact pandas implementation of the EMR-LIP preprocessing primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

Stats = dict[str, dict[str, Any]]


def _mode(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return np.nan
    counts = values.astype("string").value_counts()
    candidates = set(counts[counts.eq(counts.max())].index)
    for value in reversed(values.astype("string").tolist()):
        if value in candidates:
            return value
    return np.nan


def _valid_values(row: pd.Series, observed: pd.Series) -> list[str]:
    configured = row.get("valid_value")
    if pd.notna(configured):
        return sorted(value.strip() for value in str(configured).split("|") if value.strip())
    return sorted(observed.dropna().astype("string").unique().tolist())


def _one_stat(values: pd.Series, row: pd.Series) -> dict[str, Any]:
    value_type = row["value_type"]
    configured_cont = row.get("cont")
    if value_type == "num":
        numeric = pd.to_numeric(values, errors="coerce")
        mean = numeric.mean()
        median = numeric.median()
        standard_deviation = numeric.std()
        mean = 0.0 if pd.isna(mean) else float(mean)
        median = 0.0 if pd.isna(median) else float(median)
        standard_deviation = 1.0 if pd.isna(standard_deviation) else float(standard_deviation)
        return {
            "type": value_type,
            "mean": mean,
            "median": median,
            "sd": standard_deviation,
            "cont": float(configured_cont) if pd.notna(configured_cont) else mean,
        }
    if value_type in {"cat", "ord"}:
        mode = _mode(values)
        return {
            "type": value_type,
            "mode": mode,
            "unique_values": _valid_values(row, values),
            "cont": configured_cont if pd.notna(configured_cont) else mode,
        }
    return {
        "type": value_type,
        "cont": float(configured_cont) if pd.notna(configured_cont) else 0.0,
    }


def stats_long(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    *,
    item_column: str = "item_name",
    value_column: str = "value",
) -> Stats:
    """Calculate training statistics for long-form events."""

    output: Stats = {}
    for _, row in dictionary.iterrows():
        item = str(row["itemid"])
        values = frame.loc[frame[item_column].eq(item), value_column]
        output[item] = _one_stat(values, row)
    return output


def stats_wide(frame: pd.DataFrame, dictionary: pd.DataFrame) -> Stats:
    """Calculate training statistics for a wide feature table."""

    output: Stats = {}
    for _, row in dictionary.iterrows():
        item = str(row["itemid"])
        values = frame[item] if item in frame else pd.Series(dtype=float)
        output[item] = _one_stat(values, row)
    return output


def _aggregate(values: pd.Series, method: str) -> Any:
    non_missing = values.dropna()
    if non_missing.empty:
        return np.nan
    if method == "first":
        return non_missing.iloc[0]
    if method == "last":
        return non_missing.iloc[-1]
    if method in {"mode", "mode_w"}:
        return _mode(non_missing)
    numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    if method in {"mean", "mean_w"}:
        return numeric.mean()
    if method in {"median", "median_w"}:
        return numeric.median()
    if method == "min":
        return numeric.min()
    if method == "max":
        return numeric.max()
    if method in {"sum", "sum_w"}:
        return numeric.sum()
    if method == "any":
        return int(numeric.eq(1).any())
    if method == "all":
        return int(numeric.eq(1).all())
    raise ValueError(f"Unsupported aggregation method: {method}")


def resample_long(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    time_points: Sequence[float],
    *,
    time_window: float,
    direction: str,
    keep_empty_rows: bool,
    keep_first: bool,
    time_column: str = "chart_time",
    item_column: str = "item_name",
    value_column: str = "value",
) -> pd.DataFrame:
    """Aggregate long-form records into configurable windows."""

    source = frame.copy()
    source[time_column] = pd.to_numeric(source[time_column], errors="coerce")
    item_ids = dictionary["itemid"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for time_point in np.asarray(time_points, dtype=float):
        if direction == "left":
            mask = source[time_column].between(time_point - time_window, time_point, inclusive="both")
        elif direction == "right":
            mask = source[time_column].between(time_point, time_point + time_window, inclusive="both")
        elif direction == "both":
            half_window = time_window / 2.0
            mask = source[time_column].between(time_point - half_window, time_point + half_window, inclusive="both")
        else:
            raise ValueError("direction must be left, right, or both")
        current = source.loc[mask]
        row: dict[str, Any] = {"time": time_point, "keep": int(not current.empty)}
        for _, variable in dictionary.iterrows():
            item = str(variable["itemid"])
            values = current.loc[current[item_column].eq(item), value_column]
            row[item] = _aggregate(values, str(variable["agg_f"])) if not values.empty else np.nan
        rows.append(row)

    result = pd.DataFrame(rows, columns=["time", "keep", *item_ids])
    if not keep_empty_rows:
        result = result.loc[result["keep"].eq(1)].copy()
        if result.empty:
            result = pd.DataFrame([{"time": float(time_points[0]), "keep": 0, **dict.fromkeys(item_ids, np.nan)}])
    if len(result) > 1 and result.iloc[0][item_ids].isna().all():
        if keep_first:
            result.loc[result.index[0], "keep"] = 0
        else:
            result = result.iloc[1:]
    return result.reset_index(drop=True)


def get_mask(frame: pd.DataFrame, columns: Iterable[str], *, time_column: str = "time") -> pd.DataFrame:
    output = pd.DataFrame({time_column: frame[time_column].to_numpy()})
    for column in columns:
        output[column] = frame[column].notna().astype(np.int8)
    return output


def _fill_series(series: pd.Series, method: object, stat: dict[str, Any]) -> pd.Series:
    if pd.isna(method):
        return series
    method = str(method).lower()
    if method == "zero":
        return series.fillna(0)
    if method == "cont":
        return series.fillna(stat.get("cont", 0))
    if method in {"mean", "median", "mode"}:
        return series.fillna(stat.get(method, stat.get("cont", 0)))
    if method == "locf":
        return series.ffill()
    if method == "nocb":
        return series.bfill()
    if method == "lin":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.interpolate(method="linear", limit_direction="both")
    raise ValueError(f"Unsupported fill method: {method}")


def fill_frame(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    stats: Stats,
) -> pd.DataFrame:
    result = frame.copy()
    dictionary_by_item = dictionary.set_index(dictionary["itemid"].astype(str), drop=False)
    for column in [column for column in result.columns if column not in {"time", "keep"}]:
        if column not in dictionary_by_item.index:
            continue
        row = dictionary_by_item.loc[column]
        result[column] = _fill_series(result[column], row.get("fill1"), stats[column])
        result[column] = _fill_series(result[column], row.get("fill2"), stats[column])
    return result


def fill_last_values(
    frame: pd.DataFrame,
    mask: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    dictionary_by_item = dictionary.set_index(dictionary["itemid"].astype(str), drop=False)
    for column in [column for column in result.columns if column not in {"time", "keep"}]:
        if column not in dictionary_by_item.index:
            continue
        configured = dictionary_by_item.loc[column].get("last_value")
        if pd.isna(configured):
            continue
        observed = np.flatnonzero(mask[column].to_numpy() == 1)
        start = observed[-1] + 1 if len(observed) else 0
        if start < len(result):
            result.loc[result.index[start:], column] = configured
    return result


def one_hot(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    stats: Stats,
    *,
    time_column: str | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    if time_column is not None:
        output[time_column] = frame[time_column]
    dictionary_by_item = dictionary.set_index(dictionary["itemid"].astype(str), drop=False)
    for column in [column for column in frame.columns if column not in {time_column, "keep"}]:
        if column not in dictionary_by_item.index:
            continue
        value_type = dictionary_by_item.loc[column, "value_type"]
        if value_type in {"num", "bin"}:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
            continue
        values = stats[column].get("unique_values", [])
        for index, value in enumerate(values, start=1):
            output[f"{column}___{index}"] = (
                frame[column].astype("string").eq(str(value)).fillna(False).astype(np.int8)
            )
    return output.reset_index(drop=True)


def z_parameters(dictionary: pd.DataFrame, stats: Stats, *, include_dt: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in dictionary["itemid"].astype(str):
        stat = stats[item]
        rows.append({"var": item, "mean": stat.get("mean", 0.0), "sd": stat.get("sd", 1.0)})
    if include_dt:
        rows.append({"var": "dt", "mean": 0.0, "sd": 1.0})
    return pd.DataFrame(rows)
