"""Python replacement for the former modeling/helper.R script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _normalized_inverse_prevalence(counts: np.ndarray) -> np.ndarray:
    prevalence = counts / counts.sum()
    prevalence = np.clip(prevalence, 0.001, 0.999)
    weights = 1.0 / prevalence
    return weights / weights.sum()


def calculate_dynamic_class_weights(root_path: str | Path) -> dict[str, np.ndarray]:
    """Calculate inverse-prevalence weights from per-operation labels and masks."""

    root = Path(root_path)
    label_frames: list[pd.DataFrame] = []
    mask_frames: list[pd.DataFrame] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        label_frames.append(pd.read_csv(folder / "y_mat.csv"))
        mask_frames.append(pd.read_csv(folder / "y_mask.csv"))
    if not label_frames:
        raise ValueError(f"No operation folders found in {root}")

    labels = pd.concat(label_frames, ignore_index=True)
    masks = pd.concat(mask_frames, ignore_index=True)
    labels = labels.mask(masks.eq(0))
    weights: dict[str, np.ndarray] = {}
    for column in labels:
        values = pd.to_numeric(labels[column], errors="coerce").dropna().astype(int)
        if values.empty:
            weights[column] = np.ones(1, dtype=float)
            continue
        class_count = max(2, int(values.max()) + 1)
        counts = np.bincount(values, minlength=class_count).astype(float)
        counts[counts == 0] = 0.001
        weights[column] = _normalized_inverse_prevalence(counts)
    return weights
