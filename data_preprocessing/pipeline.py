"""End-to-end Python preprocessing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .diagnoses import annotate_diagnoses, prepare_diagnoses
from .generation import build_generation_context, generate_dataset, load_variable_dictionaries
from .operations import prepare_operations
from .outcomes import annotate_outcomes
from .sampling import select_and_split_operations
from .scores import add_clinical_scores

LOGGER = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required INSPIRE file not found: {path}")
    try:
        return pd.read_csv(path, engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, low_memory=False)


def run_pipeline(
    inspire_path: Path,
    output_dir: Path,
    *,
    variable_dictionary: Path,
    diagnosis_time_divisor: float = 60.0,
    workers: int = 1,
    generate_sequences: bool = True,
    overwrite_sequences: bool = False,
) -> pd.DataFrame:
    """Run preprocessing and return the selected operation table."""

    inspire_path = inspire_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading operations")
    operations = prepare_operations(_read_csv(inspire_path / "operations.csv"))

    LOGGER.info("Reading and annotating ICD-10 diagnoses")
    diagnoses = prepare_diagnoses(
        _read_csv(inspire_path / "diagnosis.csv"),
        time_divisor=diagnosis_time_divisor,
    )
    operations = annotate_diagnoses(operations, diagnoses)

    LOGGER.info("Reading laboratory and ward data")
    labs = _read_csv(inspire_path / "labs.csv")
    ward_vitals = _read_csv(inspire_path / "ward_vitals.csv")
    operations = annotate_outcomes(operations, labs, ward_vitals)

    LOGGER.info("Calculating clinical scores")
    operations = add_clinical_scores(operations, diagnoses, labs)
    operations.to_csv(output_dir / "operation_derived1.csv", index=False)

    LOGGER.info("Selecting cohort and splitting by subject")
    selected = select_and_split_operations(operations)
    selected.to_csv(output_dir / "operation_.csv", index=False)

    if generate_sequences:
        LOGGER.info("Generating per-operation model inputs")
        vitals = _read_csv(inspire_path / "vitals.csv")
        dictionaries = load_variable_dictionaries(variable_dictionary)
        context = build_generation_context(selected, labs, vitals, ward_vitals, dictionaries)
        context.operations.to_csv(output_dir / "operation_imputed.csv", index=False)
        generate_dataset(
            context,
            output_dir / "all_op_id",
            workers=workers,
            overwrite=overwrite_sequences,
        )

    return selected
