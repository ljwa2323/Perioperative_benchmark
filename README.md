# Perioperative Benchmark

Python preprocessing and modeling utilities for perioperative research with the
INSPIRE dataset.

## What changed

The preprocessing pipeline is now implemented in Python. It replaces the former
R scripts, removes hard-coded local paths, centralizes ICD-10 rules, and avoids
per-operation diagnosis scans.

Diagnosis annotation uses 44 disease categories defined in
`data_preprocessing/icd.py`.

Preoperative comorbidities follow these rules:

* A diagnosis normally requires `chart_time < orin_time`.
* B16, B20, E08-E13, J40-J44, and Q20-Q24 are treated as time-independent
  preoperative diagnoses.
* Postoperative diagnosis flags require
  `orin_time < chart_time < discharge_time`.
* ICD-10-CM subcodes are matched by their three-character category. For example,
  I21.01 matches I20-I25.

The diagnosis timestamp conversion remains configurable. INSPIRE diagnosis time
is divided by 60 by default, matching the unit conversion in the original code.

## Installation

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
```

## Data preprocessing

Run the complete pipeline:

```bash
python -m data_preprocessing \
  --inspire-path /path/to/inspire-1.2 \
  --output-dir /path/to/inspire_benchmark_data \
  --workers 8
```

Generate only operation-level files while checking the ICD and outcome logic:

```bash
python -m data_preprocessing \
  --inspire-path /path/to/inspire-1.2 \
  --output-dir /path/to/inspire_benchmark_data \
  --skip-sequences
```

The pipeline writes:

* `operation_derived1.csv`: all annotated operations
* `operation_.csv`: selected cohort with subject-level train, test, and validation split
* `operation_imputed.csv`: static model features after deterministic imputation
* `all_op_id/<op_id>/`: per-operation time-series model inputs
* `param_folder/`: normalization parameters calculated from the training split

Existing `all_op_id` output is protected by default. Use
`--overwrite-sequences` only when replacement is intended.

## Project structure

* `data_preprocessing/icd.py`: canonical ICD-10 disease rules
* `data_preprocessing/diagnoses.py`: preoperative and postoperative annotation
* `data_preprocessing/operations.py`: operation-level derived variables
* `data_preprocessing/outcomes.py`: laboratory and postoperative outcomes
* `data_preprocessing/scores.py`: SORT, Charlson, and RCRI scores
* `data_preprocessing/emr.py`: time-series resampling, filling, masks, and encoding
* `data_preprocessing/generation.py`: per-operation model input generation
* `data_preprocessing/pipeline.py`: end-to-end orchestration
* `modeling/`: PyTorch models, loading, plotting, and class-weight utilities
* `tests/`: synthetic tests for ICD boundaries and time-window behavior

All notebooks use a Python kernel.

## Validation

```bash
pytest
```

The test suite checks ICD range boundaries, normalization, strict preoperative
cutoffs, time-independent diagnoses, postoperative windows, operation features,
and subject-level data separation.

## Data

Raw INSPIRE files are not included. Place `operations.csv`, `diagnosis.csv`,
`labs.csv`, `vitals.csv`, and `ward_vitals.csv` in the directory passed through
`--inspire-path`.
