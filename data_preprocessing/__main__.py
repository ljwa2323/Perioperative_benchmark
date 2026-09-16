"""Command-line entry point for INSPIRE preprocessing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the INSPIRE perioperative benchmark dataset with Python.",
    )
    parser.add_argument("--inspire-path", type=Path, required=True, help="Directory containing INSPIRE CSV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for derived benchmark data")
    parser.add_argument(
        "--variable-dictionary",
        type=Path,
        default=Path(__file__).with_name("var_dict.xlsx"),
        help="Path to var_dict.xlsx",
    )
    parser.add_argument(
        "--diagnosis-time-divisor",
        type=float,
        default=60.0,
        help="Divide diagnosis chart_time by this value before operation-time comparisons",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent writers for per-operation files")
    parser.add_argument("--skip-sequences", action="store_true", help="Only produce operation-level CSV files")
    parser.add_argument(
        "--overwrite-sequences",
        action="store_true",
        help="Replace an existing all_op_id directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(
        args.inspire_path,
        args.output_dir,
        variable_dictionary=args.variable_dictionary,
        diagnosis_time_divisor=args.diagnosis_time_divisor,
        workers=max(1, args.workers),
        generate_sequences=not args.skip_sequences,
        overwrite_sequences=args.overwrite_sequences,
    )


if __name__ == "__main__":
    main()
