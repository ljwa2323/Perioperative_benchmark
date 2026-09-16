from pathlib import Path

import pandas as pd

from data_preprocessing.pipeline import run_pipeline


def test_operation_level_pipeline_smoke(tmp_path: Path):
    inspire = tmp_path / "inspire"
    output = tmp_path / "output"
    inspire.mkdir()
    pd.DataFrame(
        {
            "op_id": [1, 2],
            "subject_id": [10, 20],
            "hadm_id": [100, 200],
            "weight": [60, 70],
            "height": [160, 170],
            "admission_time": [0, 0],
            "discharge_time": [500, 500],
            "opstart_time": [110, 110],
            "opend_time": [170, 170],
            "orin_time": [100, 100],
            "orout_time": [180, 180],
            "anstart_time": [105, 105],
            "anend_time": [175, 175],
            "cpbon_time": [pd.NA, pd.NA],
            "cpboff_time": [pd.NA, pd.NA],
            "icuin_time": [200, pd.NA],
            "icuout_time": [260, pd.NA],
            "sex": ["F", "M"],
            "inhosp_death_time": [pd.NA, pd.NA],
            "icd10_pcs": ["02RF0JZ", "0DTJ0ZZ"],
            "antype": ["General", "General"],
            "age": [50, 60],
            "asa": [2, 3],
            "emop": [0, 1],
        }
    ).to_csv(inspire / "operations.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [10, 10, 20],
            "chart_time": [3000, 9000, 12000],
            "icd10_cm": ["I10", "J81", "E11.9"],
        }
    ).to_csv(inspire / "diagnosis.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [10, 10, 10, 10, 20, 20],
            "chart_time": [50, 200, 50, 200, 50, 200],
            "item_name": ["creatinine", "creatinine", "alt", "alt", "creatinine", "creatinine"],
            "value": [1.0, 1.1, 20, 30, 1.0, 2.1],
        }
    ).to_csv(inspire / "labs.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [10, 10, 20],
            "chart_time": [50, 200, 50],
            "item_name": ["vent", "vent", "crrt"],
            "value": [0, 1, 0],
        }
    ).to_csv(inspire / "ward_vitals.csv", index=False)
    pd.DataFrame(
        {
            "op_id": [1, 1, 1, 2, 2, 2],
            "chart_time": [110, 110, 110, 110, 110, 110],
            "item_name": ["hr", "nibp_sbp", "nibp_dbp"] * 2,
            "value": [70, 120, 70, 80, 130, 75],
        }
    ).to_csv(inspire / "vitals.csv", index=False)

    selected = run_pipeline(
        inspire,
        output,
        variable_dictionary=Path("data_preprocessing/var_dict.xlsx"),
        generate_sequences=True,
    )

    assert (output / "operation_derived1.csv").exists()
    assert (output / "operation_.csv").exists()
    assert (output / "operation_imputed.csv").exists()
    assert (output / "all_op_id" / "1" / "x_s.csv").exists()
    assert (output / "all_op_id" / "2" / "vit.csv").exists()
    assert (output / "param_folder" / "z_param_static.csv").exists()
    assert selected.loc[selected["op_id"].eq(1), "preop_hypertensive_disease"].item() == 1
    assert selected.loc[selected["op_id"].eq(1), "postop_pulmonary_edema"].item() == 1
    assert selected.loc[selected["op_id"].eq(2), "preop_diabetes_mellitus"].item() == 1
