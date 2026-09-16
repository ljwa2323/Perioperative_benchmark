import pandas as pd

from data_preprocessing.diagnoses import annotate_diagnoses, prepare_diagnoses


def test_preoperative_cutoff_timeless_codes_and_postoperative_window():
    operations = pd.DataFrame(
        {
            "op_id": [1, 2, 3],
            "subject_id": [10, 10, 20],
            "orin_time": [100, 250, 100],
            "discharge_time": [200, 400, 200],
        }
    )
    diagnoses = prepare_diagnoses(
        pd.DataFrame(
            {
                "subject_id": [10, 10, 10, 10, 10, 10, 20, 20],
                "chart_time": [99, 100, 150, 150, 300, 50, 50, 50],
                "icd10_cm": ["I10", "I63.9", "J81", "E11.9", "B16.1", "C96", "C97", "I1A.0"],
            }
        ),
        time_divisor=1,
    )
    result = annotate_diagnoses(operations, diagnoses).set_index("op_id")

    assert result.loc[1, "preop_hypertensive_disease"] == 1
    assert result.loc[1, "preop_ischemic_stroke"] == 0
    assert result.loc[1, "postop_ischemic_stroke"] == 0
    assert result.loc[1, "postop_pulmonary_edema"] == 1
    assert result.loc[2, "preop_pulmonary_edema"] == 1
    assert result.loc[1, "preop_diabetes_mellitus"] == 1
    assert result.loc[1, "postop_diabetes_mellitus"] == 1
    assert result.loc[1, "preop_viral_hepatitis"] == 1
    assert result.loc[1, "postop_viral_hepatitis"] == 0
    assert result.loc[1, "preop_malignant_neoplasm"] == 1
    assert result.loc[3, "preop_malignant_neoplasm"] == 0
    assert result.loc[3, "preop_hypertensive_disease"] == 1


def test_default_diagnosis_time_conversion():
    diagnoses = prepare_diagnoses(
        pd.DataFrame({"subject_id": [1], "chart_time": [6000], "icd10_cm": ["I10"]})
    )
    assert diagnoses.loc[0, "chart_time"] == 100
