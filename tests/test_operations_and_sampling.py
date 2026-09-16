import pandas as pd

from data_preprocessing.operations import prepare_operations
from data_preprocessing.sampling import select_and_split_operations


def _raw_operations():
    return pd.DataFrame(
        {
            "op_id": [1, 2, 3, 4, 5, 6],
            "subject_id": [1, 1, 2, 3, 4, 5],
            "weight": [60] * 6,
            "height": [160] * 6,
            "admission_time": [0] * 6,
            "discharge_time": [1000] * 6,
            "opstart_time": [110] * 6,
            "opend_time": [170] * 6,
            "orin_time": [100] * 6,
            "orout_time": [180] * 6,
            "anstart_time": [105] * 6,
            "anend_time": [175] * 6,
            "cpbon_time": [pd.NA] * 6,
            "cpboff_time": [pd.NA] * 6,
            "icuin_time": [200] * 6,
            "icuout_time": [260] * 6,
            "sex": ["F", "M", "F", "M", "F", "M"],
            "inhosp_death_time": [pd.NA, 500, pd.NA, pd.NA, pd.NA, pd.NA],
            "icd10_pcs": ["02RF0JZ", "0DTJ0ZZ", "0B110F4", "00B00ZZ", "0HBT0ZZ", "0T9000Z"],
            "antype": ["General"] * 6,
            "age": [50] * 6,
        }
    )


def test_operation_features_and_subject_split():
    operations = prepare_operations(_raw_operations())
    assert operations.loc[0, "sex"] == 0
    assert operations.loc[1, "sex"] == 1
    assert operations.loc[0, "op_duration"] == 60
    assert operations.loc[0, "have_icu"] == 1
    assert operations.loc[1, "death_30d"] == 1
    assert operations.loc[0, "surgery_site"] == "Heart_and_Great_Vessels"

    selected = select_and_split_operations(operations)
    split_counts = selected.groupby("subject_id")["dataset"].nunique()
    assert split_counts.max() == 1
    assert set(selected["dataset"].dropna().astype(int)).issubset({1, 2, 3})
