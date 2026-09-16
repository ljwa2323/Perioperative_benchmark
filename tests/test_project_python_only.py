import json
from pathlib import Path

import pandas as pd

from data_preprocessing.icd import ICD_RULES

ROOT = Path(__file__).resolve().parents[1]


def test_static_dictionary_contains_current_icd_features():
    static = pd.read_excel(ROOT / "data_preprocessing" / "var_dict.xlsx", sheet_name="static")
    fields = set(static["itemid"].dropna().astype(str))
    expected = {f"preop_{rule.name}" for rule in ICD_RULES}
    assert expected.issubset(fields)
    assert "preop_essential_hypertension" not in fields


def test_repository_has_no_r_code_and_notebooks_use_python():
    assert not list(ROOT.rglob("*.R"))
    for notebook_path in ROOT.glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text())
        assert notebook.get("metadata", {}).get("language_info", {}).get("name") == "python"
