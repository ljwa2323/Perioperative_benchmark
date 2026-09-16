import pandas as pd

from data_preprocessing.icd import ICD_RULES, category_matches, normalize_icd10, series_matches

EXPECTED_RULES = {
    "viral_hepatitis": (("B15-B19",), ("B16",)),
    "tuberculosis": (("A15-A19",), ()),
    "hiv": (("B20",), ("B20",)),
    "malignant_neoplasm": (("C00-C96",), ()),
    "other_neoplasm": (("D00-D49",), ()),
    "anemia": (("D50-D64",), ()),
    "coagulation_disorder": (("D65-D69",), ()),
    "hyperthyroidism": (("E05",), ()),
    "hypothyroidism": (("E00", "E02-E03"), ()),
    "diabetes_mellitus": (("E08-E13",), ("E08-E13",)),
    "mental_behavioral_neurodevelopmental_disorder": (("F01-F99",), ()),
    "nervous_system_disease": (("G00-G99",), ()),
    "transient_ischemic_attack": (("G45",), ()),
    "hypertensive_disease": (("I10-I1A",), ()),
    "ischemic_heart_disease": (("I20-I25",), ()),
    "pulmonary_heart_and_pulmonary_circulation_disease": (("I26-I28",), ()),
    "pericardial_disease": (("I30-I32",), ()),
    "valvular_heart_disease": (("I05-I08", "I34-I37"), ()),
    "valvular_endocardial_disease": (("I05-I09", "I33-I39"), ()),
    "chronic_rheumatic_heart_disease": (("I05-I09",), ()),
    "cardiomyopathy": (("I40-I43", "I5A"), ()),
    "arrhythmia_and_conduction_disorder": (("I44-I49",), ()),
    "heart_failure": (("I50",), ()),
    "hemorrhagic_cerebrovascular_disease": (("I60-I62",), ()),
    "ischemic_stroke": (("I63",), ()),
    "cerebrovascular_stenosis_or_occlusion_without_infarction": (("I65-I66",), ()),
    "cerebrovascular_disease": (("I60-I69",), ()),
    "arterial_arteriolar_and_capillary_disease": (("I70-I79",), ()),
    "venous_disease": (("I80-I87",), ()),
    "respiratory_infection": (("J00-J22",), ()),
    "copd": (("J40-J44",), ("J40-J44",)),
    "asthma": (("J45",), ()),
    "bronchiectasis": (("J47",), ()),
    "acute_respiratory_distress_syndrome": (("J80",), ()),
    "pulmonary_edema": (("J81",), ()),
    "pleural_effusion": (("J90-J91",), ()),
    "pneumothorax": (("J93",), ()),
    "gastroesophageal_reflux_disease": (("K21",), ()),
    "peptic_ulcer": (("K25-K28",), ()),
    "liver_disease": (("K70-K77",), ()),
    "kidney_disease": (("N00-N19", "N23", "N25-N27"), ()),
    "kidney_and_ureter_disease": (("N00-N20", "N23", "N25-N29"), ()),
    "congenital_heart_malformation": (("Q20-Q24",), ("Q20-Q24",)),
    "bullous_disease": (("L10-L14",), ()),
}


def test_normalize_and_range_boundaries():
    assert normalize_icd10(" i21.01 ") == "I21"
    assert normalize_icd10("I1A.0") == "I1A"
    assert normalize_icd10(None) is None
    assert category_matches("I10", ("I10-I1A",))
    assert category_matches("I1A", ("I10-I1A",))
    assert not category_matches("I20", ("I10-I1A",))
    assert category_matches("K28", ("K25-K28",))
    assert category_matches("E03", ("E02-03",))


def test_all_rules_are_unique_and_vectorized():
    assert len(ICD_RULES) == 44
    assert len({rule.name for rule in ICD_RULES}) == 44
    assert {rule.name: (rule.ranges, rule.timeless) for rule in ICD_RULES} == EXPECTED_RULES
    categories = pd.Series(["C00", "C96", "C97", None], dtype="string")
    assert series_matches(categories, ("C00-C96",)).tolist() == [True, True, False, False]
