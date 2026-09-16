"""Central ICD-10 rules used by preoperative and postoperative annotation.

ICD-10-CM subcodes are matched by their three-character category. A range such
as ``I20-I25`` therefore includes ``I21.01``. The ``timeless`` field is used
only when deciding whether a diagnosis is a preoperative comorbidity.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ICDRule:
    """One disease label and its ICD-10 category ranges."""

    name: str
    label_zh: str
    ranges: tuple[str, ...]
    timeless: tuple[str, ...] = ()


ICD_RULES: tuple[ICDRule, ...] = (
    ICDRule("viral_hepatitis", "病毒性肝炎", ("B15-B19",), ("B16",)),
    ICDRule("tuberculosis", "结核病", ("A15-A19",)),
    ICDRule("hiv", "HIV", ("B20",), ("B20",)),
    ICDRule("malignant_neoplasm", "恶性肿瘤", ("C00-C96",)),
    ICDRule("other_neoplasm", "其他肿瘤", ("D00-D49",)),
    ICDRule("anemia", "贫血", ("D50-D64",)),
    ICDRule("coagulation_disorder", "凝血功能障碍", ("D65-D69",)),
    ICDRule("hyperthyroidism", "甲亢", ("E05",)),
    ICDRule("hypothyroidism", "甲减", ("E00", "E02-E03")),
    ICDRule("diabetes_mellitus", "糖尿病", ("E08-E13",), ("E08-E13",)),
    ICDRule(
        "mental_behavioral_neurodevelopmental_disorder",
        "精神行为和神经发育障碍",
        ("F01-F99",),
    ),
    ICDRule("nervous_system_disease", "神经系统疾病", ("G00-G99",)),
    ICDRule("transient_ischemic_attack", "短暂性脑缺血发作", ("G45",)),
    ICDRule("hypertensive_disease", "高血压疾病", ("I10-I1A",)),
    ICDRule("ischemic_heart_disease", "缺血性心脏病", ("I20-I25",)),
    ICDRule(
        "pulmonary_heart_and_pulmonary_circulation_disease",
        "肺源性心脏病和肺循环疾病",
        ("I26-I28",),
    ),
    ICDRule("pericardial_disease", "心包疾病", ("I30-I32",)),
    ICDRule("valvular_heart_disease", "瓣膜疾病", ("I05-I08", "I34-I37")),
    ICDRule(
        "valvular_endocardial_disease",
        "瓣膜（心内膜）疾病",
        ("I05-I09", "I33-I39"),
    ),
    ICDRule("chronic_rheumatic_heart_disease", "慢性风湿性心脏病", ("I05-I09",)),
    ICDRule("cardiomyopathy", "心肌疾病", ("I40-I43", "I5A")),
    ICDRule(
        "arrhythmia_and_conduction_disorder",
        "心律失常及传导障碍",
        ("I44-I49",),
    ),
    ICDRule("heart_failure", "心力衰竭", ("I50",)),
    ICDRule(
        "hemorrhagic_cerebrovascular_disease",
        "出血性脑血管病",
        ("I60-I62",),
    ),
    ICDRule("ischemic_stroke", "缺血性卒中", ("I63",)),
    ICDRule(
        "cerebrovascular_stenosis_or_occlusion_without_infarction",
        "脑血管狭窄或闭塞未致梗死",
        ("I65-I66",),
    ),
    ICDRule("cerebrovascular_disease", "脑血管疾病", ("I60-I69",)),
    ICDRule(
        "arterial_arteriolar_and_capillary_disease",
        "动脉、小动脉和毛细血管疾病",
        ("I70-I79",),
    ),
    ICDRule("venous_disease", "静脉疾病", ("I80-I87",)),
    ICDRule("respiratory_infection", "呼吸道感染", ("J00-J22",)),
    ICDRule("copd", "COPD", ("J40-J44",), ("J40-J44",)),
    ICDRule("asthma", "哮喘", ("J45",)),
    ICDRule("bronchiectasis", "支气管扩张症", ("J47",)),
    ICDRule("acute_respiratory_distress_syndrome", "急性呼吸窘迫综合征", ("J80",)),
    ICDRule("pulmonary_edema", "肺水肿", ("J81",)),
    ICDRule("pleural_effusion", "胸腔积液", ("J90-J91",)),
    ICDRule("pneumothorax", "气胸", ("J93",)),
    ICDRule("gastroesophageal_reflux_disease", "胃食管反流病", ("K21",)),
    ICDRule("peptic_ulcer", "消化道溃疡", ("K25-K28",)),
    ICDRule("liver_disease", "肝脏疾病", ("K70-K77",)),
    ICDRule("kidney_disease", "肾脏疾病", ("N00-N19", "N23", "N25-N27")),
    ICDRule(
        "kidney_and_ureter_disease",
        "肾脏和输尿管疾病",
        ("N00-N20", "N23", "N25-N29"),
    ),
    ICDRule(
        "congenital_heart_malformation",
        "先天性心脏畸形",
        ("Q20-Q24",),
        ("Q20-Q24",),
    ),
    ICDRule("bullous_disease", "大疱性疾病", ("L10-L14",)),
)


_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_CATEGORY = re.compile(r"^([A-Z][0-9A-Z]{2})")


def normalize_icd10(code: object) -> str | None:
    """Normalize an ICD-10-CM value and return its three-character category."""

    if code is None or pd.isna(code):
        return None
    cleaned = _NON_ALNUM.sub("", str(code).strip().upper())
    match = _CATEGORY.match(cleaned)
    return match.group(1) if match else None


def _normalize_spec(spec: str) -> tuple[str, str]:
    normalized = spec.upper().replace("–", "-").replace("—", "-").replace(" ", "")
    if "-" not in normalized:
        category = normalize_icd10(normalized)
        if category is None:
            raise ValueError(f"Invalid ICD-10 category: {spec}")
        return category, category

    start_text, end_text = normalized.split("-", 1)
    start = normalize_icd10(start_text)
    if len(end_text) < 3:
        end_text = start_text[0] + end_text
    end = normalize_icd10(end_text)
    if start is None or end is None or start[0] != end[0] or start > end:
        raise ValueError(f"Invalid ICD-10 range: {spec}")
    return start, end


def category_matches(category: str | None, specs: Iterable[str]) -> bool:
    """Return whether a normalized three-character category matches any spec."""

    if category is None:
        return False
    return any(start <= category <= end for start, end in map(_normalize_spec, specs))


def series_matches(categories: pd.Series, specs: Iterable[str]) -> pd.Series:
    """Vectorized ICD-10 category matching."""

    clean = categories.astype("string")
    result = pd.Series(False, index=categories.index)
    for spec in specs:
        start, end = _normalize_spec(spec)
        result |= clean.between(start, end, inclusive="both").fillna(False)
    return result


def validate_rules(rules: Iterable[ICDRule] = ICD_RULES) -> None:
    """Fail early if a configured rule or timeless subset is malformed."""

    seen: set[str] = set()
    for rule in rules:
        if rule.name in seen:
            raise ValueError(f"Duplicate ICD rule name: {rule.name}")
        seen.add(rule.name)
        for spec in (*rule.ranges, *rule.timeless):
            _normalize_spec(spec)
        for timeless in rule.timeless:
            start, end = _normalize_spec(timeless)
            if not any(
                rule_start <= start and end <= rule_end
                for rule_start, rule_end in map(_normalize_spec, rule.ranges)
            ):
                raise ValueError(f"Timeless range {timeless} is outside {rule.name}")


validate_rules()
