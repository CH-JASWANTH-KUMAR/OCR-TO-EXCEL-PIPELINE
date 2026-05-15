"""Evaluation utilities for OCR extraction quality."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List


CRITICAL_FIELD_GROUPS = {
    "CUSTOMER_NAME_2": ["CUSTOMER_NAME_2"],
    "PHONE": ["PH_NO1_8"],
    "DOB": ["D_BIRTH_11", "DOB_33"],
    "EMAIL": ["EMAIL_ADDRESS_3"],
    "ADDRESS": ["RES_ADDRESS_4"],
    "MEDICINE": ["MEDICINE_36"],
    "COST": ["COST_40"],
}

EMAIL_FIELD = "EMAIL_ADDRESS_3"
PHONE_FIELD = "PH_NO1_8"


@dataclass
class EvaluationResult:
    total_records: int
    valid_emails: int
    valid_phones: int
    missing_critical_fields: int
    needs_review: int
    average_extraction_score: int
    missing_fields_counter: Counter
    invalid_email_count: int
    invalid_phone_count: int


def _is_valid_email(value: str) -> bool:
    if not value:
        return False
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(pattern, value.strip()))


def _is_valid_phone(value: str) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def evaluate_payload(payload: Dict[str, Any]) -> EvaluationResult:
    records = payload.get("records", [])
    total_records = len(records) if isinstance(records, list) else 0

    missing_fields_counter: Counter = Counter()
    valid_emails = 0
    valid_phones = 0
    missing_critical_fields = 0
    needs_review = 0
    invalid_email_count = 0
    invalid_phone_count = 0
    record_scores: List[int] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        record_missing = []
        for group_name, fields in CRITICAL_FIELD_GROUPS.items():
            if all(not record.get(field, "").strip() for field in fields):
                record_missing.append(group_name)

        if record_missing:
            missing_critical_fields += len(record_missing)
            missing_fields_counter.update(record_missing)

        email_value = record.get(EMAIL_FIELD, "")
        phone_value = record.get(PHONE_FIELD, "")

        email_valid = _is_valid_email(email_value)
        phone_valid = _is_valid_phone(phone_value)
        email_problem = bool(email_value) and not email_valid
        phone_problem = bool(phone_value) and not phone_valid

        valid_emails += 1 if email_valid else 0
        valid_phones += 1 if phone_valid else 0

        if email_problem:
            invalid_email_count += 1
        if phone_problem:
            invalid_phone_count += 1

        record_score = _score_record(len(record_missing), email_problem, phone_problem)
        record_scores.append(record_score)

        if record_missing or email_problem or phone_problem or record_score < 70:
            needs_review += 1

    average_extraction_score = _score_extraction(record_scores)

    return EvaluationResult(
        total_records=total_records,
        valid_emails=valid_emails,
        valid_phones=valid_phones,
        missing_critical_fields=missing_critical_fields,
        needs_review=needs_review,
        average_extraction_score=average_extraction_score,
        missing_fields_counter=missing_fields_counter,
        invalid_email_count=invalid_email_count,
        invalid_phone_count=invalid_phone_count,
    )


def _score_extraction(record_scores: List[int]) -> int:
    if not record_scores:
        return 0
    return int(round(sum(record_scores) / len(record_scores)))


def _score_record(
    missing_group_count: int,
    email_problem: bool,
    phone_problem: bool,
) -> int:
    score = 100
    score -= missing_group_count * 12
    score -= 10 if email_problem else 0
    score -= 10 if phone_problem else 0

    if score < 0:
        score = 0
    if score > 100:
        score = 100

    return score
