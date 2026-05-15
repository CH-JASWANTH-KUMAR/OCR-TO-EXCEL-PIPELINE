"""Export structured JSON records to a client-ready Excel file."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd


def _normalize_records(records: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for record in records:
        normalized.append({col: record.get(col, "") for col in columns})
    return normalized


def export_structured_json_to_excel(
    json_path: str,
    excel_path: str,
) -> None:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    columns = payload.get("columns", [])
    records = payload.get("records", [])

    if not isinstance(columns, list) or not columns:
        raise ValueError("Structured JSON must include a non-empty 'columns' list")

    if not isinstance(records, list):
        raise ValueError("Structured JSON must include a 'records' list")

    normalized_records = _normalize_records(records, columns)
    df = pd.DataFrame(normalized_records, columns=columns)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
