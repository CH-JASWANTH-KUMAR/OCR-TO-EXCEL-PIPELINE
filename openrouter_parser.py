"""OpenRouter-based parser for OCR text to structured JSON."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


SCHEMA_FIELDS: List[str] = [
    "ID",
    "RECORD_NO_1",
    "CUSTOMER_NAME_2",
    "EMAIL_ADDRESS_3",
    "RES_ADDRESS_4",
    "CITY_1_5",
    "STATE_1_6",
    "ZIP_1_7",
    "PH_NO1_8",
    "COUNTRY_1_9",
    "SEX_1_10",
    "D_BIRTH_11",
    "HEIGHT_12",
    "WEIGHT_13",
    "BLOOD_GROUP_14",
    "BILLING_NAME_15",
    "SHIPPER_NAME_16",
    "CITY_2_17",
    "STATE_2_18",
    "ZIP_2_19",
    "COUNTRY_2_20",
    "PH_NO2_21",
    "ALCOHOLIC_22",
    "SMOKER_23",
    "PAST_SURG_24",
    "DIABETIC_25",
    "ALLERGISED_26",
    "POLICY_NO_27",
    "D_B_LIFE_ASSURE_28",
    "P_INST_29",
    "NAME_P_HOLDER_30",
    "STM_NAME_31",
    "STM_CODE_32",
    "DOB_33",
    "SEX_2_34",
    "CARD_NAME_35",
    "MEDICINE_36",
    "DOSAGE_37",
    "TABLETS_38",
    "PILL_RATE_39",
    "COST_40",
    "SHIPPING_COST_41",
    "TOTAL_AMT_42",
    "REMARK_43",
]

OUTPUT_COLUMNS: List[str] = SCHEMA_FIELDS + ["NEEDS_REVIEW"]

CRITICAL_FIELD_GROUPS = {
    "CUSTOMER_NAME_2": ["CUSTOMER_NAME_2"],
    "PHONE": ["PH_NO1_8"],
    "DOB": ["D_BIRTH_11", "DOB_33"],
    "EMAIL": ["EMAIL_ADDRESS_3"],
    "ADDRESS": ["RES_ADDRESS_4"],
    "MEDICINE": ["MEDICINE_36"],
    "COST": ["COST_40"],
}

RETRY_MISSING_THRESHOLD = 3
REVIEW_SCORE_THRESHOLD = 70


logger = logging.getLogger("ocr_pipeline")


def _strip_row_prefix(line: str) -> str:
    match = re.match(r"^\s*\d+\s*:\s*(.*)$", line)
    if match:
        return match.group(1).strip()
    return line.strip()


def _split_customer_blocks(ocr_text: str) -> List[str]:
    lines = [line for line in ocr_text.splitlines() if line.strip()]
    cleaned_lines = [_strip_row_prefix(line) for line in lines]

    blocks: List[List[str]] = []
    current_block: List[str] = []
    record_id_pattern = re.compile(r"^\d{4,}\b")

    for line in cleaned_lines:
        if record_id_pattern.match(line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return ["\n".join(block).strip() for block in blocks if block]


@dataclass
class OpenRouterConfig:
    api_key: str
    model_name: str = "qwen/qwen-2.5-72b-instruct"
    max_retries: int = 3
    retry_backoff_sec: float = 1.5
    timeout_sec: int = 60
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterParserError(RuntimeError):
    pass


def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterParserError("Missing OPENROUTER_API_KEY in environment or .env file")
    return api_key


def _extract_json(text: str) -> str:
    # Try to find the first JSON object or array in the response.
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _coerce_digits(value: str) -> str:
    return value.translate(str.maketrans({"O": "0", "I": "1", "S": "5", "B": "8"}))


def _normalize_phone(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", _coerce_digits(value))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if 10 < len(digits) <= 15:
        return digits
    return ""


def _normalize_zip(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", _coerce_digits(value))
    if len(digits) == 5:
        return digits
    if len(digits) == 9:
        return f"{digits[:5]}-{digits[5:]}"
    return ""


def _normalize_date(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", value)
    if not match:
        return ""
    month, day, year = match.groups()
    month_int = int(month)
    day_int = int(day)
    year_int = int(year) if len(year) == 4 else int(f"20{year}")
    if not (1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100):
        return ""
    return f"{month_int:02d}/{day_int:02d}/{year_int:04d}"


def _normalize_cost(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"\d+(?:\.\d{1,2})?", value.replace(",", ""))
    if not match:
        return ""
    return f"${match.group(0)}"


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)

    normalized["PH_NO1_8"] = _normalize_phone(normalized.get("PH_NO1_8", ""))
    normalized["PH_NO2_21"] = _normalize_phone(normalized.get("PH_NO2_21", ""))
    normalized["ZIP_1_7"] = _normalize_zip(normalized.get("ZIP_1_7", ""))
    normalized["ZIP_2_19"] = _normalize_zip(normalized.get("ZIP_2_19", ""))
    normalized["D_BIRTH_11"] = _normalize_date(normalized.get("D_BIRTH_11", ""))
    normalized["DOB_33"] = _normalize_date(normalized.get("DOB_33", ""))
    normalized["COST_40"] = _normalize_cost(normalized.get("COST_40", ""))

    email_value = normalized.get("EMAIL_ADDRESS_3", "")
    email_value = email_value.replace(" ", "").replace(",", ".")
    email_value = email_value.replace("[at]", "@").replace("(at)", "@")
    email_value = email_value.replace("[dot]", ".").replace("(dot)", ".")
    normalized["EMAIL_ADDRESS_3"] = email_value

    return normalized


def _get_missing_critical_groups(record: Dict[str, Any]) -> List[str]:
    missing = []
    for group_name, fields in CRITICAL_FIELD_GROUPS.items():
        if all(not record.get(field, "").strip() for field in fields):
            missing.append(group_name)
    return missing


def _get_missing_field_names(record: Dict[str, Any]) -> List[str]:
    missing_fields: List[str] = []
    for fields in CRITICAL_FIELD_GROUPS.values():
        if all(not record.get(field, "").strip() for field in fields):
            missing_fields.extend(fields)
    return missing_fields


def _score_record(record: Dict[str, Any]) -> int:
    missing_count = len(_get_missing_critical_groups(record))
    email_value = record.get("EMAIL_ADDRESS_3", "")
    phone_value = record.get("PH_NO1_8", "")

    email_problem = bool(email_value) and not re.match(
        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
        email_value.strip(),
    )
    phone_problem = bool(phone_value) and len(re.sub(r"\D", "", phone_value)) < 10

    score = 100
    score -= missing_count * 12
    score -= 10 if email_problem else 0
    score -= 10 if phone_problem else 0
    return max(0, min(100, score))


def _find_uppercase_name(lines: List[str]) -> str:
    for line in lines:
        if re.match(r"^[A-Z][A-Z\s'-]{4,}$", line.strip()):
            words = [w for w in line.split() if len(w) >= 2]
            if len(words) >= 2:
                return line.strip()
    return ""


def _regex_fallback(block_text: str, record: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(record)
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    if not updated.get("EMAIL_ADDRESS_3", ""):
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", joined)
        if email_match:
            updated["EMAIL_ADDRESS_3"] = email_match.group(0)

    if not updated.get("PH_NO1_8", ""):
        phone_match = re.search(r"(\+?\d[\d\s().-]{8,}\d)", joined)
        if phone_match:
            updated["PH_NO1_8"] = phone_match.group(1)

    if not updated.get("D_BIRTH_11", "") and not updated.get("DOB_33", ""):
        date_match = re.search(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", joined)
        if date_match:
            updated["D_BIRTH_11"] = date_match.group(0)

    if not updated.get("CUSTOMER_NAME_2", ""):
        uppercase_name = _find_uppercase_name(lines)
        if uppercase_name:
            updated["CUSTOMER_NAME_2"] = uppercase_name

    if not updated.get("RES_ADDRESS_4", ""):
        for line in lines:
            if re.search(r"\b\d+\s+\w+\s+(ST|STREET|RD|ROAD|AVE|AVENUE|BLVD|LANE|LN|DR|CT)\b", line, re.I):
                updated["RES_ADDRESS_4"] = line
                break

    if not updated.get("MEDICINE_36", ""):
        for line in lines:
            if re.search(r"\b[A-Z]{3,}\b", line) and "MG" in line.upper():
                updated["MEDICINE_36"] = line.split()[0]
                break

    if not updated.get("COST_40", ""):
        cost_match = re.search(r"\$?\s*\d+(?:\.\d{1,2})?", joined)
        if cost_match:
            updated["COST_40"] = cost_match.group(0)

    return _normalize_record(updated)


def _retry_missing_fields(
    block_text: str,
    record: Dict[str, Any],
    missing_fields: List[str],
    config: OpenRouterConfig,
) -> Dict[str, Any]:
    prompt = f"""
You are refining OCR extraction. Only fill missing fields and return ONLY JSON.

Missing fields: {", ".join(missing_fields)}

Current extracted JSON:
{json.dumps(record, ensure_ascii=True)}

OCR TEXT:
{block_text}
"""
    response_payload = _call_openrouter(prompt, config)
    json_text = _extract_json(response_payload)
    return json.loads(json_text)


def _call_openrouter(prompt: str, config: OpenRouterConfig) -> str:
    request_payload = {
        "model": config.model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter API call happens here.
    response = requests.post(
        config.endpoint,
        headers=headers,
        json=request_payload,
        timeout=config.timeout_sec,
    )

    if response.status_code >= 400:
        raise OpenRouterParserError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    response_payload = response.json()
    choices = response_payload.get("choices", [])
    if not choices:
        raise OpenRouterParserError("OpenRouter response missing choices")

    message = choices[0].get("message", {})
    raw_text = message.get("content", "")
    if not raw_text:
        raise OpenRouterParserError("OpenRouter response missing message content")

    return raw_text


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "columns" in payload and "records" in payload:
        columns = payload.get("columns")
        records = payload.get("records")
        if not isinstance(columns, list) or not isinstance(records, list):
            raise OpenRouterParserError("Structured JSON must include 'columns' list and 'records' list")
        if not columns:
            columns = OUTPUT_COLUMNS
        normalized_records = []
        for record in records:
            if not isinstance(record, dict):
                raise OpenRouterParserError("Each record must be a JSON object")
            normalized_records.append({col: record.get(col, "") for col in columns})
        return {"columns": columns, "records": normalized_records}

    if isinstance(payload, dict):
        normalized_record = {field: payload.get(field, "") for field in OUTPUT_COLUMNS}
        return {"columns": OUTPUT_COLUMNS, "records": [normalized_record]}

    raise OpenRouterParserError("OpenRouter response must be a JSON object")


def _build_prompt(ocr_text: str) -> str:
    return f"""
You are an OCR medical data extraction engine.

Your task:
Extract structured data from OCR text and return ONLY valid JSON.

IMPORTANT RULES:

* Return ONLY JSON
* No markdown
* No explanations
* No comments
* Do NOT invent values
* If value is missing or uncertain, use empty string ""
* Preserve empty fields instead of guessing
* Preserve exact field names
* Each OCR block represents one customer/patient record
* CUSTOMER_NAME_2 must come from the nearest uppercase full name found in the OCR text

Required JSON schema:

{{
"ID": "",
"RECORD_NO_1": "",
"CUSTOMER_NAME_2": "",
"EMAIL_ADDRESS_3": "",
"RES_ADDRESS_4": "",
"CITY_1_5": "",
"STATE_1_6": "",
"ZIP_1_7": "",
"PH_NO1_8": "",
"COUNTRY_1_9": "",
"SEX_1_10": "",
"D_BIRTH_11": "",
"HEIGHT_12": "",
"WEIGHT_13": "",
"BLOOD_GROUP_14": "",
"BILLING_NAME_15": "",
"SHIPPER_NAME_16": "",
"CITY_2_17": "",
"STATE_2_18": "",
"ZIP_2_19": "",
"COUNTRY_2_20": "",
"PH_NO2_21": "",
"ALCOHOLIC_22": "",
"SMOKER_23": "",
"PAST_SURG_24": "",
"DIABETIC_25": "",
"ALLERGISED_26": "",
"POLICY_NO_27": "",
"D_B_LIFE_ASSURE_28": "",
"P_INST_29": "",
"NAME_P_HOLDER_30": "",
"STM_NAME_31": "",
"STM_CODE_32": "",
"DOB_33": "",
"SEX_2_34": "",
"CARD_NAME_35": "",
"MEDICINE_36": "",
"DOSAGE_37": "",
"TABLETS_38": "",
"PILL_RATE_39": "",
"COST_40": "",
"SHIPPING_COST_41": "",
"TOTAL_AMT_42": "",
"REMARK_43": ""
}}

OCR TEXT:
{ocr_text}
"""


def parse_ocr_to_structured_json(
    ocr_text: str,
    config: Optional[OpenRouterConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = OpenRouterConfig(api_key=_load_api_key())

    blocks = _split_customer_blocks(ocr_text)
    if not blocks:
        blocks = [ocr_text]

    logger.info("Detected %s customer block(s)", len(blocks))

    all_records: List[Dict[str, Any]] = []

    for block_index, block_text in enumerate(blocks, start=1):
        last_error: Optional[Exception] = None
        logger.info("Processing customer block %s", block_index)

        for attempt in range(1, config.max_retries + 1):
            try:
                logger.info("OpenRouter request attempt %s/%s", attempt, config.max_retries)
                prompt = _build_prompt(block_text)
                raw_text = _call_openrouter(prompt, config)

                json_text = _extract_json(raw_text)
                payload = json.loads(json_text)
                normalized = _normalize_payload(payload)
                records = normalized.get("records", [])

                for record in records:
                    refined = _normalize_record(record)
                    refined = _regex_fallback(block_text, refined)

                    missing_groups = _get_missing_critical_groups(refined)
                    missing_fields = _get_missing_field_names(refined)
                    if len(missing_groups) >= RETRY_MISSING_THRESHOLD:
                        retry_payload = _retry_missing_fields(
                            block_text,
                            refined,
                            missing_fields,
                            config,
                        )
                        if isinstance(retry_payload, dict):
                            for field in SCHEMA_FIELDS:
                                if not refined.get(field, "") and retry_payload.get(field, ""):
                                    refined[field] = retry_payload.get(field, "")
                            refined = _normalize_record(refined)
                            refined = _regex_fallback(block_text, refined)

                    score = _score_record(refined)
                    refined["NEEDS_REVIEW"] = "YES" if score < REVIEW_SCORE_THRESHOLD else ""
                    all_records.append(refined)

                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < config.max_retries:
                    logger.warning("OpenRouter request failed, retrying: %s", exc)
                    time.sleep(config.retry_backoff_sec * attempt)
                    continue
                logger.error("OpenRouter parsing failed after retries: %s", exc)
                raise OpenRouterParserError(f"OpenRouter parsing failed: {exc}") from exc

        if last_error is not None and attempt == config.max_retries:
            raise OpenRouterParserError(f"OpenRouter parsing failed: {last_error}")

    return {"columns": OUTPUT_COLUMNS, "records": all_records}


def parse_ocr_file_to_json(
    ocr_txt_path: str,
    output_json_path: str,
    config: Optional[OpenRouterConfig] = None,
) -> Dict[str, Any]:
    with open(ocr_txt_path, "r", encoding="utf-8") as file_handle:
        ocr_text = file_handle.read()

    payload = parse_ocr_to_structured_json(ocr_text, config=config)

    with open(output_json_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=True, indent=2)

    return payload
