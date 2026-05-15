"""Hybrid OCR + OpenRouter pipeline.

Setup (Windows PowerShell):
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt

Add .env:
	OPENROUTER_API_KEY=your_api_key

Run:
	python main.py
"""

from __future__ import annotations

import csv
from collections import Counter
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import pandas as pd

from excel_export import build_row_groups, format_lines_for_txt
from evaluation_utils import evaluate_payload
from openrouter_parser import OUTPUT_COLUMNS, SCHEMA_FIELDS, parse_ocr_file_to_json
from ocr_engine import build_ocr_engine, run_ocr
from preprocessing import preprocess_image, split_columns_for_ocr
from structured_excel_export import export_structured_json_to_excel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CONFIDENCE_THRESHOLD = 0.6
DISCARD_CONFIDENCE_THRESHOLD = 0.2
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.5
ROW_Y_TOLERANCE = 12.0
COLUMN_SEGMENTATION_ENABLED = True
PROCESS_LIMIT_IMAGES = 3



def _setup_logger(log_dir: Path) -> logging.Logger:
	logger = logging.getLogger("ocr_pipeline")
	logger.setLevel(logging.INFO)

	log_path = log_dir / "pipeline.log"
	handler = logging.FileHandler(log_path, encoding="utf-8")
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	handler.setFormatter(formatter)

	if logger.handlers:
		logger.handlers.clear()
	logger.addHandler(handler)
	logger.addHandler(logging.StreamHandler())

	return logger


def _save_txt(path: Path, lines: List[str]) -> None:
	with path.open("w", encoding="utf-8") as file_handle:
		file_handle.write("\n".join(lines))


def _collect_images(images_dir: Path) -> List[Path]:
	return [
		path
		for path in images_dir.iterdir()
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]



def _create_run_dirs(base_dir: Path) -> dict:
	outputs_dir = base_dir / "outputs"
	runs_dir = outputs_dir / "runs"
	history_dir = outputs_dir / "history"

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = runs_dir / timestamp
	index = 1
	while run_dir.exists():
		run_dir = runs_dir / f"{timestamp}_{index}"
		index += 1

	eval_dir = run_dir / "evaluation"
	structured_dir = run_dir / "structured_json"
	final_dir = run_dir / "final_excel"
	log_dir = run_dir / "logs"
	debug_dir = run_dir / "debug"

	for directory in [eval_dir, structured_dir, final_dir, log_dir, debug_dir, history_dir]:
		directory.mkdir(parents=True, exist_ok=True)

	return {
		"outputs": outputs_dir,
		"runs": runs_dir,
		"history": history_dir,
		"run": run_dir,
		"structured": structured_dir,
		"final": final_dir,
		"logs": log_dir,
		"debug": debug_dir,
		"evaluation": eval_dir,
	}


def _append_evaluation_history(history_path: Path, rows: List[dict]) -> None:
	if not rows:
		return

	write_header = not history_path.exists()
	with history_path.open("a", encoding="utf-8", newline="") as file_handle:
		fieldnames = list(rows[0].keys())
		writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerows(rows)


def _write_master_excel(
	master_path: Path,
	records: List[dict],
	columns: List[str],
) -> None:
	if not records:
		return

	df = pd.DataFrame(records)
	for column in columns:
		if column not in df.columns:
			df[column] = ""

	df = df[columns]
	with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
		df.to_excel(writer, index=False, sheet_name="data")


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	images_dir = base_dir / "images"
	output_dirs = _create_run_dirs(base_dir)

	logger = _setup_logger(output_dirs["logs"])

	if not images_dir.exists():
		logger.error("Missing images folder.")
		return

	images = _collect_images(images_dir)
	if not images:
		print("No images found in the images folder.")
		return

	ocr_engine = build_ocr_engine(use_gpu=False, lang="en")

	eval_rows = []
	missing_field_counter = Counter()
	invalid_email_total = 0
	invalid_phone_total = 0

	if PROCESS_LIMIT_IMAGES and PROCESS_LIMIT_IMAGES > 0:
		images = images[:PROCESS_LIMIT_IMAGES]
		logger.info("Processing limited to %s images", PROCESS_LIMIT_IMAGES)

	master_records = []
	run_id = output_dirs["run"].name

	for image_path in images:
		try:
			preprocessed = preprocess_image(str(image_path))
			base_name = image_path.stem

			debug_preprocessed_path = output_dirs["debug"] / f"{base_name}_preprocessed.png"
			cv2.imwrite(str(debug_preprocessed_path), preprocessed)

			columns = (
				split_columns_for_ocr(preprocessed)
				if COLUMN_SEGMENTATION_ENABLED
				else [(preprocessed, 0)]
			)

			blocks = []
			raw_result = []
			for col_index, (column_image, x_offset) in enumerate(columns, start=1):
				if len(columns) > 1:
					debug_column_path = output_dirs["debug"] / f"{base_name}_col{col_index}.png"
					cv2.imwrite(str(debug_column_path), column_image)

				col_raw, col_blocks = run_ocr(ocr_engine, column_image)
				raw_result.append(col_raw)

				for block in col_blocks:
					block["bbox"] = [[point[0] + x_offset, point[1]] for point in block["bbox"]]
					block["bbox_rect"] = [
						block["bbox_rect"][0] + x_offset,
						block["bbox_rect"][1],
						block["bbox_rect"][2] + x_offset,
						block["bbox_rect"][3],
					]
					block["x_center"] += x_offset
					blocks.append(block)

			if not blocks:
				logger.warning(f"No OCR blocks extracted: {image_path.name}")
				continue

			low_conf = [b for b in blocks if b["confidence"] < CONFIDENCE_THRESHOLD]
			avg_conf = sum(b["confidence"] for b in blocks) / max(len(blocks), 1)
			min_conf = min(b["confidence"] for b in blocks)
			max_conf = max(b["confidence"] for b in blocks)

			print(
				f"{image_path.name} | blocks={len(blocks)} | "
				f"avg={avg_conf:.3f} min={min_conf:.3f} max={max_conf:.3f}"
			)

			if len(low_conf) == len(blocks):
				logger.warning(
					f"All blocks below confidence threshold ({CONFIDENCE_THRESHOLD}): "
					f"{image_path.name}"
				)

			suspicious_blocks = [
				b
				for b in blocks
				if DISCARD_CONFIDENCE_THRESHOLD <= b["confidence"] < SUSPICIOUS_CONFIDENCE_THRESHOLD
			]
			for block in suspicious_blocks[:20]:
				logger.warning(
					"Suspicious OCR block (conf=%.3f): %s",
					block["confidence"],
					block["text"],
				)

			blocks = [b for b in blocks if b["confidence"] >= DISCARD_CONFIDENCE_THRESHOLD]
			blocks = sorted(blocks, key=lambda b: (b["y_center"], b["x_center"]))

			if not blocks:
				logger.warning(
					"All blocks discarded after confidence filtering: %s",
					image_path.name,
				)
				continue

			cell_rows, line_rows = build_row_groups(blocks, y_tolerance=ROW_Y_TOLERANCE)
			txt_lines = format_lines_for_txt(line_rows)

			ocr_json_path = output_dirs["evaluation"] / f"{base_name}.ocr.json"
			ocr_txt_path = output_dirs["evaluation"] / f"{base_name}.txt"
			structured_json_path = output_dirs["structured"] / f"{base_name}.structured.json"
			final_excel_path = output_dirs["final"] / f"{base_name}.xlsx"

			eval_raw_path = output_dirs["evaluation"] / f"{base_name}_raw.txt"
			eval_cleaned_path = output_dirs["evaluation"] / f"{base_name}_cleaned.txt"
			eval_structured_path = output_dirs["evaluation"] / f"{base_name}_structured.json"

			ocr_payload = {
				"image": image_path.name,
				"confidence_threshold": CONFIDENCE_THRESHOLD,
				"raw_result": raw_result,
				"blocks": blocks,
			}
			with ocr_json_path.open("w", encoding="utf-8") as file_handle:
				json.dump(ocr_payload, file_handle, ensure_ascii=True, indent=2)
			_save_txt(ocr_txt_path, txt_lines)

			raw_text_lines = [block["text"] for block in blocks]
			with eval_raw_path.open("w", encoding="utf-8") as file_handle:
				file_handle.write("\n".join(raw_text_lines))
			with eval_cleaned_path.open("w", encoding="utf-8") as file_handle:
				file_handle.write("\n".join(txt_lines))

			structured_payload = parse_ocr_file_to_json(
				str(ocr_txt_path),
				str(structured_json_path),
			)
			for record in structured_payload.get("records", []):
				record["SOURCE_IMAGE"] = image_path.name
				if "NEEDS_REVIEW" not in record:
					record["NEEDS_REVIEW"] = ""
			master_records.extend(structured_payload.get("records", []))
			with eval_structured_path.open("w", encoding="utf-8") as file_handle:
				json.dump(structured_payload, file_handle, ensure_ascii=True, indent=2)
			export_structured_json_to_excel(
				str(structured_json_path),
				str(final_excel_path),
			)

			evaluation = evaluate_payload(structured_payload)
			row_count = evaluation.total_records

			eval_rows.append(
				{
					"image": image_path.name,
					"raw_text_path": str(eval_raw_path),
					"cleaned_text_path": str(eval_cleaned_path),
					"structured_json_path": str(eval_structured_path),
					"excel_rows": row_count,
					"total_records": evaluation.total_records,
					"valid_emails": evaluation.valid_emails,
					"valid_phones": evaluation.valid_phones,
					"invalid_emails": evaluation.invalid_email_count,
					"invalid_phones": evaluation.invalid_phone_count,
					"missing_critical_fields": evaluation.missing_critical_fields,
					"needs_review": evaluation.needs_review,
					"extraction_score": evaluation.average_extraction_score,
					"run_id": run_id,
				}
			)

			missing_field_counter.update(evaluation.missing_fields_counter)
			invalid_email_total += evaluation.invalid_email_count
			invalid_phone_total += evaluation.invalid_phone_count

		except Exception as exc:
			logger.exception(f"Failed processing {image_path.name}: {exc}")

	evaluation_report_path = output_dirs["run"] / "evaluation_summary.csv"
	if eval_rows:
		with evaluation_report_path.open("w", encoding="utf-8", newline="") as file_handle:
			fieldnames = list(eval_rows[0].keys())
			writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(eval_rows)
		logger.info("Saved evaluation report: %s", evaluation_report_path)

		aggregate = {
			"image": "TOTAL",
			"raw_text_path": "",
			"cleaned_text_path": "",
			"structured_json_path": "",
			"excel_rows": sum(row["excel_rows"] for row in eval_rows),
			"total_records": sum(row["total_records"] for row in eval_rows),
			"valid_emails": sum(row["valid_emails"] for row in eval_rows),
			"valid_phones": sum(row["valid_phones"] for row in eval_rows),
			"invalid_emails": sum(row["invalid_emails"] for row in eval_rows),
			"invalid_phones": sum(row["invalid_phones"] for row in eval_rows),
			"missing_critical_fields": sum(row["missing_critical_fields"] for row in eval_rows),
			"needs_review": sum(row["needs_review"] for row in eval_rows),
			"extraction_score": int(
				round(sum(row["extraction_score"] for row in eval_rows) / len(eval_rows))
			),
			"run_id": run_id,
		}
		eval_rows.append(aggregate)
		with evaluation_report_path.open("a", encoding="utf-8", newline="") as file_handle:
			writer = csv.DictWriter(file_handle, fieldnames=list(eval_rows[0].keys()))
			writer.writerow(aggregate)

	history_path = output_dirs["history"] / "evaluation_history.csv"
	_append_evaluation_history(history_path, eval_rows)

	master_columns = OUTPUT_COLUMNS + ["SOURCE_IMAGE"]
	master_path = output_dirs["run"] / "master_output.xlsx"
	_write_master_excel(master_path, master_records, master_columns)
	logger.info("Saved master Excel: %s", master_path)

	if missing_field_counter:
		sorted_missing = sorted(
			missing_field_counter.items(),
			key=lambda item: item[1],
			reverse=True,
		)
		logger.info("Top missing critical fields:")
		for field_name, count in sorted_missing[:5]:
			logger.info("- %s: %s", field_name, count)

	if invalid_email_total or invalid_phone_total:
		logger.info(
			"Invalid email count: %s | Invalid phone count: %s",
			invalid_email_total,
			invalid_phone_total,
		)


if __name__ == "__main__":
	main()
