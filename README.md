# OCR-TO-EXCEL-PIPELINE

An end-to-end OCR and AI extraction pipeline that converts document images into structured JSON and Excel outputs. Built for professional evaluation, benchmarking, and repeatable runs.

## Project Overview

This project implements a complete flow from raw images to structured Excel outputs:

- OCR → AI extraction → Excel automation
- Designed for medical/customer form extraction and similar semi-structured documents
- Uses PaddleOCR for text detection + recognition
- Uses OpenRouter (LLM routing) for structured parsing
- Includes an evaluation and benchmarking system for accuracy and confidence scoring

## Features

- Image preprocessing (resize, denoise, binarization)
- Deskewing and alignment
- Column segmentation
- OCR extraction with PaddleOCR
- OpenRouter structured parsing (LLM-based)
- Excel export (raw + structured)
- Evaluation reports and scoring
- Confidence scoring and NEEDS_REVIEW tagging
- Persistent run history
- Master aggregation across runs

## System Architecture

```
Image
↓
Preprocessing
↓
PaddleOCR
↓
Segmentation
↓
OpenRouter/Qwen
↓
Validation + Cleanup
↓
Structured JSON
↓
Excel Export
↓
Evaluation Reports
```

## Project Structure

Key files and folders:

- [images/](images/) — Input images to process.
- [outputs/](outputs/) — All run outputs, structured JSON, evaluation files, logs, and historical results.
- [backup_images/](backup_images/) — Optional archive for processed images.
- [main.py](main.py) — Pipeline entry point (orchestrates the full run).
- [preprocessing.py](preprocessing.py) — Image cleanup, deskewing, segmentation helpers.
- [ocr_engine.py](ocr_engine.py) — OCR execution logic using PaddleOCR.
- [openrouter_parser.py](openrouter_parser.py) — LLM routing and JSON parsing using OpenRouter.
- [evaluation_utils.py](evaluation_utils.py) — Scoring, validation, and reporting utilities.
- [excel_export.py](excel_export.py) — Exports raw OCR results to Excel.
- [structured_excel_export.py](structured_excel_export.py) — Exports structured JSON to Excel.
- [requirements.txt](requirements.txt) — Python dependencies.

## Requirements

- Python 3.9+ (recommended)
- OpenRouter API key (for LLM-based structured parsing)
- Active internet connection (LLM calls)

## Installation Guide

### 1. Clone repository

```bash
git clone https://github.com/your-username/OCR-TO-EXCEL-PIPELINE.git
cd OCR-TO-EXCEL-PIPELINE
```

### 2. Create virtual environment (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

If you already use environment variables globally, you can set `OPENROUTER_API_KEY` in your OS instead.

## How To Run

1. Place input images inside [images/](images/).
2. Run the pipeline:

```bash
python main.py
```

## Output Structure

Each run creates a timestamped folder under `outputs/runs/{timestamp}/` with the following structure:

```
outputs/
	runs/
		{timestamp}/
			master_output.xlsx
			evaluation_summary.csv
			structured_json/
			debug/
			logs/
```

What each output means:

- `master_output.xlsx` — Aggregated structured data across inputs.
- `evaluation_summary.csv` — Metrics and scoring summary for the run.
- `structured_json/` — Per-file structured JSON outputs.
- `debug/` — Intermediate OCR/LLM artifacts for troubleshooting.
- `logs/` — Runtime logs and pipeline traces.

## Evaluation System

The evaluation module provides automated scoring and reliability metrics:

- Extraction score computed against validation rules
- `NEEDS_REVIEW` flag when confidence or validation thresholds are low
- Validation metrics for required fields and formatting
- Confidence scoring to help prioritize manual QA

## Supported Models

This project works with any OpenRouter-supported model. Recommended models:

- `qwen/qwen-2.5-72b-instruct` — strong accuracy, good for complex forms
- `anthropic/claude-3.5-sonnet` — consistent output, higher cost
- `deepseek/deepseek-chat` — balanced price/performance

Cost/performance tradeoffs:

- Larger models increase accuracy but raise cost and latency.
- Smaller models are cheaper but may require more validation cleanup.

## Common Issues & Fixes

- OpenRouter API errors:
	- Confirm `OPENROUTER_API_KEY` is set correctly.
	- Ensure your account has active credits.
- Protobuf issues (PaddleOCR):
	- Try `pip install --upgrade protobuf` and re-install dependencies.
- NumPy/OpenCV compatibility:
	- Ensure `numpy` and `opencv-python` versions align with requirements.
	- Recreate the virtual environment if conflicts persist.
- GitHub setup issues:
	- Verify you cloned the repository correctly and are in the project root.
- Missing API keys:
	- Create `.env` and set `OPENROUTER_API_KEY`.

## Future Improvements

- Layout-aware extraction (table/field detection)
- Positional OCR intelligence for multi-column documents
- Model benchmarking dashboard
- Parallel processing for batch runs
- Human review UI for flagged items
- Custom OCR model fine-tuning

## Technologies Used

- Python
- PaddleOCR
- OpenCV
- OpenRouter
- Pandas
- Regex
- Excel automation

## Disclaimer

This project uses experimental AI extraction and OCR. Results can vary based on image quality and document layout. Human verification is recommended for critical data workflows.