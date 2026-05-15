"""Excel export utilities for OCR blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def build_row_groups(
    blocks: List[Dict[str, Any]],
    y_tolerance: float = 12.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group blocks into rows by Y center to preserve column layout."""
    sorted_blocks = sorted(blocks, key=lambda b: (b["y_center"], b["x_center"]))

    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = []
    current_y = None

    for block in sorted_blocks:
        if current_y is None:
            current_row = [block]
            current_y = block["y_center"]
            continue

        if abs(block["y_center"] - current_y) <= y_tolerance:
            current_row.append(block)
            # Update row center to be stable with dense text.
            current_y = sum(b["y_center"] for b in current_row) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda b: b["x_center"]))
            current_row = [block]
            current_y = block["y_center"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x_center"]))

    cell_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        row_texts = [b["text"] for b in row]
        row_text = " | ".join(row_texts)
        avg_conf = sum(b["confidence"] for b in row) / max(len(row), 1)
        row_bbox = [
            min(b["bbox_rect"][0] for b in row),
            min(b["bbox_rect"][1] for b in row),
            max(b["bbox_rect"][2] for b in row),
            max(b["bbox_rect"][3] for b in row),
        ]

        line_rows.append(
            {
                "row_index": row_index,
                "row_text": row_text,
                "avg_confidence": round(avg_conf, 4),
                "row_bbox_rect": row_bbox,
            }
        )

        for col_index, block in enumerate(row, start=1):
            cell_rows.append(
                {
                    "row_index": row_index,
                    "col_index": col_index,
                    "text": block["text"],
                    "confidence": round(block["confidence"], 4),
                    "bbox_rect": block["bbox_rect"],
                }
            )

    return cell_rows, line_rows


def format_lines_for_txt(line_rows: List[Dict[str, Any]]) -> List[str]:
    """Create human-readable lines for TXT output."""
    return [f"{row['row_index']:03d}: {row['row_text']}" for row in line_rows]


def export_to_excel(
    excel_path: str,
    cell_rows: List[Dict[str, Any]],
    line_rows: List[Dict[str, Any]],
) -> None:
    """Write OCR results to an Excel file."""
    cells_df = pd.DataFrame(cell_rows)
    lines_df = pd.DataFrame(line_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        cells_df.to_excel(writer, index=False, sheet_name="cells")
        lines_df.to_excel(writer, index=False, sheet_name="lines")
