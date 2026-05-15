"""PaddleOCR wrapper and result parsing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from paddleocr import PaddleOCR


def build_ocr_engine(use_gpu: bool = False, lang: str = "en") -> PaddleOCR:
    """Create a reusable PaddleOCR instance."""
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        det_db_thresh=0.25,
        det_db_box_thresh=0.4,
        det_db_unclip_ratio=1.8,
        use_dilation=True,
    )


def _is_line_item(item: Any) -> bool:
    return (
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], list)
        and isinstance(item[1], (list, tuple))
    )


def run_ocr(ocr: PaddleOCR, image) -> Tuple[Any, List[Dict[str, Any]]]:
    """Run OCR and parse results into a list of blocks."""
    raw_result = ocr.ocr(image, cls=True)
    if not raw_result:
        return raw_result, []

    # PaddleOCR returns either a list of lines, or a list of lists (per image).
    if _is_line_item(raw_result[0]):
        lines = raw_result
    elif isinstance(raw_result[0], list) and raw_result[0] and _is_line_item(raw_result[0][0]):
        lines = raw_result[0]
    else:
        lines = []

    blocks: List[Dict[str, Any]] = []
    for line in lines:
        box, (text, confidence) = line
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        bbox_rect = [min(xs), min(ys), max(xs), max(ys)]
        x_center = (bbox_rect[0] + bbox_rect[2]) / 2
        y_center = (bbox_rect[1] + bbox_rect[3]) / 2

        blocks.append(
            {
                "text": text,
                "confidence": float(confidence),
                "bbox": box,
                "bbox_rect": bbox_rect,
                "x_center": x_center,
                "y_center": y_center,
            }
        )

    return raw_result, blocks
