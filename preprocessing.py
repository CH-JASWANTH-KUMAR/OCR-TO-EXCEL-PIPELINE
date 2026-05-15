"""OpenCV-based preprocessing for OCR."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Resize while keeping aspect ratio to a dense-text OCR range."""
    height, width = image.shape[:2]
    max_dim = max(height, width)

    target_min = 2000
    target_max = 4200

    if max_dim < target_min:
        scale = target_min / max_dim
    elif max_dim > target_max:
        scale = target_max / max_dim
    else:
        scale = 1.0

    if scale == 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, new_size, interpolation=interpolation)


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)


def _deskew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size < 100:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return gray

    height, width = gray.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _binarize_for_projection(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return 255 - binary


def _find_column_splits(mask: np.ndarray) -> List[int]:
    height, width = mask.shape[:2]
    projection = mask.sum(axis=0)
    if projection.size == 0:
        return []

    window = max(15, width // 80)
    kernel = np.ones(window) / window
    smoothed = np.convolve(projection, kernel, mode="same")
    max_val = float(smoothed.max()) if smoothed.size else 0.0
    if max_val == 0.0:
        return []

    def _find_valley(center_ratio: float):
        center = int(width * center_ratio)
        half_window = int(width * 0.12)
        left = max(0, center - half_window)
        right = min(width, center + half_window)
        if right - left < 10:
            return None
        local_index = int(np.argmin(smoothed[left:right])) + left
        if smoothed[local_index] < max_val * 0.2:
            return local_index
        return None

    split1 = _find_valley(1 / 3)
    split2 = _find_valley(2 / 3)

    if split1 is None or split2 is None:
        return []
    if split2 - split1 < width * 0.2:
        return []
    return [split1, split2]


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image to improve OCR readability."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = _resize_for_ocr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=18, templateWindowSize=7, searchWindowSize=21)
    contrast = _enhance_contrast(denoised)
    sharpened = _sharpen(contrast)
    deskewed = _deskew(sharpened)
    return deskewed


def split_columns_for_ocr(gray: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """Detect a 3-column layout and return column images with x offsets."""
    mask = _binarize_for_projection(gray)
    splits = _find_column_splits(mask)
    height, width = gray.shape[:2]

    if not splits:
        return [(gray, 0)]

    boundaries = [0] + splits + [width]
    columns: List[Tuple[np.ndarray, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < width * 0.12:
            return [(gray, 0)]
        column = gray[:, start:end]
        columns.append((column, start))

    if len(columns) != 3:
        return [(gray, 0)]

    return columns
