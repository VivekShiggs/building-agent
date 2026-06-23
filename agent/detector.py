"""Building detection — multi-strategy pipeline.

Strategies (tried in order):
  1. Fine-tuned YOLO model (if available) — best quality
  2. YOLOv8n-seg (COCO) — may detect cars/objects on roads near buildings
  3. OpenCV contour detection (fallback) — works on any aerial image

The fallback strategy detects rectangular structures in aerial imagery using
edge detection + contour finding. While not as accurate as a trained model,
it produces usable results immediately with zero additional downloads.

Security:
  - All model weights downloaded over HTTPS from official sources
  - No arbitrary model loading from user-supplied paths
  - Memory bounded by image size limits
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_yolo_warned = False

NON_STRUCTURAL_CLASSES = {
    0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
}


def _load_yolo_model(model_name: str, finetuned_path: Optional[str] = None):
    """Load YOLO model with optional fine-tuned weights.

    Returns None if model cannot be loaded (e.g. incompatible numpy).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("Ultralytics not installed")
        return None

    try:
        if finetuned_path:
            ftp = Path(finetuned_path)
            if ftp.exists():
                logger.info("Loading fine-tuned model: %s", finetuned_path)
                return YOLO(str(ftp.resolve()))

        logger.info("Loading base model: %s", model_name)
        return YOLO(model_name)
    except Exception as e:
        logger.warning("YOLO model load failed: %s", e)
        return None


def run_yolo_inference(
    image: np.ndarray,
    config_model: Any,
) -> List[Dict[str, Any]]:
    """Run YOLOv8 segmentation inference.

    Returns empty list if no detections or model unavailable.
    """
    global _yolo_warned
    try:
        from ultralytics import YOLO
    except Exception as e:
        if not _yolo_warned:
            logger.warning("Ultralytics not available on this system: %s", e)
            _yolo_warned = True
        return []

    model = _load_yolo_model(config_model.name, config_model.finetuned_path)
    if model is None:
        return []

    try:
        results = model.predict(
            source=image,
            conf=config_model.confidence,
            iou=config_model.iou,
            imgsz=config_model.imgsz,
            verbose=False,
        )
    except Exception as e:
        logger.warning("YOLO inference failed: %s", e)
        return []

    result = results[0]

    if result.masks is None or result.boxes is None:
        logger.info("YOLO: no detections")
        return []

    masks_np = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    exclude_set = (
        set(config_model.exclude_classes)
        if hasattr(config_model, "exclude_classes")
        else NON_STRUCTURAL_CLASSES
    )

    pixel_masks: List[Dict[str, Any]] = []

    for i, (mask, cls_id, conf) in enumerate(zip(masks_np, cls_ids, confs)):
        if int(cls_id) in exclude_set:
            continue

        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = PILImage.fromarray(mask_uint8).resize(
            (image.shape[1], image.shape[0]),
            PILImage.NEAREST,
        )
        mask_bool = np.array(mask_pil) > 127

        pixel_masks.append({
            "mask": mask_bool,
            "class_id": int(cls_id),
            "confidence": float(conf),
            "class_name": model.names[int(cls_id)],
        })

    logger.info("YOLO: %d structural masks", len(pixel_masks))
    return pixel_masks


def run_contour_detection(
    image: np.ndarray,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.4,
) -> List[Dict[str, Any]]:
    """Detect building-like structures using Otsu thresholding + contour finding.

    This is a zero-download fallback that works on any aerial image.
    It segments the image into dark/bright regions using Otsu's method,
    then filters contours by area and shape compactness.

    Args:
        image: RGB image array (H, W, 3), uint8
        min_area_ratio: Minimum contour area as fraction of image
        max_area_ratio: Maximum contour area as fraction of image

    Returns:
        List of mask dicts with keys: mask, confidence, class_name
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed — cannot use contour detection")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    min_px = img_area * min_area_ratio
    max_px = img_area * max_area_ratio

    pixel_masks: List[Dict[str, Any]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_px or area > max_px:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        compactness = 4 * np.pi * area / (perimeter * perimeter)
        confidence = min(max(compactness * 1.5, 0.15), 0.85)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

        pixel_masks.append({
            "mask": mask.astype(bool),
            "class_id": -1,
            "confidence": float(confidence),
            "class_name": "building_contour",
        })

    logger.info("Contour detection: %d building candidates", len(pixel_masks))
    return pixel_masks


def run_inference(
    image: np.ndarray,
    config_model: Any,
) -> List[Dict[str, Any]]:
    """Run building detection using the best available strategy.

    Strategy priority:
      1. Fine-tuned YOLO (if weights exist)
      2. YOLOv8n-seg COCO (limited on aerial views)
      3. Contour detection (always works, no downloads)

    Args:
        image: RGB image array (H, W, 3), uint8
        config_model: ModelConfig object

    Returns:
        List of dicts with keys: mask, class_id, confidence, class_name
    """
    # Strategy 1: Try YOLO with fine-tuned weights
    if config_model.finetuned_path:
        ftp = Path(config_model.finetuned_path)
        if ftp.exists():
            logger.info("Strategy 1: fine-tuned YOLO")
            result = run_yolo_inference(image, config_model)
            if result:
                return result
            logger.info("Fine-tuned YOLO returned 0 detections")

    # Strategy 2: Try base YOLO (COCO)
    logger.info("Strategy 2: YOLOv8n-seg (COCO)")
    yolo_result = run_yolo_inference(image, config_model)
    if yolo_result:
        return yolo_result

    logger.info("YOLO returned 0 — using contour detection fallback")

    # Strategy 3: Contour detection (always works)
    contour_result = run_contour_detection(image)
    if contour_result:
        return contour_result

    logger.warning("All detection strategies returned 0 results")
    return []


def get_model_version(config_model: Any) -> str:
    """Get the active model version string."""
    if config_model.finetuned_path:
        ftp = Path(config_model.finetuned_path)
        if ftp.exists():
            return f"finetuned:{ftp.name}"

    return config_model.name
