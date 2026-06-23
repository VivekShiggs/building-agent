"""Land use classification from RGB aerial imagery.

Classifies every pixel as built-up, bare soil, vegetation, or water
using color-spectrum heuristics (NDVI proxy from RGB). Aggregates
contiguous patches into LandPatch records.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.models import LandPatch, LandUseClass

logger = logging.getLogger(__name__)

MIN_PATCH_AREA_M2 = 50


def _ndvi_proxy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a normalized vegetation index proxy from RGB.

    Uses (g - r) / (g + r + 1e-6) as a greenness proxy (similar to NDVI).
    Returns values in [-1, 1].
    """
    return (g.astype(np.float32) - r.astype(np.float32)) / (g + r + 1e-6)


def _bare_soil_index(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bare soil index: high when red+bright and low greenness."""
    brightness = (r + g + b) / 3
    greenness = g - r
    return (brightness / 255.0) * (1.0 - np.clip((g - r) / 255.0, 0, 1))


def classify_pixels(rgb: np.ndarray) -> np.ndarray:
    """Classify each pixel into a LandUseClass.

    Args:
        rgb: (H, W, 3) uint8 array

    Returns:
        (H, W) int array with LandUseClass values
    """
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    h, w = rgb.shape[:2]

    ndvi = _ndvi_proxy(r, g, b)
    bare = _bare_soil_index(r, g, b)
    brightness = (r + g + b) / 3.0

    result = np.full((h, w), LandUseClass.UNKNOWN.value, dtype=np.uint8)

    # Water: dark + low NDVI
    water_mask = (brightness < 50) & (ndvi < 0.0)
    result[water_mask] = LandUseClass.WATER.value

    # Vegetation: high greenness
    veg_mask = (g > r) & (g > b) & (ndvi > 0.15) & ~water_mask
    result[veg_mask] = LandUseClass.VEGETATION.value

    # Bare soil: bright, low veg, warm tint
    soil_mask = (bare > 0.35) & (ndvi < 0.1) & ~water_mask & ~veg_mask
    result[soil_mask] = LandUseClass.BARE_SOIL.value

    # Everything else is built-up
    result[(result == LandUseClass.UNKNOWN.value) & ~water_mask] = LandUseClass.BUILT_UP.value

    return result


def _find_contiguous_patches(
    classified: np.ndarray,
    target_class: LandUseClass,
    min_px: int = 50,
) -> List[Tuple[np.ndarray, int]]:
    """Extract contiguous patches of a given class using simple connected components.

    Args:
        classified: (H, W) int array of LandUseClass values
        target_class: Class to extract
        min_px: Minimum pixel count to keep

    Returns:
        List of (mask, n_pixels) for each patch
    """
    import cv2

    binary = (classified == target_class.value).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    patches: List[Tuple[np.ndarray, int]] = []
    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < min_px:
            continue
        mask = (labels == label_id)
        patches.append((mask, int(area_px)))

    patches.sort(key=lambda x: -x[1])
    return patches


def analyze_land_use(
    rgb: np.ndarray,
    geotiff_path: str,
    scan_id: str = "",
    min_area_m2: float = MIN_PATCH_AREA_M2,
) -> Tuple[List[LandPatch], Dict[str, float]]:
    """Run land use analysis on an RGB tile.

    Args:
        rgb: (H, W, 3) uint8 array
        geotiff_path: Path to source GeoTIFF for geo-referencing
        scan_id: Scan ID to associate patches with
        min_area_m2: Minimum patch area in m²

    Returns:
        (patches, class_area_m2) — list of LandPatch + area totals per class in m²
    """
    from agent.vectorize import masks_to_geodataframe

    classified = classify_pixels(rgb)

    class_labels = [LandUseClass.BUILT_UP, LandUseClass.VEGETATION,
                    LandUseClass.BARE_SOIL, LandUseClass.WATER]

    all_patches: List[LandPatch] = []
    class_area_m2: Dict[str, float] = {c.value: 0.0 for c in class_labels}

    for lu_class in class_labels:
        patches = _find_contiguous_patches(classified, lu_class, min_px=50)

        for mask, _ in patches:
            pixel_masks = [{"mask": mask, "class_id": lu_class.value,
                            "confidence": 0.5, "class_name": lu_class.value}]

            gdf = masks_to_geodataframe(pixel_masks, geotiff_path, min_area_m2=min_area_m2)

            if gdf.empty:
                continue

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                area = float(row.get("area_m2", 0))
                centroid = geom.centroid
                class_area_m2[lu_class.value] += area

                import uuid
                patch = LandPatch(
                    patch_id=f"p_{uuid.uuid4().hex[:12]}",
                    scan_id=scan_id,
                    land_use=lu_class,
                    latitude=centroid.y,
                    longitude=centroid.x,
                    area_m2=area,
                    compactness=min(4 * np.pi * area / (geom.length ** 2 + 1e-6), 1.0),
                    geometry_geojson=json.dumps({"type": "Polygon",
                                                  "coordinates": [list(geom.exterior.coords)]}),
                )
                all_patches.append(patch)

    return all_patches, class_area_m2
