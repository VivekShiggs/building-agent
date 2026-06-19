"""Building classification — shape, size, color, and roof type analysis.

All heuristics are explainable and deterministic (no ML-based classification)
to keep inference fast and reproducible.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import rasterio
from shapely.geometry import Polygon

from agent.models import (
    BuildingType,
    RoofType,
    ShapeClass,
    SizeClass,
)

logger = logging.getLogger(__name__)


def classify_shape(polygon: Polygon) -> Tuple[ShapeClass, float, float, float]:
    """Classify building shape based on geometric properties.

    Uses:
      - compactness = 4πA / P²  (1.0 = perfect circle)
      - rectangularity = area / (min_rotated_rect_area)
      - eccentricity = major_axis / minor_axis

    Returns:
        (shape_class, compactness, rectangularity, eccentricity)
    """
    area = polygon.area
    perimeter = polygon.length

    if area <= 0 or perimeter <= 0:
        return (ShapeClass.REGULAR, 0.0, 0.0, 1.0)

    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    compactness = min(max(compactness, 0.0), 1.0)

    # Rectangularity: compute oriented minimum bounding rectangle
    try:
        mbr = polygon.minimum_rotated_rectangle
        mbr_area = mbr.area if mbr and mbr.area > 0 else area
        rectangularity = min(area / mbr_area, 1.0) if mbr_area > 0 else 0.5
    except Exception:
        rectangularity = 0.5

    if compactness >= 0.7:
        shape_class = ShapeClass.REGULAR
    elif compactness >= 0.4:
        shape_class = ShapeClass.IRREGULAR
    else:
        shape_class = ShapeClass.COMPLEX

    return (shape_class, compactness, rectangularity, 1.0 / max(compactness, 0.01))


def classify_size(area_m2: float, config_size: Any) -> SizeClass:
    """Classify building size based on footprint area."""
    if area_m2 <= config_size.small_max:
        return SizeClass.SMALL
    elif area_m2 <= config_size.medium_max:
        return SizeClass.MEDIUM
    else:
        return SizeClass.LARGE


def analyze_color(
    geometry_geojson: str,
    geotiff_path: str,
) -> Tuple[float, float, float, str, RoofType]:
    """Analyze building color from imagery by sampling pixel values under the mask.

    Args:
        geometry_geojson: GeoJSON string of the building polygon
        geotiff_path: Path to source GeoTIFF

    Returns:
        (mean_r, mean_g, mean_b, dominant_color_name, roof_type)
    """
    import json
    from shapely.geometry import shape as shapely_shape
    import rasterio.mask

    try:
        geom = shapely_shape(json.loads(geometry_geojson))
    except Exception:
        return (0.0, 0.0, 0.0, "unknown", RoofType.UNKNOWN)

    try:
        with rasterio.open(geotiff_path) as src:
            out_image, _ = rasterio.mask.mask(src, [geom], crop=True, all_touched=True)
            pixels = out_image[:, :, :].transpose(1, 2, 0)
            valid = pixels[..., 0] > 0

            if not valid.any():
                return (0.0, 0.0, 0.0, "unknown", RoofType.UNKNOWN)

            r_vals = pixels[..., 0][valid]
            g_vals = pixels[..., 1][valid]
            b_vals = pixels[..., 2][valid]

            mean_r = float(np.mean(r_vals))
            mean_g = float(np.mean(g_vals))
            mean_b = float(np.mean(b_vals))

            # Dominant color heuristic
            brightness = (mean_r + mean_g + mean_b) / 3
            if brightness > 200:
                dom = "white"
                roof = RoofType.LIGHT
            elif brightness < 60:
                dom = "dark"
                roof = RoofType.DARK
            elif mean_r > 150 and mean_g < 100 and mean_b < 100:
                dom = "red"
                roof = RoofType.TILE
            elif mean_b > mean_r and mean_b > mean_g and brightness > 100:
                dom = "blue"
                roof = RoofType.METAL
            elif mean_g > mean_r and mean_g > mean_b and brightness > 80:
                dom = "green"
                roof = RoofType.GREEN
            elif 100 < mean_r < 180 and 100 < mean_g < 180 and 100 < mean_b < 180:
                dom = "grey"
                roof = RoofType.CONCRETE
            else:
                dom = "other"
                roof = RoofType.UNKNOWN

            return (float(mean_r), float(mean_g), float(mean_b), dom, roof)

    except Exception as e:
        logger.debug("Color analysis failed: %s", e)
        return (0.0, 0.0, 0.0, "unknown", RoofType.UNKNOWN)


def classify_building_type(
    osm_building_type: Optional[str],
    shape_class: ShapeClass,
    size_class: SizeClass,
    area_m2: float,
) -> Tuple[BuildingType, str]:
    """Determine building type from OSM tags and geometric heuristics.

    Args:
        osm_building_type: building=* tag value from OSM (if available)
        shape_class: Shape classification
        size_class: Size classification
        area_m2: Footprint area

    Returns:
        (building_type, source) where source is "osm", "heuristic", or "default"
    """
    if osm_building_type and osm_building_type != "yes":
        type_map = {
            "house": BuildingType.HOUSE,
            "residential": BuildingType.APARTMENT,
            "apartments": BuildingType.APARTMENT,
            "garage": BuildingType.GARAGE,
            "garages": BuildingType.GARAGE,
            "commercial": BuildingType.COMMERCIAL,
            "retail": BuildingType.COMMERCIAL,
            "industrial": BuildingType.INDUSTRIAL,
            "warehouse": BuildingType.INDUSTRIAL,
            "shed": BuildingType.SHED,
            "church": BuildingType.CHURCH,
            "school": BuildingType.SCHOOL,
            "hospital": BuildingType.HOSPITAL,
            "chapel": BuildingType.CHURCH,
            "cathedral": BuildingType.CHURCH,
            "university": BuildingType.SCHOOL,
            "kindergarten": BuildingType.SCHOOL,
            "office": BuildingType.COMMERCIAL,
            "supermarket": BuildingType.COMMERCIAL,
            "farm": BuildingType.INDUSTRIAL,
            "stable": BuildingType.SHED,
            "cabin": BuildingType.HOUSE,
            "bungalow": BuildingType.HOUSE,
            "detached": BuildingType.HOUSE,
            "semidetached_house": BuildingType.HOUSE,
            "terrace": BuildingType.HOUSE,
            "static_caravan": BuildingType.HOUSE,
        }
        mapped = type_map.get(osm_building_type.lower())
        if mapped:
            return (mapped, "osm")
        return (BuildingType.OTHER, "osm")

    # Heuristic fallback
    if size_class == SizeClass.LARGE:
        if area_m2 > 500:
            return (BuildingType.INDUSTRIAL, "heuristic")
        return (BuildingType.COMMERCIAL, "heuristic")

    if size_class == SizeClass.MEDIUM:
        return (BuildingType.APARTMENT, "heuristic")

    if shape_class == ShapeClass.REGULAR:
        return (BuildingType.HOUSE, "heuristic")

    if shape_class == ShapeClass.IRREGULAR:
        return (BuildingType.OTHER, "heuristic")

    return (BuildingType.UNKNOWN, "default")
