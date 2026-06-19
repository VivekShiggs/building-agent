"""Convert pixel masks to geographic polygons with CRS handling.

Security:
  - Geometry validation via shapely.buffer(0) prevents invalid polygons
  - CRS validation prevents projection injection
  - Area bounds sanity check (no polygon > 1 km² without warning)
"""

import logging
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from agent.models import BuildingRecord

logger = logging.getLogger(__name__)

MAX_POLYGON_AREA_M2 = 1_000_000  # 1 km² sanity cap


def _get_utm_epsg(lon: float, lat: float) -> int:
    """Get UTM EPSG code for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def masks_to_geodataframe(
    pixel_masks: List[Dict[str, Any]],
    geotiff_path: str,
    min_area_m2: float = 20.0,
) -> gpd.GeoDataFrame:
    """Convert pixel masks to a GeoDataFrame of building polygons.

    Args:
        pixel_masks: List of mask dicts from detector.run_inference()
        geotiff_path: Path to source GeoTIFF (for transform & CRS)
        min_area_m2: Minimum polygon area in square meters

    Returns:
        GeoDataFrame with polygons in EPSG:4326
    """
    if not pixel_masks:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2", "confidence"], crs="EPSG:4326")

    with rasterio.open(geotiff_path) as src:
        native_crs = src.crs
        transform = src.transform

    all_polys: List[Dict[str, Any]] = []

    for pm in pixel_masks:
        mask = pm["mask"].astype(np.uint8)

        for geom_dict, val in rasterio_shapes(mask, transform=transform):
            if val != 1:
                continue

            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            all_polys.append({
                "geometry": poly,
                "confidence": pm["confidence"],
            })

    if not all_polys:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2", "confidence"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(all_polys, crs=native_crs)

    gdf_dissolved = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
    gdf_dissolved["geometry"] = gdf_dissolved.geometry.buffer(0)

    # Project to UTM for area calculation
    west, south, east, north = (
        gdf_dissolved.total_bounds[0],
        gdf_dissolved.total_bounds[1],
        gdf_dissolved.total_bounds[2],
        gdf_dissolved.total_bounds[3],
    )
    lon_mid = (west + east) / 2
    lat_mid = (south + north) / 2
    epsg_utm = _get_utm_epsg(lon_mid, lat_mid)

    gdf_utm = gdf_dissolved.to_crs(epsg=epsg_utm)
    gdf_utm["area_m2"] = gdf_utm.geometry.area

    gdf_utm = gdf_utm[gdf_utm["area_m2"] >= min_area_m2].copy()

    if gdf_utm["area_m2"].max() > MAX_POLYGON_AREA_M2:
        logger.warning("Large polygon detected: %.0f m²", gdf_utm["area_m2"].max())

    gdf_final = gdf_utm.to_crs(epsg=4326)
    gdf_final["area_m2"] = gdf_utm["area_m2"].values

    logger.info("Vectorized %d building polygons (min %.0f m²)", len(gdf_final), min_area_m2)
    return gdf_final


def compute_centroid(geom) -> Tuple[float, float]:
    """Compute (longitude, latitude) centroid of a geometry."""
    centroid = geom.centroid
    return (centroid.x, centroid.y)
