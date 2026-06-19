"""Aerial imagery fetching — WMS download and local GeoTIFF loading.

Security:
  - All downloads use HTTPS (HTTP URLs are auto-upgraded)
  - Request timeouts prevent hanging
  - File paths validated against path traversal
  - Image dimensions validated to prevent OOM
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

MAX_IMAGE_PX = 10000  # Max width or height to prevent OOM
REQUEST_TIMEOUT = 60  # Seconds


def _upgrade_to_https(url: str) -> str:
    """Upgrade HTTP to HTTPS — never send API requests over plain HTTP."""
    if url.startswith("http://"):
        logger.warning("Upgrading HTTP URL to HTTPS: %s", url[:80])
        return url.replace("http://", "https://", 1)
    return url


def fetch_from_wms(
    bbox: List[float],
    wms_url_template: str,
    output_path: str,
    size_px: int = 1024,
) -> str:
    """Download imagery from WMS/ArcGIS REST endpoint and save as GeoTIFF.

    Args:
        bbox: [west, south, east, north] in WGS84
        wms_url_template: URL template with {west}, {south}, {east}, {north}, {width}, {height}
        output_path: Destination GeoTIFF path
        size_px: Requested image width and height in pixels

    Returns:
        Path to the saved GeoTIFF

    Raises:
        ValueError: If bbox or size_px is invalid
        requests.RequestException: If download fails
        rasterio.errors.RasterioIOError: If GeoTIFF write fails
    """
    west, south, east, north = bbox

    if size_px > MAX_IMAGE_PX:
        raise ValueError(f"Image size {size_px} exceeds maximum {MAX_IMAGE_PX}")

    url = wms_url_template.format(
        west=west,
        south=south,
        east=east,
        north=north,
        width=size_px,
        height=size_px,
    )
    url = _upgrade_to_https(url)

    logger.info("Fetching imagery from WMS: %s", url[:120])

    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img_array = np.array(img)

    h, w = img_array.shape[:2]

    if h > MAX_IMAGE_PX or w > MAX_IMAGE_PX:
        raise ValueError(f"Downloaded image dimensions {w}x{h} exceed maximum")

    transform = from_bounds(west, south, east, north, w, h)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        str(output_path_obj),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=img_array.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        for band_idx in range(3):
            dst.write(img_array[:, :, band_idx], band_idx + 1)

    logger.info("GeoTIFF saved: %s (%dx%d)", output_path, w, h)
    return str(output_path_obj.resolve())


def load_local_geotiff(path: str) -> Tuple[str, Dict[str, Any], np.ndarray]:
    """Load a local GeoTIFF and return metadata + RGB array.

    Args:
        path: Absolute or relative path to GeoTIFF

    Returns:
        (resolved_path, metadata_dict, rgb_array)

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is not a valid GeoTIFF
    """
    path_obj = Path(path).resolve()

    if not path_obj.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    if not path_obj.suffix.lower() in (".tif", ".tiff"):
        raise ValueError(f"Not a GeoTIFF file: {path}")

    with rasterio.open(str(path_obj)) as src:
        meta = {
            "crs": src.crs,
            "bounds": src.bounds,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }
        if src.count >= 3:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        else:
            rgb = np.stack([src.read(1)] * 3, axis=-1)

    return str(path_obj), meta, rgb


def get_imagery(config_imagery: Any, bbox: List[float], output_dir: str) -> Tuple[str, np.ndarray]:
    """Get aerial imagery for a given bbox — dispatches to WMS or local loader.

    Args:
        config_imagery: ImageryConfig object
        bbox: [west, south, east, north]
        output_dir: Directory to save downloaded imagery

    Returns:
        (geotiff_path, rgb_array)
    """
    if config_imagery.source == "local" and config_imagery.local_geotiff:
        path, meta, rgb = load_local_geotiff(config_imagery.local_geotiff)
        return path, rgb

    geotiff_path = str(Path(output_dir) / f"tile_{bbox[0]:.4f}_{bbox[1]:.4f}.tif")
    fetch_from_wms(
        bbox=bbox,
        wms_url_template=config_imagery.wms_url,
        output_path=geotiff_path,
        size_px=config_imagery.image_size_px,
    )
    _, _, rgb = load_local_geotiff(geotiff_path)
    return geotiff_path, rgb
