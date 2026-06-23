"""City audit — aggregate all analysis into government-facing KPIs.

Runs land use classification, solar potential, farming suitability,
and recommendation generation across an entire scan, producing a
CityKPI record with summary metrics and actionable suggestions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.farming import estimate_crop_yield, score_patch_for_farming
from agent.land_use import analyze_land_use
from agent.models import (
    BuildingRecord,
    CityKPI,
    LandPatch,
    LandUseClass,
    Recommendation,
    RecommendationType,
)
from agent.recommendations import generate_recommendations
from agent.solar import estimate_open_land_solar, estimate_roof_solar, score_patch_for_solar
from agent.store import BuildingStore

logger = logging.getLogger(__name__)


def run_city_audit(
    rgb: np.ndarray,
    geotiff_path: str,
    scan_id: str,
    buildings: List[BuildingRecord],
    store: BuildingStore,
    min_area_m2: float = 50.0,
) -> CityKPI:
    """Execute full city audit on a single tile: land use + solar + farming + recommendations.

    Args:
        rgb: (H, W, 3) uint8 image from the tile
        geotiff_path: Path to the GeoTIFF for geo-referencing
        scan_id: Current scan ID
        buildings: Building records detected in this scan
        store: BuildingStore for saving results
        min_area_m2: Minimum patch area for land use analysis

    Returns:
        CityKPI record with all KPIs populated
    """
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Land use classification
    logger.info("Running land use classification...")
    land_patches, class_area_m2 = analyze_land_use(rgb, geotiff_path, min_area_m2=min_area_m2)

    total_area_m2 = sum(class_area_m2.values())
    built_up_m2 = class_area_m2.get(LandUseClass.BUILT_UP.value, 0.0)
    bare_soil_m2 = class_area_m2.get(LandUseClass.BARE_SOIL.value, 0.0)
    vegetation_m2 = class_area_m2.get(LandUseClass.VEGETATION.value, 0.0)
    water_m2 = class_area_m2.get(LandUseClass.WATER.value, 0.0)

    # Unused = bare soil + unknown (not built-up, not veg, not water)
    unused_land_m2 = bare_soil_m2 + class_area_m2.get(LandUseClass.UNKNOWN.value, 0.0)

    # Step 2: Solar potential
    logger.info("Calculating solar potential...")
    roof_mw, roof_kwh, roof_co2 = estimate_roof_solar(buildings)

    # Only bare soil patches for ground-mount solar
    solar_patches = [p for p in land_patches if p.land_use == LandUseClass.BARE_SOIL]
    land_mw, land_kwh, land_co2 = estimate_open_land_solar(solar_patches, usage_ratio=0.5)

    total_mw = roof_mw + land_mw
    total_kwh = roof_kwh + land_kwh
    total_co2 = roof_co2 + land_co2

    # Step 3: Farming suitability
    logger.info("Assessing farming suitability...")
    farm_patches = [p for p in land_patches if p.land_use in (
        LandUseClass.BARE_SOIL, LandUseClass.VEGETATION)]
    farmable_m2 = sum(p.area_m2 for p in farm_patches
                      if score_patch_for_farming(p) > 0.3)
    farmable_ha = farmable_m2 / 10000.0

    # Estimate yield (mix of community + commercial)
    total_yield = 0.0
    for p in farm_patches:
        farm_type = (RecommendationType.COMMERCIAL_FARM
                     if p.area_m2 >= 5000
                     else RecommendationType.COMMUNITY_FARM)
        total_yield += estimate_crop_yield(p.area_m2, farm_type)

    # Step 4: Generate recommendations
    logger.info("Generating recommendations...")
    recs = generate_recommendations(buildings, land_patches, scan_id)

    # Save recommendations as JSON
    recs_json = json.dumps([r.model_dump() for r in recs])

    # Build KPI record
    total_ha = total_area_m2 / 10000.0
    kpi = CityKPI(
        scan_id=scan_id,
        total_area_ha=total_ha,
        built_up_ha=built_up_m2 / 10000.0,
        bare_soil_ha=bare_soil_m2 / 10000.0,
        vegetation_ha=vegetation_m2 / 10000.0,
        water_ha=water_m2 / 10000.0,
        unused_land_ha=unused_land_m2 / 10000.0,
        unused_land_pct=(unused_land_m2 / max(total_area_m2, 1)) * 100,
        solar_capacity_mw=total_mw,
        solar_kwh_year=total_kwh,
        farmable_ha=farmable_ha,
        farmable_yield_tons=total_yield,
        co2_offset_tons=total_co2,
        n_recommendations=len(recs),
        created_at=now,
    )

    # Save to database
    store.save_city_kpi(kpi)
    store.save_recommendations(recs_json, scan_id)

    logger.info(
        "City audit complete: %.1f ha scanned, %.1f ha unused (%.1f%%), "
        "%.2f MW solar, %.1f ha farmable, %d recommendations",
        kpi.total_area_ha, kpi.unused_land_ha, kpi.unused_land_pct,
        kpi.solar_capacity_mw, kpi.farmable_ha, kpi.n_recommendations,
    )

    return kpi
