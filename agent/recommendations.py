"""Recommendation engine — merge land use, solar, and farming analyses
into ranked, actionable suggestions for city planners.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from agent.farming import classify_farm_type, estimate_crop_yield, score_patch_for_farming
from agent.models import (
    BuildingRecord,
    LandPatch,
    LandUseClass,
    Recommendation,
    RecommendationPriority,
    RecommendationType,
)
from agent.solar import estimate_open_land_solar, score_patch_for_solar

logger = logging.getLogger(__name__)

SOLAR_MIN_M2 = 500
FARM_MIN_M2 = 200


def generate_recommendations(
    buildings: List[BuildingRecord],
    land_patches: List[LandPatch],
    scan_id: str,
    solar_min_m2: float = SOLAR_MIN_M2,
    farm_min_m2: float = FARM_MIN_M2,
) -> List[Recommendation]:
    """Generate ranked recommendations from building + land use data.

    Rule-based engine producing:
      - Rooftop solar recommendations (per large building)
      - Solar farm recommendations (per bare-soil patch)
      - Community / commercial farm recommendations
      - Unused land flags

    Args:
        buildings: List of BuildingRecord from scan
        land_patches: List of LandPatch from land use analysis
        scan_id: Associated scan ID
        solar_min_m2: Minimum patch area for solar consideration
        farm_min_m2: Minimum patch area for farming consideration

    Returns:
        List of Recommendation sorted by score descending
    """
    recommendations: List[Recommendation] = []
    now = datetime.now(timezone.utc).isoformat()

    # --- Rooftop solar for large buildings ---
    for b in buildings:
        if b.area_m2 < 200:
            continue
        if b.roof_type.value in ("green", "unknown"):
            continue

        capacity_kw = b.area_m2 * 0.4 * 0.200  # 40% usable, 200 Wp/m²
        annual_kwh = capacity_kw * 3.8 * 365 * 0.80
        co2_tons = annual_kwh * 0.000233

        score = min(b.area_m2 / 2000.0, 1.0)

        recommendations.append(Recommendation(
            recommendation_id=f"rec_roof_{uuid.uuid4().hex[:8]}",
            scan_id=scan_id,
            rec_type=RecommendationType.ROOFTOP_SOLAR,
            priority=RecommendationPriority.MEDIUM,
            title=f"Rooftop solar — {b.area_m2:.0f} m²",
            description=f"Install solar panels on {b.building_type.value} "
                        f"({b.area_m2:.0f} m² roof area, ~{capacity_kw:.1f} kW capacity, "
                        f"{annual_kwh:,.0f} kWh/yr)",
            latitude=b.latitude,
            longitude=b.longitude,
            area_m2=b.area_m2,
            estimated_kwh_year=annual_kwh,
            estimated_co2_tons=co2_tons,
            geometry_geojson=b.geometry_geojson,
            score=score,
        ))

    # --- Land-based recommendations ---
    for patch in land_patches:
        if patch.land_use not in (LandUseClass.BARE_SOIL, LandUseClass.VEGETATION):
            continue

        # Solar farm
        if patch.area_m2 >= solar_min_m2:
            solar_score = score_patch_for_solar(patch)
            if solar_score > 0.3:
                _, annual_kwh, co2_tons = estimate_open_land_solar(
                    [patch], usage_ratio=0.5
                )
                capacity_mw = annual_kwh / (3.8 * 365 * 0.80 * 1000) if annual_kwh else 0

                priority = _priority_from_score(solar_score)
                recommendations.append(Recommendation(
                    recommendation_id=f"rec_solar_{uuid.uuid4().hex[:8]}",
                    scan_id=scan_id,
                    rec_type=RecommendationType.SOLAR_FARM,
                    priority=priority,
                    title=f"Solar farm — {patch.area_m2:.0f} m² ({capacity_mw:.2f} MW)",
                    description=f"Ground-mount solar on {patch.land_use.value} "
                                f"({patch.area_m2:.0f} m², ~{capacity_mw:.2f} MW capacity, "
                                f"{annual_kwh:,.0f} kWh/yr)",
                    latitude=patch.latitude,
                    longitude=patch.longitude,
                    area_m2=patch.area_m2,
                    estimated_kwh_year=annual_kwh,
                    estimated_co2_tons=co2_tons,
                    geometry_geojson=patch.geometry_geojson,
                    score=solar_score,
                ))

        # Farming
        if patch.area_m2 >= farm_min_m2:
            farm_score = score_patch_for_farming(patch)
            if farm_score > 0.3:
                farm_type = classify_farm_type(patch.area_m2)
                yield_tons = estimate_crop_yield(patch.area_m2, farm_type)

                priority = _priority_from_score(farm_score)
                label = "Community garden" if farm_type == RecommendationType.COMMUNITY_FARM else "Commercial farm"
                recommendations.append(Recommendation(
                    recommendation_id=f"rec_farm_{uuid.uuid4().hex[:8]}",
                    scan_id=scan_id,
                    rec_type=farm_type,
                    priority=priority,
                    title=f"{label} — {patch.area_m2:.0f} m² ({yield_tons:.1f} t/yr)",
                    description=f"Convert {patch.land_use.value} ({patch.area_m2:.0f} m²) "
                                f"to {label.lower()} — estimated {yield_tons:.1f} tons/yr",
                    latitude=patch.latitude,
                    longitude=patch.longitude,
                    area_m2=patch.area_m2,
                    estimated_yield_tons=yield_tons,
                    geometry_geojson=patch.geometry_geojson,
                    score=farm_score,
                ))

    # Sort by score descending
    recommendations.sort(key=lambda r: -r.score)
    return recommendations


def _priority_from_score(score: float) -> RecommendationPriority:
    if score >= 0.7:
        return RecommendationPriority.HIGH
    elif score >= 0.4:
        return RecommendationPriority.MEDIUM
    return RecommendationPriority.LOW
