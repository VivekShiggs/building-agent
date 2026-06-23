"""Solar potential analysis — estimate rooftop and open-land solar capacity.

Uses roof area + orientation from building records and open land patches
to estimate solar panel capacity, annual kWh generation, and CO2 offset.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from agent.models import BuildingRecord, LandPatch, RecommendationType

logger = logging.getLogger(__name__)

# Slovakia-specific constants (Trnava)
DEFAULT_SUN_HOURS_KWH = 3.8  # avg daily solar irradiation kWh/m²
PANEL_W_PER_M2 = 200  # typical panel ~200 Wp per m²
SYSTEM_EFFICIENCY = 0.80  # inverter + wiring + temp losses
CO2_PER_KWH = 0.000233  # tons CO₂ per kWh (Slovak grid mix, ~233 g/kWh)


def estimate_roof_solar(
    buildings: List[BuildingRecord],
    sun_hours: float = DEFAULT_SUN_HOURS_KWH,
) -> Tuple[float, float, float]:
    """Estimate rooftop solar potential from building footprints.

    Uses 40% of roof area as usable (south-facing or flat roofs).

    Args:
        buildings: List of BuildingRecord from a scan
        sun_hours: Avg daily solar irradiation in kWh/m²

    Returns:
        (capacity_mw, annual_kwh, co2_tons)
    """
    total_usable_m2 = sum(b.area_m2 * 0.4 for b in buildings)

    if total_usable_m2 <= 0:
        return (0.0, 0.0, 0.0)

    capacity_kw = total_usable_m2 * PANEL_W_PER_M2 / 1000.0
    annual_kwh = capacity_kw * sun_hours * 365 * SYSTEM_EFFICIENCY
    co2_tons = annual_kwh * CO2_PER_KWH

    return (capacity_kw / 1000.0, annual_kwh, co2_tons)


def estimate_open_land_solar(
    patches: List[LandPatch],
    sun_hours: float = DEFAULT_SUN_HOURS_KWH,
    usage_ratio: float = 0.5,
) -> Tuple[float, float, float]:
    """Estimate solar farm potential on bare soil / unused land patches.

    Args:
        patches: Land patches classified as bare soil or unused
        sun_hours: Avg daily solar irradiation in kWh/m²
        usage_ratio: Fraction of patch area usable for panels (pathways, spacing)

    Returns:
        (capacity_mw, annual_kwh, co2_tons)
    """
    total_usable_m2 = sum(p.area_m2 * usage_ratio for p in patches)

    if total_usable_m2 <= 0:
        return (0.0, 0.0, 0.0)

    capacity_kw = total_usable_m2 * PANEL_W_PER_M2 / 1000.0
    annual_kwh = capacity_kw * sun_hours * 365 * SYSTEM_EFFICIENCY
    co2_tons = annual_kwh * CO2_PER_KWH

    return (capacity_kw / 1000.0, annual_kwh, co2_tons)


def score_patch_for_solar(patch: LandPatch) -> float:
    """Score a land patch for solar suitability (0-1).

    Factors:
      - Compactness (regular shapes = better for panel layout)
      - Area (larger = more economical)
      - Already bare soil (no clearing needed)

    Returns:
        0-1 score
    """
    score = 0.0

    # Compactness: 0.5 weight
    score += 0.5 * patch.compactness

    # Area bonus: larger is better, sigmoid scale centered at 5000 m²
    area_factor = 1.0 / (1.0 + math.exp(-(patch.area_m2 - 5000) / 2000))
    score += 0.3 * area_factor

    # Land use bonus
    if patch.land_use.value in ("bare_soil", "unknown"):
        score += 0.2

    return min(max(score, 0.0), 1.0)
