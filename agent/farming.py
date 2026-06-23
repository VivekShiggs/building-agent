"""Smart farming suitability analysis — score open land for agriculture.

Evaluates bare soil and vegetation patches for community or commercial
farming based on size, shape, and proximity to infrastructure.
"""

import logging
import math
from typing import List

from agent.models import LandPatch, RecommendationType

logger = logging.getLogger(__name__)

# Typical crop yields in tons per hectare per year (Central Europe)
YIELD_TONS_PER_HA: dict = {
    "community_farm": 8.0,  # mixed vegetables
    "commercial_farm": 6.0,  # grains / row crops
}

# Minimum patch size for different farm types
MIN_COMMUNITY_M2 = 200
MIN_COMMERCIAL_M2 = 5000


def score_patch_for_farming(
    patch: LandPatch,
    min_community_m2: float = MIN_COMMUNITY_M2,
    min_commercial_m2: float = MIN_COMMERCIAL_M2,
) -> float:
    """Score a land patch for farming suitability (0-1).

    Factors:
      - Area (larger = more viable)
      - Compactness (regular shapes = easier to till)
      - Current land use (bare soil = easiest, vegetation = some clearing needed)

    Returns:
        0-1 score
    """
    score = 0.0

    # Area score: sigmoid centered at 2000 m²
    area_score = 1.0 / (1.0 + math.exp(-(patch.area_m2 - 2000) / 800))
    score += 0.4 * area_score

    # Compactness: 0.3 weight
    score += 0.3 * patch.compactness

    # Land use: bare soil is best (no clearing), veg needs work
    if patch.land_use.value == "bare_soil":
        score += 0.3
    elif patch.land_use.value == "vegetation":
        score += 0.15

    return min(max(score, 0.0), 1.0)


def classify_farm_type(area_m2: float) -> RecommendationType:
    """Determine whether a patch is suitable for community or commercial farming."""
    if area_m2 >= MIN_COMMERCIAL_M2:
        return RecommendationType.COMMERCIAL_FARM
    return RecommendationType.COMMUNITY_FARM


def estimate_crop_yield(area_m2: float, farm_type: RecommendationType) -> float:
    """Estimate annual crop yield in tons for a given area."""
    ha = area_m2 / 10000.0
    key = "commercial_farm" if farm_type == RecommendationType.COMMERCIAL_FARM else "community_farm"
    return ha * YIELD_TONS_PER_HA.get(key, 6.0)
