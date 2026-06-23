"""Pydantic data models for building detection and city audit results.

All data models use strict validation to prevent injection and type errors.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ShapeClass(str, Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"
    COMPLEX = "complex"


class SizeClass(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RoofType(str, Enum):
    TILE = "tile"
    METAL = "metal"
    CONCRETE = "concrete"
    GREEN = "green"
    DARK = "dark"
    LIGHT = "light"
    UNKNOWN = "unknown"


class BuildingType(str, Enum):
    HOUSE = "house"
    GARAGE = "garage"
    APARTMENT = "apartment"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    SHED = "shed"
    CHURCH = "church"
    SCHOOL = "school"
    HOSPITAL = "hospital"
    OTHER = "other"
    UNKNOWN = "unknown"


class BuildingStatus(str, Enum):
    NEW = "new"
    EXISTING = "existing"
    DEMOLISHED = "demolished"
    UNCHANGED = "unchanged"


class BuildingRecord(BaseModel):
    """Core building record — single detected building with all derived data."""

    building_id: str = Field(description="Unique building identifier")
    tile_id: str = Field(description="Tile grid cell identifier")
    scan_date: str = Field(description="ISO 8601 scan timestamp")
    model_version: str = Field(default="yolov8n-seg", description="Model version used")

    # Geometry
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    area_m2: float = Field(ge=0, description="Footprint area in square meters")
    perimeter_m: float = Field(ge=0, description="Footprint perimeter in meters")

    # Classification
    shape_class: ShapeClass = Field(default=ShapeClass.REGULAR)
    size_class: SizeClass = Field(default=SizeClass.MEDIUM)
    compactness: float = Field(default=0.0, ge=0, le=1)
    rectangularity: float = Field(default=0.0, ge=0, le=1)
    eccentricity: float = Field(default=0.0, ge=0)

    # Color analysis
    mean_r: float = Field(default=0, ge=0, le=255)
    mean_g: float = Field(default=0, ge=0, le=255)
    mean_b: float = Field(default=0, ge=0, le=255)
    dominant_color: str = Field(default="unknown")
    roof_type: RoofType = Field(default=RoofType.UNKNOWN)

    # Building type
    building_type: BuildingType = Field(default=BuildingType.UNKNOWN)
    building_type_source: str = Field(default="heuristic")

    # Detection confidence
    confidence: float = Field(default=0.0, ge=0, le=1)

    # Address
    house_number: Optional[str] = Field(default=None)
    street: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    postcode: Optional[str] = Field(default=None)
    addr_status: str = Field(default="unknown")

    # Audit
    is_unrecorded: bool = Field(default=False)
    unrecorded_score: float = Field(default=0.0, ge=0, le=1)
    osm_building_id: Optional[str] = Field(default=None)
    osm_building_type: Optional[str] = Field(default=None)
    parcel_id: Optional[str] = Field(default=None)
    parcel_has_unrecorded: bool = Field(default=False)

    # Change detection
    status: BuildingStatus = Field(default=BuildingStatus.NEW)
    previous_id: Optional[str] = Field(default=None, description="building_id from previous scan")

    # Raw geometry (GeoJSON string for storage)
    geometry_geojson: Optional[str] = Field(default=None)

    @field_validator("scan_date")
    @classmethod
    def validate_scan_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            return datetime.now(timezone.utc).isoformat()
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def columns() -> List[str]:
        return list(BuildingRecord.model_fields.keys())


class TileState(BaseModel):
    """Track processing state of a single tile."""

    tile_id: str
    west: float
    south: float
    east: float
    north: float
    status: str = Field(default="pending")  # pending, in_progress, done, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    n_buildings: int = 0


class ScanRecord(BaseModel):
    """Metadata for a complete scan run."""

    scan_id: str
    bbox: List[float]
    region_name: Optional[str] = Field(default=None, description="City or region name for export filenames")
    started_at: str
    completed_at: Optional[str] = None
    n_tiles: int = 0
    n_tiles_completed: int = 0
    n_buildings: int = 0
    n_unrecorded: int = 0
    model_version: str = "yolov8n-seg"
    status: str = "in_progress"


# ── Sustainable City AI — Phase 1 models ─────────────────────────


class LandUseClass(str, Enum):
    """Land use classification for a detected patch."""

    BUILT_UP = "built_up"
    BARE_SOIL = "bare_soil"
    VEGETATION = "vegetation"
    WATER = "water"
    UNKNOWN = "unknown"


class RecommendationPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    SOLAR_FARM = "solar_farm"
    ROOFTOP_SOLAR = "rooftop_solar"
    COMMUNITY_FARM = "community_farm"
    COMMERCIAL_FARM = "commercial_farm"
    PARK = "park"
    UNUSED_LAND = "unused_land"


class LandPatch(BaseModel):
    """A contiguous patch of land with use classification and suitability scores."""

    patch_id: str = Field(description="Unique patch identifier")
    scan_id: str = Field(description="Associated scan ID")
    land_use: LandUseClass = Field(default=LandUseClass.UNKNOWN)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    area_m2: float = Field(ge=0, description="Area in square meters")
    compactness: float = Field(default=0.0, ge=0, le=1)
    solar_score: float = Field(default=0.0, ge=0, le=1, description="0-1 suitability for solar")
    farm_score: float = Field(default=0.0, ge=0, le=1, description="0-1 suitability for farming")
    road_distance_m: Optional[float] = Field(default=None)
    geometry_geojson: Optional[str] = Field(default=None)


class Recommendation(BaseModel):
    """An actionable recommendation for city planners."""

    recommendation_id: str = Field(description="Unique ID")
    scan_id: str = Field(description="Associated scan ID")
    rec_type: RecommendationType = Field(...)
    priority: RecommendationPriority = Field(default=RecommendationPriority.MEDIUM)
    title: str = Field(description="Short human-readable title")
    description: str = Field(description="Detailed explanation")
    latitude: float = Field(...)
    longitude: float = Field(...)
    area_m2: float = Field(ge=0, description="Recommended area in m²")
    estimated_kwh_year: Optional[float] = Field(default=None, description="Solar: annual kWh")
    estimated_co2_tons: Optional[float] = Field(default=None, description="CO₂ offset in tons/yr")
    estimated_yield_tons: Optional[float] = Field(default=None, description="Farming: crop tons/yr")
    geometry_geojson: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0, le=1, description="Overall recommendation score")


class CityKPI(BaseModel):
    """Aggregated city-level key performance indicators from a scan."""

    scan_id: str = Field(description="Associated scan ID")
    total_area_ha: float = Field(default=0, ge=0, description="Total scanned area in hectares")
    built_up_ha: float = Field(default=0, ge=0, description="Built-up / building area in hectares")
    bare_soil_ha: float = Field(default=0, ge=0, description="Bare soil area in hectares")
    vegetation_ha: float = Field(default=0, ge=0, description="Vegetation area in hectares")
    water_ha: float = Field(default=0, ge=0, description="Water area in hectares")
    unused_land_ha: float = Field(default=0, ge=0, description="Vacant / unused land area")
    unused_land_pct: float = Field(default=0, ge=0, le=100, description="Unused land as % of total")
    solar_capacity_mw: float = Field(default=0, ge=0, description="Estimated rooftop solar potential in MW")
    solar_kwh_year: float = Field(default=0, ge=0, description="Estimated annual solar generation in kWh")
    farmable_ha: float = Field(default=0, ge=0, description="Land suitable for farming in hectares")
    farmable_yield_tons: float = Field(default=0, ge=0, description="Estimated annual crop yield in tons")
    co2_offset_tons: float = Field(default=0, ge=0, description="CO₂ offset from solar in tons/yr")
    n_recommendations: int = Field(default=0, description="Number of recommendations generated")
    created_at: str = Field(default="", description="ISO 8601 timestamp")
