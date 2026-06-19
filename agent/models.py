"""Pydantic data models for building detection results.

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
    started_at: str
    completed_at: Optional[str] = None
    n_tiles: int = 0
    n_tiles_completed: int = 0
    n_buildings: int = 0
    n_unrecorded: int = 0
    model_version: str = "yolov8n-seg"
    status: str = "in_progress"
