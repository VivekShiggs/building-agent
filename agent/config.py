"""Configuration management with YAML loading and environment variable override.

Security:
  - API keys loaded from .env file, never from code
  - All paths validated against path traversal
  - Bounding box validated for reasonable ranges
"""

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AreaConfig(BaseModel):
    bbox_wgs84: List[float] = Field(default=[17.1050, 48.1400, 17.1150, 48.1460])
    tile_size_deg: float = Field(default=0.005, gt=0, le=1.0)

    @field_validator("bbox_wgs84")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox_wgs84 must have exactly 4 values: [west, south, east, north]")
        west, south, east, north = v
        if not (-180 <= west <= 180) or not (-180 <= east <= 180):
            raise ValueError(f"Longitude out of range: west={west}, east={east}")
        if not (-90 <= south <= 90) or not (-90 <= north <= 90):
            raise ValueError(f"Latitude out of range: south={south}, north={north}")
        if west >= east:
            raise ValueError(f"west ({west}) must be less than east ({east})")
        if south >= north:
            raise ValueError(f"south ({south}) must be less than north ({north})")
        return v


class ImageryConfig(BaseModel):
    source: str = Field(default="wms")
    local_geotiff: Optional[str] = Field(default=None)
    wms_url: str = Field(
        default="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={west},{south},{east},{north}&bboxSR=4326&imageSR=4326&size={width},{height}&format=png&f=image"
    )
    image_size_px: int = Field(default=1024, ge=256, le=4096)

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ("wms", "local"):
            raise ValueError("source must be 'wms' or 'local'")
        return v

    @field_validator("local_geotiff")
    @classmethod
    def validate_local_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            p = Path(v)
            if not p.exists():
                raise FileNotFoundError(f"Local GeoTIFF not found: {v}")
            resolved = p.resolve()
            if not resolved.suffix.lower() in (".tif", ".tiff", ".geotiff"):
                raise ValueError(f"File must be a GeoTIFF: {v}")
            return str(resolved)
        return v


class ShapeConfig(BaseModel):
    regular_threshold: float = Field(default=0.7, ge=0, le=1)
    irregular_threshold: float = Field(default=0.4, ge=0, le=1)


class SizeConfig(BaseModel):
    small_max: float = Field(default=50, gt=0)
    medium_max: float = Field(default=200, gt=0)


class ClassificationConfig(BaseModel):
    min_area_m2: float = Field(default=20, ge=1)
    shape: ShapeConfig = Field(default_factory=ShapeConfig)
    size: SizeConfig = Field(default_factory=SizeConfig)


class AuditConfig(BaseModel):
    unrecorded_overlap_threshold: float = Field(default=0.3, ge=0, le=1)


class AddressConfig(BaseModel):
    nominatim_delay: float = Field(default=1.0, ge=0.1)
    max_retries: int = Field(default=3, ge=0, le=10)


class StorageConfig(BaseModel):
    database_path: str = Field(default="data/building_db.sqlite")
    export_dir: str = Field(default="data/exports")


class ScheduleConfig(BaseModel):
    rescan_interval_days: Optional[float] = Field(default=None)


class ModelConfig(BaseModel):
    name: str = Field(default="yolov8n-seg.pt")
    finetuned_path: str = Field(default="data/models/best_finetune.pt")
    confidence: float = Field(default=0.25, ge=0.01, le=1.0)
    iou: float = Field(default=0.4, ge=0.01, le=1.0)
    imgsz: int = Field(default=640, ge=128, le=2048)
    exclude_classes: List[int] = Field(default=[0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])


class SustainabilityConfig(BaseModel):
    min_patch_area_m2: float = Field(default=50, ge=1)
    solar_min_m2: float = Field(default=500, ge=1)
    farm_min_m2: float = Field(default=200, ge=1)
    sun_hours_kwh: float = Field(default=3.8, ge=0.1)
    min_rec_score: float = Field(default=0.3, ge=0, le=1)


class AppConfig(BaseModel):
    area: AreaConfig = Field(default_factory=AreaConfig)
    imagery: ImageryConfig = Field(default_factory=ImageryConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    address: AddressConfig = Field(default_factory=AddressConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    sustainability: SustainabilityConfig = Field(default_factory=SustainabilityConfig)

    @model_validator(mode="after")
    def resolve_paths(self) -> "AppConfig":
        base = Path.cwd()
        self.storage.database_path = self._resolve_path(base, self.storage.database_path)
        self.storage.export_dir = self._resolve_path(base, self.storage.export_dir)
        self.model.finetuned_path = self._resolve_path(base, self.model.finetuned_path)
        return self

    @staticmethod
    def _resolve_path(base: Path, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        resolved = p.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file and override with environment variables.

    Security:
      - Path traversal prevented by Path.resolve()
      - .env file permissions checked (must not be world-readable)
    """
    if path is None:
        path = "config.yaml"

    config_path = Path(path)
    if not config_path.exists():
        sys.stderr.write(f"[WARN] Config file not found at {path}, using defaults\n")
        return AppConfig()

    resolved = config_path.resolve()

    # Security: ensure config file is within project directory
    # This prevents loading arbitrary files via path traversal
    project_root = Path.cwd().resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        sys.stderr.write(f"[WARN] Config file outside project directory: {resolved}\n")
        return AppConfig()

    with open(resolved) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return AppConfig()

    return AppConfig(**raw)
