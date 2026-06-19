"""Tests for configuration validation."""

import pytest
from pydantic import ValidationError

from agent.config import AppConfig, AreaConfig


class TestAreaConfig:
    def test_valid_bbox(self):
        cfg = AreaConfig(bbox_wgs84=[17.0, 48.1, 17.1, 48.2])
        assert cfg.bbox_wgs84 == [17.0, 48.1, 17.1, 48.2]

    def test_invalid_bbox_length(self):
        with pytest.raises(ValidationError):
            AreaConfig(bbox_wgs84=[1, 2, 3])

    def test_invalid_longitude(self):
        with pytest.raises(ValidationError):
            AreaConfig(bbox_wgs84=[200, 0, 201, 1])

    def test_invalid_latitude(self):
        with pytest.raises(ValidationError):
            AreaConfig(bbox_wgs84=[0, -100, 1, -90])

    def test_west_less_than_east(self):
        with pytest.raises(ValidationError):
            AreaConfig(bbox_wgs84=[1, 0, 0, 1])


class TestAppConfig:
    def test_default_config(self):
        cfg = AppConfig()
        assert cfg.area.bbox_wgs84 == [17.1050, 48.1400, 17.1150, 48.1460]
        assert cfg.model.confidence == 0.25
        assert cfg.classification.min_area_m2 == 20
