"""Tests for the building classifier."""

from shapely.geometry import Polygon

from agent.classifier import classify_shape, classify_size, classify_building_type
from agent.models import ShapeClass, SizeClass, BuildingType


class TestShapeClassification:
    def test_regular_square(self):
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        shape_class, compactness, rect, ecc = classify_shape(square)
        assert shape_class == ShapeClass.REGULAR
        assert compactness > 0.7

    def test_irregular_l_shape(self):
        l_shape = Polygon([(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10), (0, 0)])
        shape_class, compactness, rect, ecc = classify_shape(l_shape)
        assert shape_class in (ShapeClass.IRREGULAR, ShapeClass.COMPLEX)


class TestSizeClassification:
    def test_small(self):
        class FakeSizeConfig:
            small_max = 50
            medium_max = 200

        assert classify_size(25.0, FakeSizeConfig()) == SizeClass.SMALL

    def test_medium(self):
        class FakeSizeConfig:
            small_max = 50
            medium_max = 200

        assert classify_size(100.0, FakeSizeConfig()) == SizeClass.MEDIUM

    def test_large(self):
        class FakeSizeConfig:
            small_max = 50
            medium_max = 200

        assert classify_size(500.0, FakeSizeConfig()) == SizeClass.LARGE
