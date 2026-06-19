"""Tests for the SQLite building store."""

from agent.models import BuildingRecord, BuildingStatus


class TestBuildingStore:
    def test_create_scan(self, store):
        scan = store.create_scan([17.0, 48.0, 17.1, 48.1], "test-model")
        assert scan.scan_id is not None
        assert scan.status == "in_progress"
        assert scan.model_version == "test-model"

    def test_save_and_retrieve_building(self, store):
        scan = store.create_scan([17.0, 48.0, 17.1, 48.1], "test")

        record = BuildingRecord(
            building_id="test_001",
            tile_id="tile_0000",
            scan_date="2026-01-01T00:00:00",
            model_version="test",
            latitude=48.14,
            longitude=17.11,
            area_m2=100.0,
            perimeter_m=40.0,
        )
        store.save_building(record, scan.scan_id)

        buildings = store.get_buildings_by_scan(scan.scan_id)
        assert len(buildings) == 1
        assert buildings[0].building_id == "test_001"
        assert abs(buildings[0].latitude - 48.14) < 0.001

    def test_get_latest_scan(self, store):
        scan1 = store.create_scan([17.0, 48.0, 17.1, 48.1], "v1")
        scan2 = store.create_scan([17.0, 48.0, 17.1, 48.1], "v2")

        store.update_scan(scan1.scan_id, status="completed")
        store.update_scan(scan2.scan_id, status="completed")

        latest = store.get_latest_scan()
        assert latest is not None
        assert latest.scan_id == scan2.scan_id

    def test_detect_changes(self, store):
        scan1 = store.create_scan([17.0, 48.0, 17.1, 48.1], "v1")
        scan2 = store.create_scan([17.0, 48.0, 17.1, 48.1], "v2")

        record1 = BuildingRecord(
            building_id="bld_001",
            tile_id="tile_0000",
            scan_date="2026-01-01T00:00:00",
            model_version="v1",
            latitude=48.14,
            longitude=17.11,
            area_m2=100.0,
            perimeter_m=40.0,
        )
        store.save_building(record1, scan1.scan_id)

        record2 = BuildingRecord(
            building_id="bld_002",
            tile_id="tile_0000",
            scan_date="2026-06-01T00:00:00",
            model_version="v2",
            latitude=48.14,
            longitude=17.11,
            area_m2=100.0,
            perimeter_m=40.0,
        )
        store.save_building(record2, scan2.scan_id)

        changes = store.detect_changes(scan2.scan_id, scan1.scan_id)
        assert "new" in changes or "unchanged" in changes
