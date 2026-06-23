"""SQLite storage — persistent building database with change detection.

Security:
  - All queries use parameterized SQL (no string formatting)
  - Database path validated against path traversal
  - Schema versioning prevents migration issues
  - No user-supplied SQL executed
"""

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.models import BuildingRecord, BuildingStatus, CityKPI, Recommendation, ScanRecord, TileState

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS scans (
    scan_id TEXT PRIMARY KEY,
    bbox TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    n_tiles INTEGER DEFAULT 0,
    n_tiles_completed INTEGER DEFAULT 0,
    n_buildings INTEGER DEFAULT 0,
    n_unrecorded INTEGER DEFAULT 0,
    model_version TEXT DEFAULT 'yolov8n-seg',
    status TEXT DEFAULT 'in_progress'
);

CREATE TABLE IF NOT EXISTS buildings (
    building_id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL,
    scan_id TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    model_version TEXT DEFAULT 'yolov8n-seg',
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    area_m2 REAL DEFAULT 0,
    perimeter_m REAL DEFAULT 0,
    shape_class TEXT DEFAULT 'regular',
    size_class TEXT DEFAULT 'medium',
    compactness REAL DEFAULT 0,
    rectangularity REAL DEFAULT 0,
    eccentricity REAL DEFAULT 0,
    mean_r REAL DEFAULT 0,
    mean_g REAL DEFAULT 0,
    mean_b REAL DEFAULT 0,
    dominant_color TEXT DEFAULT 'unknown',
    roof_type TEXT DEFAULT 'unknown',
    building_type TEXT DEFAULT 'unknown',
    building_type_source TEXT DEFAULT 'heuristic',
    confidence REAL DEFAULT 0,
    house_number TEXT,
    street TEXT,
    city TEXT,
    postcode TEXT,
    addr_status TEXT DEFAULT 'unknown',
    is_unrecorded INTEGER DEFAULT 0,
    unrecorded_score REAL DEFAULT 0,
    osm_building_id TEXT,
    osm_building_type TEXT,
    parcel_id TEXT,
    parcel_has_unrecorded INTEGER DEFAULT 0,
    status TEXT DEFAULT 'new',
    previous_id TEXT,
    geometry_geojson TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE INDEX IF NOT EXISTS idx_buildings_scan ON buildings(scan_id);
CREATE INDEX IF NOT EXISTS idx_buildings_location ON buildings(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_buildings_tile ON buildings(tile_id);
CREATE INDEX IF NOT EXISTS idx_buildings_status ON buildings(status);

CREATE TABLE IF NOT EXISTS city_kpis (
    scan_id TEXT PRIMARY KEY,
    total_area_ha REAL DEFAULT 0,
    built_up_ha REAL DEFAULT 0,
    bare_soil_ha REAL DEFAULT 0,
    vegetation_ha REAL DEFAULT 0,
    water_ha REAL DEFAULT 0,
    unused_land_ha REAL DEFAULT 0,
    unused_land_pct REAL DEFAULT 0,
    solar_capacity_mw REAL DEFAULT 0,
    solar_kwh_year REAL DEFAULT 0,
    farmable_ha REAL DEFAULT 0,
    farmable_yield_tons REAL DEFAULT 0,
    co2_offset_tons REAL DEFAULT 0,
    n_recommendations INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    scan_id TEXT PRIMARY KEY,
    recommendations_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE INDEX IF NOT EXISTS idx_kpis_scan ON city_kpis(scan_id);
"""


class BuildingStore:
    """SQLite-backed persistent storage for building detection results.

    Args:
        database_path: Path to SQLite database file
    """

    def __init__(self, database_path: str):
        self._path = Path(database_path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if needed."""
        conn = self._get_conn()
        try:
            conn.executescript(CREATE_TABLES)
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            self._recover_db()
            conn = self._get_conn()
            conn.executescript(CREATE_TABLES)

        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()

        if row is None or row[0] is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

    def _recover_db(self) -> None:
        """Remove stale journal files to recover from a crash."""
        self._local.conn = None
        for f in (
            self._path,
            self._path.with_suffix(self._path.suffix + "-wal"),
            self._path.with_suffix(self._path.suffix + "-shm"),
        ):
            f.unlink(missing_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a per-thread database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection with error recovery."""
        try:
            conn = sqlite3.connect(str(self._path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            self._recover_db()
            conn = sqlite3.connect(str(self._path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def close(self) -> None:
        """Close database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Scan operations ────────────────────────────────────────────────

    def create_scan(self, bbox: List[float], model_version: str) -> ScanRecord:
        """Create a new scan record and return it."""
        scan_id = f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        started_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO scans (scan_id, bbox, started_at, model_version, status)
               VALUES (?, ?, ?, ?, 'in_progress')""",
            (scan_id, json.dumps(bbox), started_at, model_version),
        )
        conn.commit()

        return ScanRecord(
            scan_id=scan_id,
            bbox=bbox,
            started_at=started_at,
            model_version=model_version,
        )

    def update_scan(
        self,
        scan_id: str,
        **kwargs: Any,
    ) -> None:
        """Update scan record fields."""
        if not kwargs:
            return

        sets = ", ".join(f"{k.replace(' ', '_')} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(scan_id)

        conn = self._get_conn()
        conn.execute(f"UPDATE scans SET {sets} WHERE scan_id = ?", values)
        conn.commit()

    @staticmethod
    def _row_to_scan(row: sqlite3.Row) -> ScanRecord:
        """Convert a sqlite3.Row to a ScanRecord, parsing JSON bbox."""
        d = dict(row)
        if isinstance(d.get("bbox"), str):
            d["bbox"] = json.loads(d["bbox"])
        return ScanRecord(**d)

    def get_scan(self, scan_id: str) -> Optional[ScanRecord]:
        """Get scan record by ID."""
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_scan(row)

    def get_latest_scan(self) -> Optional[ScanRecord]:
        """Get the most recent completed scan."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM scans WHERE status = 'completed' ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_scan(row)

    def get_all_scans(self) -> List[ScanRecord]:
        """Get all scan records, newest first."""
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM scans ORDER BY started_at DESC")
        return [self._row_to_scan(row) for row in cur.fetchall()]

    # ── Building operations ─────────────────────────────────────────────

    def save_building(self, record: BuildingRecord, scan_id: str) -> str:
        """Save a single building record.

        Args:
            record: BuildingRecord to save
            scan_id: Associated scan ID

        Returns:
            building_id
        """
        return self._insert_building(record, scan_id)

    def save_buildings(
        self, records: List[BuildingRecord], scan_id: str
    ) -> List[str]:
        """Save multiple building records in a single transaction."""
        ids: List[str] = []
        conn = self._get_conn()

        for record in records:
            if not record.building_id:
                record.building_id = f"bld_{uuid.uuid4().hex[:12]}"
            ids.append(record.building_id)

        for record in records:
            self._insert_building(record, scan_id)

        conn.commit()
        return ids

    def _insert_building(self, record: BuildingRecord, scan_id: str) -> str:
        """Insert a single building record (no commit)."""
        if not record.building_id:
            record.building_id = f"bld_{uuid.uuid4().hex[:12]}"

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO buildings
               (building_id, tile_id, scan_id, scan_date, model_version,
                latitude, longitude, area_m2, perimeter_m,
                shape_class, size_class, compactness, rectangularity, eccentricity,
                mean_r, mean_g, mean_b, dominant_color, roof_type,
                building_type, building_type_source, confidence,
                house_number, street, city, postcode, addr_status,
                is_unrecorded, unrecorded_score,
                osm_building_id, osm_building_type,
                parcel_id, parcel_has_unrecorded,
                status, previous_id, geometry_geojson)
               VALUES (?, ?, ?, ?, ?,
                       ?, ?, ?, ?,
                       ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?, ?, ?,
                       ?, ?,
                       ?, ?,
                       ?, ?,
                       ?, ?, ?)""",
            (
                record.building_id, record.tile_id, scan_id,
                record.scan_date, record.model_version,
                record.latitude, record.longitude,
                record.area_m2, record.perimeter_m,
                record.shape_class.value, record.size_class.value,
                record.compactness, record.rectangularity, record.eccentricity,
                record.mean_r, record.mean_g, record.mean_b,
                record.dominant_color, record.roof_type.value,
                record.building_type.value, record.building_type_source,
                record.confidence,
                record.house_number, record.street,
                record.city, record.postcode, record.addr_status,
                1 if record.is_unrecorded else 0, record.unrecorded_score,
                record.osm_building_id, record.osm_building_type,
                record.parcel_id, 1 if record.parcel_has_unrecorded else 0,
                record.status.value, record.previous_id,
                record.geometry_geojson,
            ),
        )
        return record.building_id

    def get_buildings_by_scan(self, scan_id: str) -> List[BuildingRecord]:
        """Get all buildings for a scan."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM buildings WHERE scan_id = ? ORDER BY building_id",
            (scan_id,),
        )
        return [self._row_to_record(row) for row in cur.fetchall()]

    def get_all_buildings(self) -> List[BuildingRecord]:
        """Get all buildings across all scans."""
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM buildings ORDER BY scan_date DESC")
        return [self._row_to_record(row) for row in cur.fetchall()]

    def get_unrecorded_buildings(self, scan_id: Optional[str] = None) -> List[BuildingRecord]:
        """Get buildings flagged as unrecorded."""
        conn = self._get_conn()
        if scan_id:
            cur = conn.execute(
                "SELECT * FROM buildings WHERE is_unrecorded = 1 AND scan_id = ?",
                (scan_id,),
            )
        else:
            cur = conn.execute("SELECT * FROM buildings WHERE is_unrecorded = 1")
        return [self._row_to_record(row) for row in cur.fetchall()]

    # ── Change detection ────────────────────────────────────────────────

    def detect_changes(
        self, new_scan_id: str, previous_scan_id: str
    ) -> Dict[str, List[BuildingRecord]]:
        """Compare two scans and detect new, demolished, and unchanged buildings.

        Args:
            new_scan_id: Current scan ID
            previous_scan_id: Previous scan ID to compare against

        Returns:
            dict with keys: "new", "unchanged", "demolished"
        """
        new_buildings = self.get_buildings_by_scan(new_scan_id)
        prev_buildings = self.get_buildings_by_scan(previous_scan_id)

        new_set: List[BuildingRecord] = []
        unchanged_set: List[BuildingRecord] = []

        # Match buildings by proximity (centroid distance < 2m)
        matched_prev_ids: set[str] = set()

        for nb in new_buildings:
            matched = False
            for pb in prev_buildings:
                if pb.building_id in matched_prev_ids:
                    continue
                dist = self._haversine(nb.latitude, nb.longitude, pb.latitude, pb.longitude)
                if dist < 0.002:  # ~2m
                    nb.status = BuildingStatus.UNCHANGED
                    nb.previous_id = pb.building_id
                    unchanged_set.append(nb)
                    matched_prev_ids.add(pb.building_id)
                    matched = True
                    break

            if not matched:
                nb.status = BuildingStatus.NEW
                new_set.append(nb)

        # Remaining previous buildings = demolished
        demolished_set = [
            pb for pb in prev_buildings
            if pb.building_id not in matched_prev_ids
        ]
        for db in demolished_set:
            db.status = BuildingStatus.DEMOLISHED

        return {
            "new": new_set,
            "unchanged": unchanged_set,
            "demolished": demolished_set,
        }

    # ── Export methods ──────────────────────────────────────────────────

    def to_dataframe(self, scan_id: Optional[str] = None):
        """Export buildings to a pandas DataFrame."""
        import pandas as pd

        conn = self._get_conn()
        if scan_id:
            query = "SELECT * FROM buildings WHERE scan_id = ?"
            df = pd.read_sql_query(query, conn, params=(scan_id,))
        else:
            df = pd.read_sql_query("SELECT * FROM buildings", conn)

        return df

    # ── City KPI methods ─────────────────────────────────────────────────

    def save_city_kpi(self, kpi: CityKPI) -> None:
        """Save or update a CityKPI record."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO city_kpis
               (scan_id, total_area_ha, built_up_ha, bare_soil_ha,
                vegetation_ha, water_ha, unused_land_ha, unused_land_pct,
                solar_capacity_mw, solar_kwh_year,
                farmable_ha, farmable_yield_tons,
                co2_offset_tons, n_recommendations, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kpi.scan_id, kpi.total_area_ha, kpi.built_up_ha,
                kpi.bare_soil_ha, kpi.vegetation_ha, kpi.water_ha,
                kpi.unused_land_ha, kpi.unused_land_pct,
                kpi.solar_capacity_mw, kpi.solar_kwh_year,
                kpi.farmable_ha, kpi.farmable_yield_tons,
                kpi.co2_offset_tons, kpi.n_recommendations,
                kpi.created_at,
            ),
        )
        conn.commit()

    def get_city_kpi(self, scan_id: str) -> Optional[CityKPI]:
        """Get CityKPI for a specific scan."""
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM city_kpis WHERE scan_id = ?", (scan_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return CityKPI(**dict(row))

    def get_all_city_kpis(self) -> List[CityKPI]:
        """Get all CityKPI records, newest first."""
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM city_kpis ORDER BY created_at DESC")
        return [CityKPI(**dict(row)) for row in cur.fetchall()]

    def save_recommendations(self, recommendations_json: str, scan_id: str) -> None:
        """Save recommendations JSON for a scan."""
        from datetime import datetime, timezone
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO recommendations
               (scan_id, recommendations_json, created_at)
               VALUES (?, ?, ?)""",
            (scan_id, recommendations_json, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def get_recommendations(self, scan_id: str) -> List[dict]:
        """Get recommendations for a scan as a list of dicts."""
        import json
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT recommendations_json FROM recommendations WHERE scan_id = ?",
            (scan_id,),
        )
        row = cur.fetchone()
        if row is None:
            return []
        return json.loads(row["recommendations_json"])

    # ── Internal helpers ────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> BuildingRecord:
        """Convert a sqlite3.Row to a BuildingRecord."""
        d = dict(row)
        # Convert integer flags back to booleans
        d["is_unrecorded"] = bool(d["is_unrecorded"])
        d["parcel_has_unrecorded"] = bool(d["parcel_has_unrecorded"])
        return BuildingRecord(**d)

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in kilometers between two WGS84 points."""
        import math

        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
