"""Pipeline orchestrator — coordinates the full building detection workflow.

Self-improvement loop:
  Run → Collect → Export labels → Fine-tune → Improved model → Better results

Security:
  - All subprocess calls use absolute paths
  - No shell injection vectors (no user input passed to shell)
  - Errors are caught and logged, not silently ignored
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from typing import Callable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping, box
from shapely.ops import unary_union

from agent.addresses import AddressResolver
from agent.classifier import (
    analyze_color,
    classify_building_type,
    classify_shape,
    classify_size,
)
from agent.config import AppConfig
from agent.detector import get_model_version, run_inference
from agent.export import export_excel, export_geojson, export_google_sheets
from agent.imagery import get_imagery
from agent.models import BuildingRecord, BuildingStatus, RoofType, ShapeClass, SizeClass
from agent.scheduler import (
    generate_tiles,
    get_next_pending_tile,
    get_tile_bounds,
    get_tile_directory,
    load_checkpoint,
    mark_tile_done,
    mark_tile_failed,
    save_checkpoint,
)
from agent.store import BuildingStore
from agent.vectorize import compute_centroid, masks_to_geodataframe

logger = logging.getLogger(__name__)


class BuildingPipeline:
    """End-to-end building detection, classification, and data collection pipeline.

    Args:
        config: Application configuration
        store: BuildingStore instance
    """

    def __init__(self, config: AppConfig, store: BuildingStore):
        self.config = config
        self.store = store
        nominatim_url = "https://nominatim.openstreetmap.org"
        self._address_resolver = AddressResolver(
            nominatim_url=nominatim_url,
            rate_delay=config.address.nominatim_delay,
            max_retries=config.address.max_retries,
        )

    def run_scan(
        self,
        bbox: Optional[List[float]] = None,
        scan_name: Optional[str] = None,
        region_name: Optional[str] = None,
        tile_size_deg: Optional[float] = None,
        progress_callback: Optional[Callable[[str, str, int, int, int], None]] = None,
    ) -> str:
        """Execute a complete scan of the AOI.

        Args:
            bbox: Optional override bounding box [west, south, east, north]
            scan_name: Optional scan name for identification
            region_name: Optional city/region name for export filenames
            tile_size_deg: Optional override tile size in degrees (default: config value)
            progress_callback: Optional callback(tile_id, status, n_buildings, done, total)

        Returns:
            scan_id
        """
        if bbox is None:
            bbox = self.config.area.bbox_wgs84

        model_version = get_model_version(self.config.model)
        scan = self.store.create_scan(bbox, model_version, region_name=region_name)

        logger.info(
            "Starting scan %s | bbox=%s | model=%s",
            scan.scan_id, bbox, model_version,
        )

        tiles = self._get_tiles(bbox, tile_size_deg=tile_size_deg)
        scan = self.store.get_scan(scan.scan_id)
        if scan:
            self.store.update_scan(scan.scan_id, n_tiles=len(tiles))

        total_buildings = 0
        total_unrecorded = 0

        while True:
            tile = get_next_pending_tile(tiles)
            if tile is None:
                break

            logger.info(
                "Processing tile %s (%d/%d done)",
                tile.tile_id,
                sum(1 for t in tiles if t.status == "done"),
                len(tiles),
            )

            try:
                n_buildings, n_unrecorded = self._process_tile(tile, scan.scan_id)
                total_buildings += n_buildings
                total_unrecorded += n_unrecorded

                mark_tile_done(tiles, tile.tile_id, n_buildings)
                save_checkpoint(tiles)

                self.store.update_scan(
                    scan.scan_id,
                    n_tiles_completed=sum(1 for t in tiles if t.status == "done"),
                    n_buildings=total_buildings,
                    n_unrecorded=total_unrecorded,
                )

                if progress_callback:
                    done = sum(1 for t in tiles if t.status == "done")
                    progress_callback(tile.tile_id, "done", n_buildings, done, len(tiles))

            except Exception as e:
                logger.error("Tile %s failed: %s", tile.tile_id, e, exc_info=True)
                mark_tile_failed(tiles, tile.tile_id, str(e))
                save_checkpoint(tiles)

                if progress_callback:
                    done = sum(1 for t in tiles if t.status == "done")
                    failed = sum(1 for t in tiles if t.status == "failed")
                    progress_callback(tile.tile_id, "failed", 0, done + failed, len(tiles))

        completed_at = datetime.now(timezone.utc).isoformat()
        n_failed = sum(1 for t in tiles if t.status == "failed")
        self.store.update_scan(
            scan.scan_id,
            completed_at=completed_at,
            status="completed",
            n_tiles_completed=sum(1 for t in tiles if t.status == "done"),
            n_buildings=total_buildings,
            n_unrecorded=total_unrecorded,
        )

        # Auto-export results
        self._auto_export(scan.scan_id)

        logger.info(
            "Scan %s complete: %d buildings (%d unrecorded), %d/%d tiles, %d failed",
            scan.scan_id, total_buildings, total_unrecorded,
            sum(1 for t in tiles if t.status == "done"), len(tiles), n_failed,
        )

        return scan.scan_id

    def re_scan(
        self,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """Re-scan an area and detect changes since the last scan.

        Args:
            bbox: Optional override bounding box

        Returns:
            New scan_id
        """
        latest_scan = self.store.get_latest_scan()
        new_scan_id = self.run_scan(bbox=bbox)

        if latest_scan:
            changes = self.store.detect_changes(new_scan_id, latest_scan.scan_id)
            n_new = len(changes["new"])
            n_demolished = len(changes["demolished"])
            n_unchanged = len(changes["unchanged"])

            logger.info(
                "Change detection: +%d new, -%d demolished, %d unchanged",
                n_new, n_demolished, n_unchanged,
            )

        return new_scan_id

    def _get_tiles(self, bbox: List[float], tile_size_deg: Optional[float] = None):
        """Get or create tile grid for the given bbox.

        Args:
            bbox: [west, south, east, north]
            tile_size_deg: Override tile size (default: config value)

        Returns:
            List of TileState
        """
        existing = load_checkpoint()
        if existing and self._checkpoint_matches_bbox(existing, bbox):
            return existing
        tile_size = tile_size_deg if tile_size_deg is not None else self.config.area.tile_size_deg
        return generate_tiles(bbox, tile_size)

    @staticmethod
    def _checkpoint_matches_bbox(tiles, bbox: List[float]) -> bool:
        west, south, east, north = bbox
        tile_ws = min(t.west for t in tiles)
        tile_sn = min(t.south for t in tiles)
        tile_es = max(t.east for t in tiles)
        tile_nt = max(t.north for t in tiles)
        return (abs(tile_ws - west) < 1e-8 and abs(tile_sn - south) < 1e-8
                and abs(tile_es - east) < 1e-8 and abs(tile_nt - north) < 1e-8)

    def _process_tile(
        self, tile, scan_id: str
    ) -> Tuple[int, int]:
        """Process a single tile: imagery → detection → classification → store.

        Args:
            tile: TileState object
            scan_id: Current scan ID

        Returns:
            (n_buildings, n_unrecorded)
        """
        tile_dir = get_tile_directory(self.config.storage.export_dir, tile.tile_id)
        bbox = get_tile_bounds(tile)

        # Step 1: Get imagery
        geotiff_path, rgb = get_imagery(
            self.config.imagery, bbox, tile_dir,
        )

        # Step 2: Run detection
        pixel_masks = run_inference(rgb, self.config.model)

        if not pixel_masks:
            return (0, 0)

        # Step 3: Vectorize masks → polygons
        gdf_polygons = masks_to_geodataframe(
            pixel_masks,
            geotiff_path,
            min_area_m2=self.config.classification.min_area_m2,
        )

        if gdf_polygons.empty:
            return (0, 0)

        # Step 4: Fetch OSM reference data for this tile
        gdf_osm_buildings, gdf_parcels = self._fetch_osm_data(bbox)

        # Step 5: Spatial audit
        gdf_audited = self._audit_buildings(
            gdf_polygons, gdf_osm_buildings, gdf_parcels,
        )

        # Step 6: Classify and enrich each building
        records = self._enrich_buildings(
            gdf_audited, geotiff_path, tile.tile_id, scan_id,
        )

        # Step 7: Store results
        self.store.save_buildings(records, scan_id)

        # Step 8: City audit (land use, solar, farming, recommendations)
        try:
            from agent.city_audit import run_city_audit

            sust = self.config.sustainability
            run_city_audit(
                rgb, geotiff_path, scan_id, records,
                self.store,
                min_area_m2=sust.min_patch_area_m2,
                solar_min_m2=sust.solar_min_m2,
                farm_min_m2=sust.farm_min_m2,
                sun_hours_kwh=sust.sun_hours_kwh,
                min_rec_score=sust.min_rec_score,
            )
        except Exception as e:
            logger.warning("City audit failed for tile %s: %s", tile.tile_id, e)

        n_unrecorded = sum(1 for r in records if r.is_unrecorded)
        return (len(records), n_unrecorded)

    def _fetch_osm_data(self, bbox: List[float]):
        """Fetch OSM building footprints and parcels for a tile.

        Args:
            bbox: [west, south, east, north]

        Returns:
            (gdf_buildings, gdf_parcels)
        """
        west, south, east, north = bbox
        result = self._overpass_query(bbox, "buildings")
        gdf_buildings = self._parse_overpass_result(result)

        result_parcels = self._overpass_query(bbox, "parcels")
        gdf_parcels = self._parse_overpass_result(result_parcels)

        if gdf_parcels.empty:
            gdf_parcels = gpd.GeoDataFrame(
                [{"geometry": box(west, south, east, north), "osm_id": "0"}],
                crs="EPSG:4326",
            )

        return gdf_buildings, gdf_parcels

    def _overpass_query(self, bbox: List[float], query_type: str) -> dict:
        """Execute an Overpass API query for a specific element type.

        Retries once on timeout with a smaller bbox.

        Args:
            bbox: [west, south, east, north]
            query_type: "buildings" or "parcels"

        Returns:
            Overpass JSON response as dict
        """
        west, south, east, north = bbox

        if query_type == "buildings":
            ql = f"""
                [out:json][timeout:30];
                (way["building"]({south},{west},{north},{east}););
                out body;
                >;
                out skel qt;
            """
        elif query_type == "parcels":
            ql = f"""
                [out:json][timeout:30];
                (way["landuse"]({south},{west},{north},{east}););
                out body;
                >;
                out skel qt;
            """
        else:
            return {"elements": []}

        for attempt in range(2):
            try:
                import requests as req_lib
                resp = req_lib.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": ql.strip()},
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "BuildingAgent/1.0 (data-collection-agent)",
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = (attempt + 1) * 5.0
                    logger.warning("Overpass rate limited, retrying %s in %.0fs...", query_type, wait)
                    import time
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 0:
                    logger.warning("Overpass %s failed, retrying: %s", query_type, e)
                    continue
                logger.warning("Overpass query failed (%s): %s", query_type, e)
                return {"elements": []}

        return {"elements": []}

    def _parse_overpass_result(self, data: dict) -> gpd.GeoDataFrame:
        """Parse Overpass JSON result into a GeoDataFrame."""
        elements = data.get("elements", [])
        nodes = {
            el["id"]: (el["lon"], el["lat"])
            for el in elements if el["type"] == "node"
        }

        features = []
        for el in elements:
            if el["type"] != "way":
                continue
            coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
            if len(coords) < 3:
                continue

            from shapely.geometry import Polygon
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            features.append({
                "geometry": poly,
                "osm_id": str(el["id"]),
                **{k: str(v) for k, v in el.get("tags", {}).items()},
            })

        if not features:
            return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

        return gpd.GeoDataFrame(features, crs="EPSG:4326")

    def _audit_buildings(
        self,
        gdf_detected: gpd.GeoDataFrame,
        gdf_osm_buildings: gpd.GeoDataFrame,
        gdf_parcels: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Spatial audit: flag unrecorded buildings and associate with parcels.

        Args:
            gdf_detected: AI-detected building polygons (EPSG:4326)
            gdf_osm_buildings: OSM recorded footprints (EPSG:4326)
            gdf_parcels: OSM parcels (EPSG:4326)

        Returns:
            GeoDataFrame with added audit columns
        """
        if gdf_detected.empty:
            return gdf_detected

        bbox = self.config.area.bbox_wgs84
        lon_mid = (bbox[0] + bbox[2]) / 2
        lat_mid = (bbox[1] + bbox[3]) / 2
        zone = int((lon_mid + 180) / 6) + 1
        epsg_utm = 32600 + zone if lat_mid >= 0 else 32700 + zone

        gdf = gdf_detected.to_crs(epsg=epsg_utm).copy()
        gdf["ai_area_m2"] = gdf.geometry.area
        gdf["overlap_area_m2"] = 0.0
        gdf["osm_building_id"] = None
        gdf["osm_building_type"] = None
        gdf["parcel_id"] = None

        if not gdf_osm_buildings.empty:
            osm_utm = gdf_osm_buildings.to_crs(epsg=epsg_utm)
            recorded_union = unary_union(osm_utm.geometry)

            for idx, row in gdf.iterrows():
                intersection = row.geometry.intersection(recorded_union)
                gdf.at[idx, "overlap_area_m2"] = intersection.area

            joined = gpd.sjoin(
                gdf,
                osm_utm[["geometry", "osm_id", "building"]],
                how="left",
                predicate="intersects",
            )
            for idx, row in joined.iterrows():
                if pd.notna(row.get("osm_id_right")):
                    gdf.at[idx, "osm_building_id"] = str(row["osm_id_right"])
                    gdf.at[idx, "osm_building_type"] = str(row.get("building", "yes"))

        gdf["unrecorded_score"] = (
            1 - gdf["overlap_area_m2"] / gdf["ai_area_m2"].clip(lower=1e-9)
        ).clip(lower=0.0)
        gdf["is_unrecorded"] = (
            gdf["unrecorded_score"] > self.config.audit.unrecorded_overlap_threshold
        )

        # Parcel assignment
        if not gdf_parcels.empty:
            parcels_utm = gdf_parcels.to_crs(epsg=epsg_utm)
            parcels_utm = parcels_utm.reset_index(drop=True)
            joined_parcels = gpd.sjoin(
                gdf,
                parcels_utm[["geometry", "osm_id"]].rename(
                    columns={"osm_id": "parcel_osm_id"}
                ),
                how="left",
                predicate="intersects",
            )
            for idx, row in joined_parcels.iterrows():
                if pd.notna(row.get("parcel_osm_id")):
                    gdf.at[idx, "parcel_id"] = str(row["parcel_osm_id"])

        result = gdf.to_crs(epsg=4326)
        result["area_m2"] = gdf["ai_area_m2"].values

        return result

    def _enrich_buildings(
        self,
        gdf: gpd.GeoDataFrame,
        geotiff_path: str,
        tile_id: str,
        scan_id: str,
    ) -> List[BuildingRecord]:
        """Enrich each building with classification, color analysis, and address.

        Args:
            gdf: GeoDataFrame from spatial audit
            geotiff_path: Source GeoTIFF for color analysis
            tile_id: Current tile ID
            scan_id: Current scan ID

        Returns:
            List of BuildingRecord
        """
        import pandas as pd

        records: List[BuildingRecord] = []

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            lon, lat = compute_centroid(geom)
            area_m2 = float(row.get("area_m2", 0))
            perimeter_m = geom.length if geom.length > 0 else 0

            # Shape classification
            shape_class, compactness, rect, ecc = classify_shape(geom)

            # Size classification
            size_class = classify_size(area_m2, self.config.classification.size)

            # Color analysis
            geom_geojson_str = json.dumps(mapping(geom))
            mean_r, mean_g, mean_b, dom_color, roof_type = analyze_color(
                geom_geojson_str, geotiff_path,
            )

            # Building type
            osm_building_type = row.get("osm_building_type")
            bldg_type, bldg_type_src = classify_building_type(
                osm_building_type, shape_class, size_class, area_m2,
            )

            # Address resolution (only for high-confidence or unrecorded)
            house_number = None
            street = None
            city = None
            postcode = None
            addr_status = "unknown"
            confidence = float(row.get("confidence", 0.5))

            if confidence > 0.5 or row.get("is_unrecorded", False):
                addr = self._address_resolver.resolve(lat, lon)
                house_number = addr.get("house_number")
                street = addr.get("street")
                city = addr.get("city")
                postcode = addr.get("postcode")
                addr_status = addr.get("status", "unknown")

            record = BuildingRecord(
                building_id=f"bld_{scan_id}_{tile_id}_{hash(geom.wkt) % 10**8:08x}",
                tile_id=tile_id,
                scan_date=datetime.now(timezone.utc).isoformat(),
                model_version=get_model_version(self.config.model),
                latitude=lat,
                longitude=lon,
                area_m2=area_m2,
                perimeter_m=perimeter_m,
                shape_class=shape_class,
                size_class=size_class,
                compactness=compactness,
                rectangularity=rect,
                eccentricity=ecc,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                dominant_color=dom_color,
                roof_type=roof_type,
                building_type=bldg_type,
                building_type_source=bldg_type_src,
                confidence=confidence,
                house_number=house_number,
                street=street,
                city=city,
                postcode=postcode,
                addr_status=addr_status,
                is_unrecorded=bool(row.get("is_unrecorded", False)),
                unrecorded_score=float(row.get("unrecorded_score", 0)),
                osm_building_id=str(row.get("osm_building_id", "")) if row.get("osm_building_id") else None,
                osm_building_type=osm_building_type,
                parcel_id=str(row.get("parcel_id", "")) if row.get("parcel_id") else None,
                geometry_geojson=geom_geojson_str,
            )
            records.append(record)

        return records

    def _auto_export(self, scan_id: str) -> None:
        """Auto-export results after a scan completes."""
        export_dir = Path(self.config.storage.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        scan = self.store.get_scan(scan_id)
        region = scan.region_name if scan and scan.region_name else ""
        prefix = f"{region}_" if region else ""

        excel_path = str(export_dir / f"{prefix}{scan_id}_buildings.xlsx")
        geojson_path = str(export_dir / f"{prefix}{scan_id}_buildings.geojson")

        try:
            export_excel(self.store, excel_path, scan_id)
        except Exception as e:
            logger.warning("Excel export failed: %s", e)

        try:
            export_geojson(self.store, geojson_path, scan_id)
        except Exception as e:
            logger.warning("GeoJSON export failed: %s", e)

        try:
            export_google_sheets(self.store, scan_id=scan_id)
        except Exception as e:
            logger.debug("Google Sheets sync skipped: %s", e)
