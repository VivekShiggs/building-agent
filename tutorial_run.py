#!/usr/bin/env python3
"""
Building Agent — Interactive Tutorial
======================================
Run this file step by step to learn how the system works.
Each section is independent so you can run them in any order.

Usage:
    python3 tutorial_run.py
    # Or open in VS Code and run cells individually
"""

# ═══════════════════════════════════════════════════════════════
# SECTION 0: Setup
# ═══════════════════════════════════════════════════════════════

import sys
from pathlib import Path

# Ensure we can import the agent package
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("  🏗️  BUILDING AGENT — TUTORIAL")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# SECTION 1: Configuration
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("📁 SECTION 1: Configuration")
print("─" * 60)

from agent.config import load_config

config = load_config()

print(f"\nLoaded config.yaml:")
print(f"  AOI (bounding box): {config.area.bbox_wgs84}")
print(f"  This means:")
print(f"    West longitude : {config.area.bbox_wgs84[0]}")
print(f"    South latitude : {config.area.bbox_wgs84[1]}")
print(f"    East longitude : {config.area.bbox_wgs84[2]}")
print(f"    North latitude : {config.area.bbox_wgs84[3]}")
print(f"  Tile size       : {config.area.tile_size_deg}° (~500m)")
print(f"\n  Model: {config.model.name}")
print(f"  AI confidence threshold: {config.model.confidence}")
print(f"  Minimum building area  : {config.classification.min_area_m2} m²")
print(f"\n  💡 TIP: Edit config.yaml to change your area of interest!")
print(f"     Use Google Maps to find coordinates for YOUR town.")

# ═══════════════════════════════════════════════════════════════
# SECTION 2: Loading a Building Record (Data Model)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("📦 SECTION 2: Data Model — What a Building Record Looks Like")
print("─" * 60)

from agent.models import BuildingRecord, ShapeClass, SizeClass, RoofType, BuildingType

# Create a sample building record
sample = BuildingRecord(
    building_id="demo_001",
    tile_id="tile_0000",
    scan_date="2026-06-17T12:00:00",
    latitude=48.1432,
    longitude=17.1085,
    area_m2=120.5,
    perimeter_m=45.2,
    shape_class=ShapeClass.REGULAR,
    size_class=SizeClass.MEDIUM,
    compactness=0.82,
    rectangularity=0.91,
    eccentricity=0.3,
    mean_r=180.0,
    mean_g=120.0,
    mean_b=90.0,
    dominant_color="red",
    roof_type=RoofType.TILE,
    building_type=BuildingType.HOUSE,
    building_type_source="heuristic",
    confidence=0.87,
    house_number="42",
    street="Main Street",
    city="Bratislava",
    postcode="81101",
    addr_status="geocoded",
    is_unrecorded=True,
    unrecorded_score=0.85,
    geometry_geojson='{"type":"Polygon","coordinates":[[[17.108,48.143],[17.109,48.143],[17.109,48.144],[17.108,48.144],[17.108,48.143]]]}',
)

print(f"\nOne building record has {len(BuildingRecord.model_fields)} fields.")
print(f"\nSample building:")
print(f"  ID        : {sample.building_id}")
print(f"  Location  : {sample.latitude:.4f}, {sample.longitude:.4f}")
print(f"  Size      : {sample.area_m2:.0f} m² ({sample.size_class.value})")
print(f"  Shape     : {sample.shape_class.value} (compactness={sample.compactness:.2f})")
print(f"  Color     : {sample.dominant_color} → roof: {sample.roof_type.value}")
print(f"  Type      : {sample.building_type.value}")
print(f"  Address   : {sample.house_number} {sample.street}, {sample.city}")
print(f"  Confidence: {sample.confidence:.0%}")
print(f"  Unrecorded: {'⚠️ YES' if sample.is_unrecorded else '✅ No'}")

# ═══════════════════════════════════════════════════════════════
# SECTION 3: Shape Classification
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("🔷 SECTION 3: Shape Classification")
print("─" * 60)

from shapely.geometry import Polygon
from agent.classifier import classify_shape

# A perfect square (regular)
square = Polygon([(0,0), (10,0), (10,10), (0,10), (0,0)])
sc, comp, rect, ecc = classify_shape(square)
print(f"\nSquare:")
print(f"  Compactness = {comp:.3f}  (1.0 = perfect circle)")
print(f"  Shape class = {sc.value}")

# An L-shape (irregular)
l_shape = Polygon([(0,0), (10,0), (10,3), (3,3), (3,10), (0,10), (0,0)])
sc, comp, rect, ecc = classify_shape(l_shape)
print(f"\nL-shape:")
print(f"  Compactness = {comp:.3f}")
print(f"  Shape class = {sc.value}")

# A complex shape
complex_shape = Polygon([(0,0), (5,2), (10,0), (8,5), (12,10), (7,8), (3,12), (0,10), (2,5), (0,0)])
sc, comp, rect, ecc = classify_shape(complex_shape)
print(f"\nComplex shape:")
print(f"  Compactness = {comp:.3f}")
print(f"  Shape class = {sc.value}")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: Size Classification
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("📏 SECTION 4: Size Classification")
print("─" * 60)

from agent.classifier import classify_size

class SizeCfg:
    small_max = 50
    medium_max = 200

for area in [25, 100, 500]:
    sc = classify_size(area, SizeCfg())
    print(f"  {area:4.0f} m² → {sc.value}")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: Address Resolution (Dry Run)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("📍 SECTION 5: Address Resolution")
print("─" * 60)
print()
print("  The address resolver uses a 3-tier strategy:")
print("    1. OSM addr:* tags (fast, free, unlimited)")
print("    2. Nominatim reverse geocode (1 req/s)")
print("    3. Log gaps for manual filling")
print()
print("  To test live address resolution, run:")
print("  >>> from agent.addresses import AddressResolver")
print("  >>> resolver = AddressResolver()")
print("  >>> addr = resolver.resolve(48.1432, 17.1085)")
print("  >>> print(addr)")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: Database Operations
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("🗄️  SECTION 6: SQLite Database")
print("─" * 60)

from agent.store import BuildingStore

store = BuildingStore("data/tutorial_db.sqlite")

# Create a scan
scan = store.create_scan([17.1, 48.14, 17.11, 48.146], "tutorial-model")
print(f"\nCreated scan: {scan.scan_id}")
print(f"  Status: {scan.status}")

# Save the sample building
store.save_building(sample, scan.scan_id)
print(f"  Saved sample building to database")

# Read it back
buildings = store.get_buildings_by_scan(scan.scan_id)
print(f"  Buildings in scan: {len(buildings)}")
b = buildings[0]
print(f"  First building: {b.building_id[:20]}... @ ({b.latitude:.4f}, {b.longitude:.4f})")

# Close and clean up
store.close()
Path("data/tutorial_db.sqlite").unlink(missing_ok=True)
print(f"  (cleaned up tutorial database)")

# ═══════════════════════════════════════════════════════════════
# SECTION 7: Export Demo
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("📤 SECTION 7: Export Demo")
print("─" * 60)
print()
print("  The export module supports 3 formats:")
print()
print("  1. Excel (.xlsx)")
print("     >>> from agent.export import export_excel")
print("     >>> path = export_excel(store, 'buildings.xlsx', scan_id)")
print()
print("  2. GeoJSON (.geojson)")
print("     >>> from agent.export import export_geojson")
print("     >>> path = export_geojson(store, 'buildings.geojson', scan_id)")
print()
print("  3. Google Sheets")
print("     >>> from agent.export import export_google_sheets")
print("     >>> url = export_google_sheets(store, scan_id=scan_id)")
print("     (requires service_account.json in .env)")

# ═══════════════════════════════════════════════════════════════
# SECTION 8: Running the Full Pipeline
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("🚀 SECTION 8: Running the Full Pipeline")
print("─" * 60)
print()
print("  This is the end-to-end scan. It will:")
print()
print("  1. Query OpenStreetMap for building footprints & parcels")
print("  2. Download satellite imagery from ESRI (free WMS)")
print("  3. Run YOLOv8n-seg to detect buildings")
print("  4. Convert pixel masks to GPS polygons")
print("  5. Classify each building (shape, size, color, roof)")
print("  6. Resolve street addresses")
print("  7. Audit against OSM records (find unrecorded buildings)")
print("  8. Save to SQLite database")
print("  9. Export Excel + GeoJSON")
print()
print("  To run it, execute:")
print()
print("  from agent.pipeline import BuildingPipeline")
print("  pipeline = BuildingPipeline(config, store)")
print("  scan_id = pipeline.run_scan()")
print()
print("  ⚠️  First run downloads the YOLO model (~6MB)")
print("  ⚠️  Requires internet (imagery + OSM data)")
print("  ⏱️  Expect 5-15 minutes for a small area on CPU")

# ═══════════════════════════════════════════════════════════════
# SECTION 9: Change Detection
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("🔄 SECTION 9: Change Detection (Re-scan)")
print("─" * 60)
print()
print("  After running 2+ scans, you can detect changes:")
print()
print("  >>> changes = store.detect_changes(new_scan_id, old_scan_id)")
print("  >>> print(f'New: {len(changes[\"new\"])}')")
print("  >>> print(f'Demolished: {len(changes[\"demolished\"])}')")
print()
print("  Buildings are matched by proximity (< 2m between centroids).")
print("  This is great for monitoring construction over time!")

# ═══════════════════════════════════════════════════════════════
# SECTION 10: Self-Improvement Loop
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("🧠 SECTION 10: Self-Improvement Loop")
print("─" * 60)
print()
print("  The agent can improve its own AI model over time:")
print()
print("  1. Run scan → collect building masks")
print("  2. Export as YOLO training data:")
print("     >>> from agent.training import export_yolo_labels")
print("     >>> dataset_yaml = export_yolo_labels(store, scan_id=scan_id)")
print()
print("  3. Fine-tune YOLO on your data:")
print("     yolo segment train model=yolov8n-seg.pt data=dataset.yaml epochs=100")
print()
print("  4. Place best_finetune.pt in data/models/")
print("  5. Next scan uses the improved model → better results!")
print()
print("  💡 Each scan's results become training data for the next scan.")
print("     The model gets better at detecting buildings in YOUR area,")
print("     with YOUR local architecture styles.")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ✅ TUTORIAL COMPLETE")
print("=" * 60)
print()
print("  Next steps:")
print("  1. Edit config.yaml with your AOI")
print("  2. Run: python3 -c \"from agent.pipeline import BuildingPipeline; ...\"")
print("  3. Or open the Streamlit dashboard:")
print("     streamlit run app/streamlit_app.py")
print("  4. Or open the Jupyter notebook:")
print("     jupyter notebook notebooks/building_agent.ipynb")
print()
