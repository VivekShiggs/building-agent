# 🏗️ Building Agent — Complete Tutorial

Welcome! This guide walks you through the entire project from scratch —
understanding, configuring, running, and extending the AI building detection agent.

---

## 📋 Table of Contents

1. [What We're Building](#1-what-were-building)
2. [Project Architecture](#2-project-architecture)
3. [Configuration (Your First Edit)](#3-configuration)
4. [Running Tests](#4-running-tests)
5. [Step-by-Step Code Walkthrough](#5-step-by-step-code-walkthrough)
6. [Running a Full Scan](#6-running-a-full-scan)
7. [Viewing Results](#7-viewing-results)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [The Self-Improvement Loop](#9-the-self-improvement-loop)
10. [Google Sheets Export](#10-google-sheets-export)
11. [Scheduling Re-Scans](#11-scheduling-re-scans)
12. [Security Checklist](#12-security-checklist)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What We're Building

```
User Input:   A bounding box on a map (e.g., a 1km² district)

Agent Action:
  1. Downloads satellite imagery (free from ESRI)
  2. Runs YOLOv8 AI to find building rooftops
  3. Converts pixel blobs to GPS coordinates
  4. Measures each building (area, perimeter, shape)
  5. Classifies (size, shape, color, roof type, building type)
  6. Looks up the street address from OpenStreetMap
  7. Cross-checks against official records
  8. Flags any unrecorded buildings

Output:      SQLite database + Excel + GeoJSON + Google Sheets
             Interactive map showing all results
             Training data to make the AI smarter next time
```

**Real-world use case:** A city planner wants to find buildings that were
constructed without permits. They scan the area monthly and compare results.

---

## 2. Project Architecture

```
                    ┌──────────────────────┐
                    │    config.yaml        │  ← You edit this
                    │  (AOI, model, etc.)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   BuildingPipeline    │  ← agent/pipeline.py
                    │   (orchestrator)      │     The "brain"
                    └──────────┬───────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐    ┌──────▼──────┐    ┌───────▼──────┐
    │  scheduler  │    │   imagery   │    │   detector   │
    │  tile grid  │    │  WMS/local  │    │  YOLOv8 AI   │
    └──────┬──────┘    └──────┬──────┘    └───────┬──────┘
           │                  │                    │
    ┌──────▼──────┐    ┌──────▼──────┐    ┌───────▼──────┐
    │  addresses  │    │  vectorize  │    │  classifier  │
    │ OSM→GeoCode │    │ mask→poly   │    │ size/shape…  │
    └──────┬──────┘    └──────┬──────┘    └───────┬──────┘
           │                  │                    │
           └──────────────────┼────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │      store.py      │
                    │   SQLite Database  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │     export.py      │
                    │ Excel + Sheets     │
                    └────────────────────┘
```

### Key Files Explained

| File | What It Does | Analogy |
|------|-------------|---------|
| `config.yaml` | Your settings (area, model, thresholds) | The control panel |
| `agent/pipeline.py` | Orchestrates everything | The project manager |
| `agent/detector.py` | Runs YOLO AI model | The photographer |
| `agent/classifier.py` | Classifies shapes, sizes, colors | The appraiser |
| `agent/addresses.py` | Finds street addresses | The mail carrier |
| `agent/store.py` | SQLite database | The filing cabinet |
| `agent/export.py` | Excel & Google Sheets | The report printer |
| `agent/training.py` | YOLO fine-tuning export | The teacher |

---

## 3. Configuration

Open `config.yaml` — this is the only file you need to edit to change what the agent does:

```yaml
area:
  bbox_wgs84: [17.1050, 48.1400, 17.1150, 48.1460]   # Bratislava district
  tile_size_deg: 0.005                                 # ~500m tiles
```

### How to Find Your Own Area

Use Google Maps:
1. Go to https://maps.google.com
2. Right-click on the top-left corner of your area → copy coordinates
3. Right-click on the bottom-right corner → copy coordinates
4. Format as: `[west_longitude, south_latitude, east_longitude, north_latitude]`

**Example — a neighborhood in New York:**
```yaml
bbox_wgs84: [-73.9850, 40.7480, -73.9750, 40.7550]
```

**Example — a village in India:**
```yaml
bbox_wgs84: [77.1000, 28.5000, 77.1100, 28.5100]
```

### Other Settings Explained

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `model.confidence` | `0.25` | How sure the AI must be (0-1). Lower = more detections but more false positives |
| `classification.min_area_m2` | `20` | Ignore anything smaller than this (m²). A small shed is ~15m² |
| `audit.unrecorded_overlap_threshold` | `0.3` | How much of a building must be outside OSM records to flag it (0-1) |
| `imagery.source` | `"wms"` | `"wms"` for auto-download or `"local"` for your own GeoTIFF file |

---

## 4. Running Tests

Tests verify the core logic without needing an internet connection or GPU:

```bash
cd /Users/vikky/Documents/self\ detecting/building_agent
python3 -m pytest tests/ -v
```

You should see **15 passed** tests:
- `test_config.py` — validates bounding boxes, default config
- `test_store.py` — database create/read/update, change detection
- `test_classifier.py` — shape classification, size classification

**If tests fail:** run `pip3 install pytest` and try again.

---

## 5. Step-by-Step Code Walkthrough

Let's walk through the code module by module. I'll explain the most important parts.

### 5a. Configuration (config.py)

This module loads your `config.yaml` and validates every value:

```python
from agent.config import load_config

config = load_config()
print(f"AOI: {config.area.bbox_wgs84}")
print(f"Model: {config.model.name}")
```

**Why Pydantic?** Every value is type-checked and bounds-checked. If someone sets
`confidence: 999` it will be rejected. This is a security measure.

### 5b. Data Models (models.py)

This is the **schema** — it defines what a "building record" looks like:

```python
from agent.models import BuildingRecord

record = BuildingRecord(
    building_id="bld_001",
    tile_id="tile_0000",
    latitude=48.14,
    longitude=17.11,
    area_m2=100.0,
    # ... 40+ fields
)
```

Every building gets: GPS coordinates, size, shape class (regular/irregular/complex),
size class (small/medium/large), roof type, building type, confidence score,
address (house number, street, city), audit flags.

### 5c. Imagery Fetcher (imagery.py)

Downloads satellite imagery from a free WMS server:

```python
from agent.imagery import fetch_from_wms

path = fetch_from_wms(
    bbox=[17.105, 48.140, 17.115, 48.146],
    wms_url_template="https://...{west}...{east}...",
    output_path="/tmp/tile.tif",
    size_px=1024,  # 1024×1024 pixel image
)
```

**What happens:**
1. HTTP request to ESRI's ArcGIS server (free, no key)
2. Downloads a PNG image
3. Geo-references it (converts pixel coordinates to GPS coordinates)
4. Saves as a GeoTIFF file
5. Returns the file path and the RGB pixel array

### 5d. AI Detector (detector.py)

The heart of the system — runs YOLOv8 segmentation:

```python
from agent.detector import run_inference

pixel_masks = run_inference(rgb_image, config.model)
# Returns: [{"mask": bool_array, "confidence": 0.87, "class_name": "roof"}, ...]
```

**What YOLO does:**
1. Takes the 1024×1024 RGB image
2. Divides it into a grid
3. For each grid cell, predicts if there's a building
4. Draws a pixel-perfect outline (mask) around each building
5. Returns a list of masks + confidence scores

**Important:** The default YOLOv8n-seg model is trained on COCO (80 everyday objects).
It has NO "building" class. We use a heuristic: discard people/animals, keep everything
else (cars, trucks, boats, etc. on rooftops often overlap with buildings).

For production, you'd fine-tune on building data (see step 9).

### 5e. Vectorizer (vectorize.py)

Converts pixel masks to GPS polygons:

```python
from agent.vectorize import masks_to_geodataframe

gdf = masks_to_geodataframe(pixel_masks, "path/to/tile.tif", min_area_m2=20)
# Returns: GeoDataFrame with polygons in WGS84 coordinates
```

**The trick:** The GeoTIFF knows the GPS coordinate of every pixel (via its
affine transform). We use `rasterio.features.shapes()` to trace the outline
of each mask blob, then multiply by the transform to get GPS coordinates.

### 5f. Classifier (classifier.py)

Classifies each building by analyzing its geometry and pixel colors:

```python
from agent.classifier import classify_shape, classify_size, analyze_color

# Shape: uses compactness (4πA/P²)
shape_class, compactness, rect, ecc = classify_shape(polygon)
# → "regular" if compactness > 0.7 (square/circle)
# → "irregular" if compactness > 0.4
# → "complex" otherwise

# Size: uses area in m²
size_class = classify_size(area_m2, config.classification.size)
# → "small" < 50m²
# → "medium" < 200m²
# → "large" > 200m²

# Color: samples pixels under the polygon from the GeoTIFF
mean_r, mean_g, mean_b, dominant_color, roof_type = analyze_color(geojson, geotiff_path)
# Roof type heuristic:
#   red/orange → "tile"
#   grey/silver → "metal"
#   bright → "light"
#   dark → "dark"
```

### 5g. Address Resolver (addresses.py)

Resolves street addresses with a 3-tier strategy:

```python
from agent.addresses import AddressResolver

resolver = AddressResolver()
addr = resolver.resolve(lat=48.14, lon=17.11)
# Returns: {"house_number": "42", "street": "Main St", "city": "Bratislava", ...}
```

**The 3 tiers:**
1. **OSM addr:* tags** — fast, free, unlimited (handled during OSM data fetch)
2. **Nominatim reverse geocode** — rate-limited to 1 request/second
3. **Log gaps** — buildings without addresses are tracked separately

### 5h. SQLite Store (store.py)

Persistent storage with change detection:

```python
from agent.store import BuildingStore

store = BuildingStore("data/building_db.sqlite")

# Create a scan
scan = store.create_scan(bbox=[17.1, 48.1, 17.2, 48.2], model_version="yolov8n-seg")

# Save buildings
store.save_buildings(records, scan.scan_id)

# Compare two scans (change detection)
changes = store.detect_changes(new_scan_id, old_scan_id)
# → {"new": [...], "unchanged": [...], "demolished": [...]}
```

**Change detection** uses Haversine distance (< 2m = same building).

### 5i. Pipeline Orchestrator (pipeline.py)

The "brain" that ties everything together:

```python
from agent.config import load_config
from agent.store import BuildingStore
from agent.pipeline import BuildingPipeline

config = load_config()
store = BuildingStore(config.storage.database_path)
pipeline = BuildingPipeline(config, store)

# Run a full scan
scan_id = pipeline.run_scan()
```

**What happens inside `run_scan()`:**

```
1. Create scan record in database
2. Divide AOI into tiles (e.g., 4 tiles for 1km²)
3. For EACH tile:
   a. Download satellite imagery
   b. Run YOLO → get pixel masks
   c. Vectorize masks → GPS polygons
   d. Fetch OSM buildings & parcels for this tile
   e. Spatial audit (flag unrecorded)
   f. Classify each building (shape, size, color, roof type)
   g. Resolve address
   h. Save all results to database
   i. Save checkpoint (so we can resume if interrupted)
4. Auto-export Excel + GeoJSON
5. Return scan_id
```

---

## 6. Running a Full Scan

Let's do a real scan. First, I'll edit the config to use a small area
(so it's fast for demonstration):

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="bash">
<｜｜DSML｜｜parameter name="command" string="true">python3 << 'PYEOF'
import sys, json
sys.path.insert(0, ".")
from agent.config import load_config
from agent.store import BuildingStore
from agent.pipeline import BuildingPipeline
from agent.detector import get_model_version

config = load_config()
print("=" * 50)
print("CONFIGURATION")
print("=" * 50)
print(f"  AOI bbox: {config.area.bbox_wgs84}")
print(f"  Tile size: {config.area.tile_size_deg}°")
print(f"  Model: {config.model.name}")
print(f"  Confidence threshold: {config.model.confidence}")
print(f"  Min building area: {config.classification.min_area_m2} m²")
print()

# Calculate number of tiles
west, south, east, north = config.area.bbox_wgs84
n_tiles_lon = max(1, int((east - west) / config.area.tile_size_deg))
n_tiles_lat = max(1, int((north - south) / config.area.tile_size_deg))
print(f"  Tiles: {n_tiles_lon} × {n_tiles_lat} = {n_tiles_lon * n_tiles_lat}")
print()

# Check if YOLO model will be downloaded
model_version = get_model_version(config.model)
print(f"  Model version: {model_version}")
print(f"  (First run downloads ~6MB from Ultralytics)")
print()

# Show what the scan will do
print("=" * 50)
print("SCAN PLAN")
print("=" * 50)
print("  1. Download OSM data (buildings + parcels)")
print("  2. For each tile:")
print("     a. Download satellite imagery from ESRI WMS")
print("     b. Run YOLOv8n-seg AI detection")
print("     c. Classify each building (shape, size, color, roof)")
print("     d. Resolve street addresses (OSM → Nominatim)")
print("     e. Save to SQLite database")
print("  3. Export Excel + GeoJSON")
print("  4. Display interactive Folium map")
print()
print("  ⏱ Estimated time: 5-15 minutes per tile (CPU) or 1-2 min (GPU)")
print("  📦 Total data: ~5MB per tile (imagery) + ~50KB (database)")

store = BuildingStore(config.storage.database_path)
pipeline = BuildingPipeline(config, store)

print()
print("=" * 50)
print("READY TO LAUNCH")
print("=" * 50)
print()
print("  Uncomment the line below to run the actual scan:")
print("  scan_id = pipeline.run_scan()")
print()
print("  This will download real imagery and run AI detection.")
print("  For a quick test, you can use a smaller bbox in config.yaml.")
print()

# Show existing buildings if any
buildings = store.get_all_buildings()
if buildings:
    print(f"  Database already has {len(buildings)} buildings from previous scans.")
    for b in buildings[:3]:
        print(f"    {b.building_id[:16]} | {b.area_m2:.0f}m² | {b.latitude:.4f},{b.longitude:.4f}")
    if len(buildings) > 3:
        print(f"    ... and {len(buildings) - 3} more")

PYEOF