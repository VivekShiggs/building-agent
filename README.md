# 🏗️ Building Agent

**Autonomous AI-powered building detection, classification, and data collection pipeline.**

Detects buildings from aerial imagery (free WMS), classifies by size/shape/color/roof-type, resolves addresses via OSM + Nominatim, audits against official records to flag unrecorded buildings, and stores everything in a searchable database with Excel/Google Sheets export.

Built for **zero-cost operation** using free-tier services: Google Colab (GPU), ESRI World Imagery, OSM APIs, Streamlit Cloud, and Google Sheets.

---

## Architecture

```
Imagery (WMS/local) ──► YOLOv8n-seg ──► Pixel Masks ──► Vectorize
                          │                                │
                    OSM Overpass ──────► Parcels + Bldgs    │
                          │                                ▼
                    Address Resolver              Shape/Size/Color
                    (OSM → Nominatim)             Classification
                          │                                │
                          └──────────┬─────────────────────┘
                                     ▼
                            SQLite Database
                                │    │
                          Excel/Sheets  GeoJSON
                                │
                     Self-Improvement Loop
                     (export labels → fine-tune YOLO)
```

---

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/building-agent/blob/main/notebooks/building_agent.ipynb)

1. Run the notebook cells
2. Edit `config.yaml` to set your area of interest
3. Execute the pipeline
4. View results in the interactive Folium map
5. Export to Excel or Google Sheets

---

## Local Installation

```bash
git clone https://github.com/your-org/building-agent.git
cd building-agent
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[web]"     # Streamlit dashboard
pip install -e ".[sheets]"  # Google Sheets sync
pip install -e ".[all]"     # Everything
```

---

## Usage

### CLI Scan

```bash
python -c "
from agent.config import load_config
from agent.store import BuildingStore
from agent.pipeline import BuildingPipeline

config = load_config()
store = BuildingStore(config.storage.database_path)
pipeline = BuildingPipeline(config, store)
scan_id = pipeline.run_scan()
print(f'Scan complete: {scan_id}')
"
```

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/building_agent.ipynb
```

---

## Configuration

Edit `config.yaml` to set your area of interest:

```yaml
area:
  bbox_wgs84: [17.1050, 48.1400, 17.1150, 48.1460]  # Bratislava
  tile_size_deg: 0.005                                  # ~500m tiles

model:
  confidence: 0.25
  finetuned_path: data/models/best_finetune.pt          # Self-improved model

classification:
  min_area_m2: 20
```

---

## Self-Improvement Loop

The agent can improve its own detection model over time:

1. **Scan** → detects buildings, stores masks as YOLO-format labels
2. **Export** → `agent/training.py` exports labeled training data
3. **Fine-tune** → `yolo segment train` creates a building-specific model
4. **Deploy** → place `best_finetune.pt` in `data/models/`
5. **Repeat** → each scan uses the improved model → better results

```bash
python -c "
from agent.training import export_yolo_labels
export_yolo_labels(store, scan_id='scan_...')
"
yolo segment train model=yolov8n-seg.pt data=data/training/dataset.yaml epochs=100 imgsz=640
```

---

## Data Schema

Each building record contains:

| Column | Type | Source |
|--------|------|--------|
| `building_id` | string | Generated UUID |
| `latitude`, `longitude` | float | Centroid of polygon |
| `area_m2`, `perimeter_m` | float | Footprint geometry |
| `shape_class` | enum | `regular` / `irregular` / `complex` |
| `size_class` | enum | `small` / `medium` / `large` |
| `roof_type` | enum | `tile` / `metal` / `concrete` / `green` / `light` / `dark` |
| `building_type` | enum | `house` / `apartment` / `commercial` / `industrial` / etc. |
| `confidence` | float (0-1) | YOLO detection score |
| `house_number`, `street`, `city` | string | OSM addr tags → Nominatim fallback |
| `is_unrecorded` | bool | Spatial audit result |
| `status` | enum | `new` / `existing` / `demolished` (change detection) |

---

## Security

- **No secrets in code** — API keys loaded from `.env` file only
- **Parameterized SQL** — all database queries use safe parameterization
- **Path traversal protection** — all file paths validated via `Path.resolve()`
- **HTTPS only** — HTTP URLs are upgraded to HTTPS automatically
- **Rate limiting** — Nominatim requests respect 1 req/s policy
- **Input validation** — bounding boxes validated for range and order
- **Geometry sanitization** — `buffer(0)` removes invalid self-intersections

---

## Output Files

```
data/
├── building_db.sqlite           # Master database (SQLite)
├── exports/
│   ├── buildings_<scan_id>.xlsx    # Excel export
│   ├── buildings_<scan_id>.geojson # GeoJSON for GIS tools
├── tiles/                       # Per-tile GeoTIFF cache
├── models/
│   └── best_finetune.pt         # Fine-tuned YOLO weights
└── training/
    ├── dataset.yaml             # YOLO dataset config
    ├── images/train/            # Training images
    ├── images/val/              # Validation images
    ├── labels/train/            # YOLO segmentation labels
    └── labels/val/              # YOLO segmentation labels
```

---

## Zero-Cost Stack

| Resource | Service | Limit |
|----------|---------|-------|
| GPU | Google Colab (T4) | ~12h session |
| Storage | Google Drive | 15 GB free |
| Imagery | ESRI World Imagery (WMS) | Free |
| OSM Data | Overpass API | Unlimited |
| Geocoding | Nominatim | 1 req/s |
| Dashboard | Streamlit Cloud | 1 GB RAM, free |
| Spreadsheet | Google Sheets | Free |

---

## Testing

```bash
pip install building-agent[dev]
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Upgrade Path

For production use:
- Replace YOLOv8n-seg with **SpaceNet/Maxar building model** or **SAM** for better detection
- Use **AWS S3 + Dask** for city-scale parallel processing
- Deploy with **FastAPI + Streamlit** for multi-user web access
- Integrate **high-resolution commercial imagery** (Maxar, Airbus)
- Add **change alerting** via email/webhook when new unrecorded buildings appear
