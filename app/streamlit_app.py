"""Streamlit web application — Building Agent dashboard.

Tabs:
  - Scan: configure and run building detection
  - Results: browse, filter, and export detected buildings
  - Review: human review of low-confidence detections
  - Train: export training data and fine-tune model
  - Dashboard: statistics and charts

Security:
  - No secrets in the UI
  - Session state isolated per user
  - All file paths read-only from web
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import folium
import pandas as pd
import streamlit as st
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config
from agent.export import export_excel, export_geojson, export_google_sheets
from agent.store import BuildingStore

try:
    from streamlit_folium import st_folium
    HAS_FOLIUM_COMPONENT = True
except ImportError:
    HAS_FOLIUM_COMPONENT = False

st.set_page_config(
    page_title="Building Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ──────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = load_config()
if "store" not in st.session_state:
    try:
        db_path = st.session_state.config.storage.database_path
        st.session_state.store = BuildingStore(db_path)
    except Exception:
        import tempfile
        db_path = str(Path(tempfile.gettempdir()) / "building_agent_fallback.sqlite")
        st.session_state.store = BuildingStore(db_path)
        st.warning(f"Could not open primary database — using temporary: {db_path}")
if "scan_id" not in st.session_state:
    st.session_state.scan_id = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "drawn_bbox" not in st.session_state:
    st.session_state.drawn_bbox = None
if "map_center" not in st.session_state:
    cfg = st.session_state.config
    b = cfg.area.bbox_wgs84
    st.session_state.map_center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]


def run_scan(bbox: List[float]) -> str:
    """Run building detection scan and return scan_id."""
    from agent.pipeline import BuildingPipeline

    config = load_config()
    store = st.session_state.store
    pipeline = BuildingPipeline(config, store)
    scan_id = pipeline.run_scan(bbox=bbox)
    return scan_id


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🏙️ City Audit AI")
st.sidebar.markdown("Sustainable city analytics — detect, classify, recommend")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "City Audit", "Scan", "Results", "Review", "Train", "Export"],
    index=0,
)
st.session_state.page = page

st.sidebar.markdown("---")
st.sidebar.markdown("**Configuration**")
st.sidebar.code(
    f"AOI: {st.session_state.config.area.bbox_wgs84}\n"
    f"Model: {st.session_state.config.model.name}\n"
    f"Min area: {st.session_state.config.classification.min_area_m2} m²",
)

# ── Dashboard Tab ─────────────────────────────────────────────────────
if page == "Dashboard":
    st.title("📊 Building Agent Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    try:
        all_buildings = st.session_state.store.get_all_buildings()
        all_scans = st.session_state.store.get_all_scans()
    except Exception:
        all_buildings = []
        all_scans = []

    with col1:
        st.metric("Total Buildings", len(all_buildings))
    with col2:
        st.metric("Total Scans", len(all_scans))
    with col3:
        n_unrecorded = sum(1 for b in all_buildings if b.is_unrecorded)
        st.metric("Unrecorded", n_unrecorded, delta_color="inverse")
    with col4:
        if all_scans:
            latest = all_scans[0]
            st.metric("Latest Scan", latest.scan_id[:20] + "...")
        else:
            st.metric("Latest Scan", "None")

    if all_scans:
        st.subheader("Recent Scans")
        scan_data = []
        for s in all_scans[:10]:
            scan_data.append({
                "Scan ID": s.scan_id[:30],
                "Date": s.started_at[:19] if s.started_at else "",
                "Buildings": s.n_buildings,
                "Unrecorded": s.n_unrecorded,
                "Tiles": f"{s.n_tiles_completed}/{s.n_tiles}",
                "Status": s.status,
            })
        st.dataframe(pd.DataFrame(scan_data), width="stretch")

    st.info(
        "Run a scan from the **Scan** tab to start collecting building data. "
        "Use the self-improvement loop in the **Train** tab to fine-tune the "
        "detection model with your collected data."
    )

# ── City Audit Tab ────────────────────────────────────────────────────
elif page == "City Audit":
    st.title("🏙️ Sustainable City Audit")
    st.markdown("""
    Government KPIs and actionable recommendations from your scan data.
    Run a scan first from the **Scan** tab.
    """)

    kpis = st.session_state.store.get_all_city_kpis()
    if not kpis:
        st.info("No audit data yet. Run a scan from the **Scan** tab to generate city KPIs.")
        st.stop()

    selected = st.selectbox(
        "Select scan",
        [f"{k.scan_id[:25]}... ({k.created_at[:19]})" for k in kpis],
    )
    idx = [f"{k.scan_id[:25]}... ({k.created_at[:19]})" for k in kpis].index(selected)
    kpi = kpis[idx]

    # KPI cards
    st.subheader("📊 City KPIs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Area", f"{kpi.total_area_ha:.1f} ha")
    with c2:
        st.metric("Built-up", f"{kpi.built_up_ha:.1f} ha", f"{kpi.built_up_ha / max(kpi.total_area_ha, 0.01) * 100:.0f}%")
    with c3:
        st.metric("Unused Land", f"{kpi.unused_land_ha:.1f} ha", f"{kpi.unused_land_pct:.1f}%")
    with c4:
        st.metric("Vegetation", f"{kpi.vegetation_ha:.1f} ha")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("☀️ Solar Capacity", f"{kpi.solar_capacity_mw:.2f} MW")
    with c2:
        st.metric("🌱 Farmable Land", f"{kpi.farmable_ha:.1f} ha")
    with c3:
        st.metric("🌿 CO₂ Offset", f"{kpi.co2_offset_tons:.1f} t/yr")

    # Land use breakdown
    st.subheader("🧱 Land Use Breakdown")
    land_data = pd.DataFrame([
        {"Class": "Built-up", "ha": kpi.built_up_ha},
        {"Class": "Vegetation", "ha": kpi.vegetation_ha},
        {"Class": "Bare Soil", "ha": kpi.bare_soil_ha},
        {"Class": "Water", "ha": kpi.water_ha},
    ])
    st.bar_chart(land_data.set_index("Class"))

    # Recommendations
    st.subheader("💡 Recommendations")
    recs = st.session_state.store.get_recommendations(kpi.scan_id)

    if not recs:
        st.info("No recommendations generated for this scan.")
    else:
        for r in recs:
            prio = r.get("priority", "medium")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(prio, "⚪")
            with st.expander(f"{icon} **{r['title']}** (score: {r['score']:.2f})"):
                st.write(r["description"])
                cols = st.columns(3)
                if r.get("estimated_kwh_year"):
                    cols[0].metric("Annual kWh", f"{r['estimated_kwh_year']:,.0f}")
                if r.get("estimated_co2_tons"):
                    cols[1].metric("CO₂ Offset", f"{r['estimated_co2_tons']:.1f} t/yr")
                if r.get("estimated_yield_tons"):
                    cols[2].metric("Crop Yield", f"{r['estimated_yield_tons']:.1f} t/yr")

    # Solar deep-dive
    st.subheader("☀️ Solar Potential Detail")
    st.markdown(f"""
    - **Rooftop + ground-mount solar capacity**: {kpi.solar_capacity_mw:.2f} MW
    - **Estimated annual generation**: {kpi.solar_kwh_year:,.0f} kWh
    - **CO₂ offset**: {kpi.co2_offset_tons:.1f} tons/year
    """)
    st.caption(
        "Based on {:.0f} buildings (40% usable roof area) and {:.1f} ha bare soil (50% panel coverage). "
        "Trnava avg: 3.8 kWh/m²/day.".format(
            len(st.session_state.store.get_buildings_by_scan(kpi.scan_id)) if kpi.scan_id else 0,
            kpi.bare_soil_ha,
        )
    )

# ── Scan Tab ──────────────────────────────────────────────────────────
elif page == "Scan":
    st.title("🔍 Choose Your Location")

    st.markdown("""
    **Draw a rectangle** on the map below to select your area of interest,  
    or manually enter coordinates.
    """)

    default_bbox = st.session_state.config.area.bbox_wgs84
    west, south, east, north = default_bbox

    # ── Coordinate inputs (collapsible) ────────────────────────────────
    with st.expander("✏️ Manual coordinate input", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            west_in = st.number_input("West longitude", value=west, format="%.4f", key="w")
            south_in = st.number_input("South latitude", value=south, format="%.4f", key="s")
        with col2:
            east_in = st.number_input("East longitude", value=east, format="%.4f", key="e")
            north_in = st.number_input("North latitude", value=north, format="%.4f", key="n")

        if st.button("📌 Set from coordinates", type="secondary"):
            if west_in < east_in and south_in < north_in:
                st.session_state.drawn_bbox = [west_in, south_in, east_in, north_in]
                center_lat = (south_in + north_in) / 2
                center_lon = (west_in + east_in) / 2
                st.session_state.map_center = [center_lat, center_lon]
                st.rerun()
            else:
                st.error("Invalid bounds: west < east and south < north required")

    # ── Interactive Map ────────────────────────────────────────────────
    st.subheader("🗺️ Draw your AOI on the map")

    if st.session_state.drawn_bbox:
        db = st.session_state.drawn_bbox
        current_bbox = db
    else:
        current_bbox = default_bbox

    center = st.session_state.map_center
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    # Draw the current AOI rectangle
    folium.Rectangle(
        bounds=[[current_bbox[1], current_bbox[0]], [current_bbox[3], current_bbox[2]]],
        color="#ff4444",
        weight=3,
        fill=True,
        fill_opacity=0.15,
        tooltip="AOI",
        popup=f"AOI: {current_bbox}",
    ).add_to(m)

    # Add a tile layer selector
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    # Render the map and capture interactions
    if HAS_FOLIUM_COMPONENT:
        map_data = st_folium(
            m,
            width=None,
            height=500,
            key="scan_map",
            returned_objects=["last_active_drawing", "last_bounds", "center", "zoom"],
        )

        # Detect rectangle drawn on map
        if map_data and map_data.get("last_active_drawing"):
            drawing = map_data["last_active_drawing"]
            if drawing["geometry"]["type"] == "Polygon":
                coords = drawing["geometry"]["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                drawn_bbox = [min(lons), min(lats), max(lons), max(lats)]
                st.session_state.drawn_bbox = drawn_bbox
                st.session_state.map_center = [
                    (drawn_bbox[1] + drawn_bbox[3]) / 2,
                    (drawn_bbox[0] + drawn_bbox[2]) / 2,
                ]
                st.rerun()

        # Handle map click for center
        if map_data and map_data.get("last_bounds"):
            st.session_state.map_center = [
                (map_data["last_bounds"][0][0] + map_data["last_bounds"][1][0]) / 2,
                (map_data["last_bounds"][0][1] + map_data["last_bounds"][1][1]) / 2,
            ]
    else:
        st.warning("Install streamlit-folium for interactive map drawing: `pip install streamlit-folium`")
        st.components.v1.html(m._repr_html_(), height=400)

    # ── Show selected coordinates ──────────────────────────────────────
    if st.session_state.drawn_bbox:
        db = st.session_state.drawn_bbox
        st.success(f"✅ **Selected area:** West `{db[0]:.4f}`  South `{db[1]:.4f}`  East `{db[2]:.4f}`  North `{db[3]:.4f}`")
        area_deg = (db[2] - db[0]) * (db[3] - db[1])
        area_approx_km2 = area_deg * 111 * 111  # rough: 1° ≈ 111km
        st.caption(f"Area: ~{area_approx_km2:.2f} km²  |  Tiles: ~{max(1, int((db[2]-db[0])/0.005)) * max(1, int((db[3]-db[1])/0.005))}")
    else:
        db = default_bbox
        st.info(f"Current AOI: West {db[0]}  South {db[1]}  East {db[2]}  North {db[3]}")

    # ── Scan button ────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Start Scan", type="primary", width="stretch"):
            bbox_to_scan = st.session_state.drawn_bbox if st.session_state.drawn_bbox else default_bbox

            if bbox_to_scan[0] >= bbox_to_scan[2] or bbox_to_scan[1] >= bbox_to_scan[3]:
                st.error("Invalid bbox: west < east and south < north required")
            else:
                with st.spinner(f"🔍 Scanning {bbox_to_scan}...\n\nThis downloads imagery, runs AI detection, and audits against OSM records. May take 5-15 minutes."):
                    try:
                        scan_id = run_scan(bbox_to_scan)
                        st.session_state.scan_id = scan_id
                        st.session_state.drawn_bbox = None
                        st.balloons()
                        st.success(f"✅ Scan complete! ID: `{scan_id}`")
                        st.page_link("app/streamlit_app.py?page=Results", label="📋 View Results →")
                    except Exception as e:
                        st.error(f"Scan failed: {e}")
    with col2:
        if st.button("🗑️ Clear selection"):
            st.session_state.drawn_bbox = None
            st.rerun()

# ── Results Tab ───────────────────────────────────────────────────────
elif page == "Results":
    st.title("📋 Building Results")

    scans = st.session_state.store.get_all_scans()
    scan_options = {s.scan_id[:30]: s.scan_id for s in scans}
    if not scan_options:
        st.info("No scans found. Run a scan first.")
        st.stop()

    selected_label = st.selectbox("Select scan", list(scan_options.keys()))
    selected_scan_id = scan_options[selected_label]

    buildings = st.session_state.store.get_buildings_by_scan(selected_scan_id)
    df = pd.DataFrame([b.to_dict() for b in buildings])

    if df.empty:
        st.info("No buildings detected in this scan.")
        st.stop()

    st.metric("Buildings", len(df))

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        if "is_unrecorded" in df.columns:
            show_unrecorded = st.checkbox("Show only unrecorded")
            if show_unrecorded:
                df = df[df["is_unrecorded"] == True]
    with col2:
        if "roof_type" in df.columns:
            roof_types = ["All"] + list(df["roof_type"].unique())
            selected_roof = st.selectbox("Roof type", roof_types)
            if selected_roof != "All":
                df = df[df["roof_type"] == selected_roof]
    with col3:
        if "building_type" in df.columns:
            bldg_types = ["All"] + list(df["building_type"].unique())
            selected_type = st.selectbox("Building type", bldg_types)
            if selected_type != "All":
                df = df[df["building_type"] == selected_type]

    display_cols = [
        "building_id", "latitude", "longitude", "area_m2",
        "shape_class", "size_class", "roof_type", "building_type",
        "confidence", "is_unrecorded", "house_number", "street",
        "addr_status",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(df[display_cols], width="stretch")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        csv,
        f"buildings_{selected_scan_id[:12]}.csv",
        "text/csv",
    )

    # Map view
    st.subheader("Map View")
    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        map_center = [df["latitude"].mean(), df["longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=17, tiles="OpenStreetMap")

        for _, row in df.iterrows():
            color = "red" if row.get("is_unrecorded") else "blue"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row.get('building_id', '')}: {row.get('area_m2', 0):.0f}m²",
            ).add_to(m)

        st.components.v1.html(m._repr_html_(), height=500)

# ── Review Tab ────────────────────────────────────────────────────────
elif page == "Review":
    st.title("✅ Human Review")

    scans = st.session_state.store.get_all_scans()
    if not scans:
        st.info("No scans available.")
        st.stop()

    selected_scan_id = scans[0].scan_id
    buildings = st.session_state.store.get_buildings_by_scan(selected_scan_id)
    low_conf = [b for b in buildings if b.confidence < 0.5]

    if not low_conf:
        st.success("No low-confidence detections to review!")
        st.stop()

    st.warning(f"{len(low_conf)} low-confidence buildings need review")

    for b in low_conf[:20]:
        with st.expander(
            f"{b.building_id[:16]} — {b.area_m2:.0f}m² — conf={b.confidence:.2f}"
        ):
            st.json(b.to_dict())
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Accept {b.building_id[:8]}", key=f"accept_{b.building_id}"):
                    b.confidence = max(b.confidence, 0.8)
                    st.session_state.store.save_building(b, selected_scan_id)
                    st.success("Accepted!")
            with col2:
                if st.button(f"❌ Reject {b.building_id[:8]}", key=f"reject_{b.building_id}"):
                    st.info("Rejected (remove in production)")

# ── Train Tab ─────────────────────────────────────────────────────────
elif page == "Train":
    st.title("🧠 Self-Improvement Training")

    st.markdown("""
    The self-improvement loop lets you fine-tune the YOLO detection model
    using buildings collected during scans. This creates a custom model
    that better detects buildings in your specific area.
    """)

    scans = st.session_state.store.get_all_scans()
    scan_options = {s.scan_id[:30]: s.scan_id for s in scans}
    if not scan_options:
        st.info("No scans available. Run a scan first.")
        st.stop()

    selected_label = st.selectbox("Select scan for training data", list(scan_options.keys()))
    selected_scan_id = scan_options[selected_label]

    buildings = st.session_state.store.get_buildings_by_scan(selected_scan_id)
    st.metric("Available buildings with geometry", len([b for b in buildings if b.geometry_geojson]))

    if st.button("🚀 Export Training Data", type="primary"):
        from agent.training import export_yolo_labels

        with st.spinner("Exporting training data..."):
            dataset_yaml = export_yolo_labels(
                st.session_state.store,
                scan_id=selected_scan_id,
            )
            if dataset_yaml:
                st.success(f"Training data exported to `{dataset_yaml}`")
                st.code(
                    f"yolo segment train "
                    f"model={st.session_state.config.model.name} "
                    f"data={dataset_yaml} epochs=100 imgsz=640"
                )
            else:
                st.error("No training data exported")

# ── Export Tab ────────────────────────────────────────────────────────
elif page == "Export":
    st.title("📤 Export Data")

    scans = st.session_state.store.get_all_scans()
    scan_options = {s.scan_id[:30]: s.scan_id for s in scans}
    if not scan_options:
        st.info("No scans found.")
        st.stop()

    selected_label = st.selectbox("Select scan to export", list(scan_options.keys()))
    selected_scan_id = scan_options[selected_label]

    export_dir = st.session_state.config.storage.export_dir
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Excel", width="stretch"):
            path = export_excel(st.session_state.store, f"{export_dir}/buildings.xlsx", selected_scan_id)
            st.success(f"Excel saved: {path}")
            with open(path, "rb") as f:
                st.download_button("Download Excel", f, "buildings.xlsx")

    with col2:
        if st.button("📥 Export GeoJSON", width="stretch"):
            path = export_geojson(st.session_state.store, f"{export_dir}/buildings.geojson", selected_scan_id)
            st.success(f"GeoJSON saved: {path}")
            with open(path, "rb") as f:
                st.download_button("Download GeoJSON", f, "buildings.geojson")

    st.subheader("Google Sheets Sync")
    st.info(
        "Configure GOOGLE_SHEETS_CREDENTIALS and GOOGLE_SHEET_ID in your .env file "
        "to enable Google Sheets sync."
    )
    if st.button("🔄 Sync to Google Sheets"):
        url = export_google_sheets(
            st.session_state.store,
            scan_id=selected_scan_id,
        )
        if url:
            st.success(f"Synced: {url}")
        else:
            st.error("Sync failed. Check credentials configuration.")
