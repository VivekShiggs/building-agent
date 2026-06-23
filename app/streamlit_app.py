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
    st.title("🔍 Scan a Slovak City")

    from agent.geocode import list_cities, lookup_city

    default_bbox = st.session_state.config.area.bbox_wgs84

    # ── Input mode toggle ──────────────────────────────────────────────
    input_mode = st.radio(
        "Input mode",
        ["🏙️ Select city", "✏️ Manual coordinates"],
        horizontal=True,
        label_visibility="collapsed",
    )

    bbox = None
    region_name: Optional[str] = None

    # ── City mode ──────────────────────────────────────────────────────
    if "Select city" in input_mode:
        col1, col2 = st.columns([3, 1])
        with col1:
            cities = list_cities()
            default_idx = cities.index("Trnava") if "Trnava" in cities else 0
            city = st.selectbox("City", cities, index=default_idx)
        with col2:
            presets = st.selectbox("Preset size", ["Full city", "City centre", "District"], index=0)
            preset_factors = {"Full city": 1.0, "City centre": 0.5, "District": 0.2}

        city_bbox = lookup_city(city)
        if city_bbox:
            factor = preset_factors.get(presets, 1.0)
            if factor < 1.0:
                cx = (city_bbox[0] + city_bbox[2]) / 2
                cy = (city_bbox[1] + city_bbox[3]) / 2
                hw = (city_bbox[2] - city_bbox[0]) * factor / 2
                hh = (city_bbox[3] - city_bbox[1]) * factor / 2
                bbox = [cx - hw, cy - hh, cx + hw, cy + hh]
            else:
                bbox = list(city_bbox)

            region_name = city

            st.success(
                f"✅ **{city}**  –  "
                f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"
            )

            if st.button("📌 Apply city bounds", type="secondary", use_container_width=True):
                st.session_state.drawn_bbox = bbox
                st.session_state.map_center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
                st.rerun()

    # ── Manual coordinate mode ─────────────────────────────────────────
    if "Manual coordinates" in input_mode:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                w_in = st.number_input("West", value=default_bbox[0], format="%.4f", key="scan_w")
                s_in = st.number_input("South", value=default_bbox[1], format="%.4f", key="scan_s")
            with col2:
                e_in = st.number_input("East", value=default_bbox[2], format="%.4f", key="scan_e")
                n_in = st.number_input("North", value=default_bbox[3], format="%.4f", key="scan_n")

            if st.button("📌 Set from coordinates", type="secondary", use_container_width=True):
                if w_in < e_in and s_in < n_in:
                    st.session_state.drawn_bbox = [w_in, s_in, e_in, n_in]
                    st.session_state.map_center = [(s_in + n_in) / 2, (w_in + e_in) / 2]
                    st.rerun()
                else:
                    st.error("Invalid bounds: west < east and south < north required")

            if bbox is None:
                bbox = [w_in, s_in, e_in, n_in]
                region_name = None

    # ── Resolve final bounding box ──────────────────────────────────────
    if bbox is None:
        bbox = st.session_state.drawn_bbox if st.session_state.drawn_bbox else default_bbox
        region_name = None

    # ── Scan resolution ────────────────────────────────────────────────
    st.subheader("⚙️ Scan resolution")
    res_option = st.select_slider(
        "Tile count",
        options=["Coarse (~10)", "Standard (~25)", "Detailed (~50)"],
        value="Standard (~25)",
        label_visibility="visible",
    )
    target_map = {"Coarse (~10)": 10, "Standard (~25)": 25, "Detailed (~50)": 50}
    target_tiles = target_map[res_option]

    w = max(bbox[2] - bbox[0], 0.0001)
    h = max(bbox[3] - bbox[1], 0.0001)
    ratio = w / h
    n_cols = max(1, int(round((ratio * target_tiles) ** 0.5)))
    n_rows = max(1, int(round(target_tiles / n_cols)))
    tile_w = w / n_cols
    tile_h = h / n_rows
    actual_tiles = n_cols * n_rows
    est_min = actual_tiles * 1
    est_max = actual_tiles * 4

    st.caption(
        f"Grid: {n_cols}×{n_rows} = **{actual_tiles} tiles**  |  "
        f"Tile: {tile_w:.4f}°×{tile_h:.4f}°  |  "
        f"Est: {est_min}–{est_max} min"
    )

    # ── Map with AOI + tile grid ───────────────────────────────────────
    st.subheader("🗺️ Area preview")
    center = st.session_state.map_center
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="#ff4444",
        weight=3,
        fill=True,
        fill_opacity=0.05,
        tooltip="AOI",
        popup=f"AOI: {bbox}",
    ).add_to(m)

    grid_color = "#ffaa00"
    for row in range(n_rows + 1):
        lat = bbox[1] + row * tile_h
        folium.PolyLine(
            [[lat, bbox[0]], [lat, bbox[2]]],
            color=grid_color, weight=1, opacity=0.4,
        ).add_to(m)
    for col in range(n_cols + 1):
        lon = bbox[0] + col * tile_w
        folium.PolyLine(
            [[bbox[1], lon], [bbox[3], lon]],
            color=grid_color, weight=1, opacity=0.4,
        ).add_to(m)

    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    st.components.v1.html(m._repr_html_(), height=450)

    # ── Scan button ────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        start_scan = st.button("🚀 Start City Scan", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.drawn_bbox = None
            st.rerun()

    if start_scan:
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            st.error("Invalid bbox: west < east and south < north required")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(tile_id: str, status: str, n_bld: int, done: int, total: int) -> None:
                progress_bar.progress(done / total)
                icon = "✅" if status == "done" else "❌"
                bld_info = f" — {n_bld} buildings" if n_bld else ""
                status_text.markdown(f"**{icon} {done}/{total}**  `{tile_id}`{bld_info}")

            from agent.pipeline import BuildingPipeline
            pipeline = BuildingPipeline(st.session_state.config, st.session_state.store)
            with st.spinner("Scanning city — this may take several minutes..."):
                try:
                    scan_id = pipeline.run_scan(
                        bbox=bbox,
                        region_name=region_name,
                        tile_size_deg=tile_w,
                        progress_callback=on_progress,
                    )
                    st.session_state.scan_id = scan_id
                    st.session_state.drawn_bbox = None
                    st.balloons()
                    st.success(f"✅ City scan complete! ID: `{scan_id}`")
                except Exception as e:
                    st.error(f"Scan failed: {e}")

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
