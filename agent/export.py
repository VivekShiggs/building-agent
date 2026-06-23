"""Data exporter — Excel (.xlsx) and Google Sheets output.

Security:
  - No secrets in exported files
  - Google Sheets uses service account credentials from environment
  - All file paths validated
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agent.store import BuildingStore

logger = logging.getLogger(__name__)


def _excel_column_letter(col_idx: int) -> str:
    """Convert 1-based column index to Excel column letter (A, B, ..., Z, AA, AB...)."""
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def export_excel(
    store: BuildingStore,
    output_path: str,
    scan_id: Optional[str] = None,
) -> str:
    """Export building data to a formatted Excel file.

    Args:
        store: BuildingStore instance
        output_path: Output .xlsx path
        scan_id: Optional scan ID filter

    Returns:
        Path to saved Excel file
    """
    df = store.to_dataframe(scan_id)

    if df.empty:
        logger.warning("No data to export")
        return output_path

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Buildings", index=False)

        ws = writer.sheets["Buildings"]
        for col_idx, col in enumerate(df.columns, 1):
            if not df[col].isna().all():
                max_len = int(df[col].astype(str).str.len().max())
            else:
                max_len = 0
            col_letter = _excel_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, len(str(col)) + 2), 50)

    logger.info("Excel exported: %s (%d rows)", out, len(df))
    return str(out)


def export_geojson(
    store: BuildingStore,
    output_path: str,
    scan_id: Optional[str] = None,
) -> str:
    """Export building data as GeoJSON FeatureCollection.

    Args:
        store: BuildingStore instance
        output_path: Output .geojson path
        scan_id: Optional scan ID filter
    """
    import json
    from shapely.geometry import shape, mapping

    records = store.get_buildings_by_scan(scan_id) if scan_id else store.get_all_buildings()

    features = []
    for rec in records:
        if rec.geometry_geojson:
            try:
                geom = shape(json.loads(rec.geometry_geojson))
            except Exception:
                geom = None
        else:
            geom = None

        feature = {
            "type": "Feature",
            "geometry": mapping(geom) if geom else {"type": "Point", "coordinates": [rec.longitude, rec.latitude]},
            "properties": rec.model_dump(exclude={"geometry_geojson"}),
        }
        features.append(feature)

    fc = {"type": "FeatureCollection", "features": features}

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(fc, f, indent=2, default=str)

    logger.info("GeoJSON exported: %s (%d features)", out, len(features))
    return str(out)


def export_google_sheets(
    store: BuildingStore,
    sheet_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    scan_id: Optional[str] = None,
) -> Optional[str]:
    """Sync building data to Google Sheets via gspread.

    Args:
        store: BuildingStore instance
        sheet_id: Google Sheet ID (from .env or config)
        credentials_path: Path to service account JSON (from .env)
        scan_id: Optional scan ID filter

    Returns:
        Sheet URL if successful, None otherwise
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        logger.warning("gspread not installed. Run: pip install building-agent[sheets]")
        return None

    if not credentials_path:
        credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

    if not sheet_id:
        sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_path or not sheet_id:
        logger.warning(
            "Google Sheets credentials or sheet ID not configured. "
            "Set GOOGLE_SHEETS_CREDENTIALS and GOOGLE_SHEET_ID in .env"
        )
        return None

    cred_path = Path(credentials_path)
    if not cred_path.exists():
        logger.warning("Credentials file not found: %s", credentials_path)
        return None

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(str(cred_path), scopes=scopes)
        client = gspread.authorize(creds)

        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.sheet1

        df = store.to_dataframe(scan_id)
        if df.empty:
            logger.warning("No data to sync to Google Sheets")
            return None

        df = df.fillna("")
        data = [df.columns.tolist()] + df.values.tolist()

        worksheet.clear()
        worksheet.update(range_name="A1", values=data)

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        logger.info("Google Sheets synced: %s (%d rows)", url, len(df))
        return url

    except ImportError as e:
        logger.error("Google Sheets sync failed — missing dependency: %s", e)
        return None
    except Exception as e:
        msg = str(e)
        if "not found" in msg.lower() or "notfound" in msg.lower():
            logger.error("Google Sheet not found — check GOOGLE_SHEET_ID: %s", e)
        elif "credentials" in msg.lower() or "auth" in msg.lower() or "unauthorized" in msg.lower():
            logger.error("Google Sheets auth failed — check GOOGLE_SHEETS_CREDENTIALS: %s", e)
        else:
            logger.error("Google Sheets sync failed: %s", e)
        return None
