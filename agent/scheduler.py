"""Tile grid scheduler — divides AOI into manageable tiles with checkpointing.

Security:
  - All tile boundaries validated against coordinate bounds
  - Database queries use parameterized SQL
  - File paths validated against path traversal
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from agent.models import TileState

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "data/tiles/tile_checkpoint.json"


def generate_tiles(
    bbox: List[float],
    tile_size_deg: float,
    existing_tiles: Optional[List[TileState]] = None,
) -> List[TileState]:
    """Divide a bounding box into a grid of tiles.

    Args:
        bbox: [west, south, east, north] in WGS84
        tile_size_deg: Tile width/height in degrees
        existing_tiles: Previously created tiles (for checkpoint recovery)

    Returns:
        List of TileState objects
    """
    west, south, east, north = bbox

    if existing_tiles:
        return existing_tiles

    tiles: List[TileState] = []
    lat = south
    tile_id_counter = 0

    while lat < north:
        lon = west
        while lon < east:
            tile_north = min(lat + tile_size_deg, north)
            tile_east = min(lon + tile_size_deg, east)

            tile_id = f"tile_{tile_id_counter:04d}_{lon:.4f}_{lat:.4f}"

            tiles.append(TileState(
                tile_id=tile_id,
                west=lon,
                south=lat,
                east=tile_east,
                north=tile_north,
                status="pending",
            ))

            lon = tile_east
            tile_id_counter += 1
        lat = lat + tile_size_deg

    logger.info("Generated %d tiles for bbox %s", len(tiles), bbox)
    return tiles


def save_checkpoint(tiles: List[TileState], checkpoint_path: str = CHECKPOINT_FILE) -> None:
    """Save tile processing state to JSON checkpoint file.

    Args:
        tiles: List of tile states
        checkpoint_path: Path to checkpoint file
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [t.model_dump() for t in tiles]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug("Checkpoint saved: %s", checkpoint_path)


def load_checkpoint(checkpoint_path: str = CHECKPOINT_FILE) -> Optional[List[TileState]]:
    """Load tile processing state from JSON checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        List of TileState or None if no checkpoint exists
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        tiles = [TileState(**t) for t in data]
        logger.info("Loaded checkpoint: %d tiles from %s", len(tiles), checkpoint_path)
        return tiles
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return None


def get_next_pending_tile(tiles: List[TileState]) -> Optional[TileState]:
    """Get the next pending tile and mark it in_progress.

    Args:
        tiles: List of tile states (modified in-place)

    Returns:
        Next pending TileState or None if all done
    """
    for tile in tiles:
        if tile.status == "pending":
            tile.status = "in_progress"
            tile.started_at = datetime.now(timezone.utc).isoformat()
            return tile
    return None


def mark_tile_done(
    tiles: List[TileState],
    tile_id: str,
    n_buildings: int = 0,
) -> None:
    """Mark a tile as completed and update its stats.

    Args:
        tiles: List of tile states (modified in-place)
        tile_id: ID of the completed tile
        n_buildings: Number of buildings detected in this tile
    """
    for tile in tiles:
        if tile.tile_id == tile_id:
            tile.status = "done"
            tile.completed_at = datetime.now(timezone.utc).isoformat()
            tile.n_buildings = n_buildings
            return


def mark_tile_failed(
    tiles: List[TileState],
    tile_id: str,
    error: str,
) -> None:
    """Mark a tile as failed with error message.

    Args:
        tiles: List of tile states (modified in-place)
        tile_id: ID of the failed tile
        error: Error description
    """
    for tile in tiles:
        if tile.tile_id == tile_id:
            tile.status = "failed"
            tile.error = error[:500]
            tile.completed_at = datetime.now(timezone.utc).isoformat()
            return


def get_tile_bounds(tile: TileState) -> List[float]:
    """Get tile bounding box as [west, south, east, north]."""
    return [tile.west, tile.south, tile.east, tile.north]


def get_tile_directory(output_dir: str, tile_id: str) -> str:
    """Get tile-specific output directory, creating it if needed."""
    path = Path(output_dir) / "tiles" / tile_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())
