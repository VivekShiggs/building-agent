"""Tile grid generator for Trnava city — programmatic bbox grid → tiles.json.

Usage:
    python tile_grid.py                     # default Trnava 3×3 grid
    python tile_grid.py --west 17.56 --south 48.37 --tile-w 0.0262 --tile-h 0.0154 --cols 3 --rows 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def generate_grid(
    west_start: float,
    south_start: float,
    tile_w: float,
    tile_h: float,
    n_cols: int,
    n_rows: int,
) -> List[Dict[str, Any]]:
    """Generate a grid of tile bounding boxes.

    Tiles are numbered left-to-right, top-to-bottom (row-major).

    Args:
        west_start: West longitude of the top-left tile
        south_start: South latitude of the top-left tile
        tile_w: Tile width in degrees longitude
        tile_h: Tile height in degrees latitude
        n_cols: Number of columns (east-west)
        n_rows: Number of rows (north-south)

    Returns:
        List of dicts: {tile_id, west, south, east, north}
    """
    tiles: List[Dict[str, Any]] = []

    for row in range(n_rows):
        for col in range(n_cols):
            west = west_start + col * tile_w
            east = west + tile_w
            south = south_start + row * tile_h
            north = south + tile_h

            tile_id = f"tile_r{row:02d}c{col:02d}"

            tiles.append({
                "tile_id": tile_id,
                "west": round(west, 6),
                "south": round(south, 6),
                "east": round(east, 6),
                "north": round(north, 6),
            })

    return tiles


def print_grid(tiles: List[Dict[str, Any]]) -> None:
    """Pretty-print the tile grid to stdout."""
    for t in tiles:
        area_km2_est = (t["east"] - t["west"]) * (t["north"] - t["south"]) * (111_000**2) / 1_000_000
        print(
            f"  {t['tile_id']:>12s}  →  "
            f"[{t['west']:.4f}, {t['south']:.4f}, {t['east']:.4f}, {t['north']:.4f}]"
            f"  ~{area_km2_est:.1f} km²"
        )


def save_grid(tiles: List[Dict[str, Any]], path: str = "tiles.json") -> str:
    """Save tile grid to a JSON file.

    Args:
        tiles: List of tile dicts from generate_grid()
        path: Output JSON path

    Returns:
        Absolute path to saved file
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(tiles, f, indent=2)
    return str(out)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tile grid for Trnava city and save as tiles.json",
    )
    parser.add_argument("--west", type=float, default=17.5644, help="West of top-left tile")
    parser.add_argument("--south", type=float, default=48.3723, help="South of top-left tile")
    parser.add_argument("--tile-w", type=float, default=0.0262, help="Tile width in degrees lon (~5 km²)")
    parser.add_argument("--tile-h", type=float, default=0.0154, help="Tile height in degrees lat (~5 km²)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns (east-west)")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows (north-south)")
    parser.add_argument("--output", type=str, default="tiles.json", help="Output JSON file path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    tiles = generate_grid(
        west_start=args.west,
        south_start=args.south,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        n_cols=args.cols,
        n_rows=args.rows,
    )

    total_km2 = len(tiles) * (args.tile_w * args.tile_h * (111_000 ** 2) / 1_000_000)

    print(f"Tile grid: {args.cols}×{args.rows} = {len(tiles)} tiles  (~{total_km2:.0f} km²)")
    print(f"Tile size: {args.tile_w:.4f}° lon × {args.tile_h:.4f}° lat")
    print(f"Bounds:    W {args.west:.4f}  S {args.south:.4f}  "
          f"E {args.west + args.cols * args.tile_w:.4f}  "
          f"N {args.south + args.rows * args.tile_h:.4f}")
    print()

    print_grid(tiles)

    out_path = save_grid(tiles, args.output)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
