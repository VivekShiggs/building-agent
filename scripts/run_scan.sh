#!/usr/bin/env bash
set -euo pipefail

# Run a building scan from the command line
# Usage: ./scripts/run_scan.sh [bbox_json]

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

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
