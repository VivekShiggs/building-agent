#!/usr/bin/env bash
set -euo pipefail

# Export training data and fine-tune YOLO model
# Usage: ./scripts/run_training.sh [scan_id]

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

SCAN_ID="${1:-}"

python -c "
import sys
from agent.config import load_config
from agent.store import BuildingStore
from agent.training import export_yolo_labels

config = load_config()
store = BuildingStore(config.storage.database_path)

scan_id = '$SCAN_ID' or None
dataset_yaml = export_yolo_labels(store, scan_id=scan_id)
if dataset_yaml:
    print(f'Training data exported: {dataset_yaml}')
    print('To fine-tune:')
    print(f'  yolo segment train model={config.model.name} data={dataset_yaml} epochs=100 imgsz=640')
else:
    print('No training data to export')
    sys.exit(1)
"
