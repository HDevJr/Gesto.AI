from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw" / "videos"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
LANDMARKS_DIR = DATA_DIR / "interim" / "landmarks_raw"

LABELS_CSV = ANNOTATIONS_DIR / "labels.csv"
CLASSES_JSON = ANNOTATIONS_DIR / "classes.json"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODEL_CHECKPOINT = CHECKPOINTS_DIR / "lstm_gesture_model.pt"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
