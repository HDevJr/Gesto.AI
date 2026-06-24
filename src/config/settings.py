from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from src.config.paths import (
    CLASSES_JSON,
    LABELS_CSV,
    LANDMARKS_DIR,
    METRICS_DIR,
    MODEL_CHECKPOINT,
    RAW_VIDEOS_DIR,
)


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 100
    input_size: int = 258
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 2
    epochs: int = 10
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    random_seed: int = 42
    min_samples_total: int = 2
    min_samples_per_class_warning: int = 5


@dataclass(frozen=True)
class InferenceConfig:
    confidence_threshold: float = 0.60


@dataclass(frozen=True)
class ProjectConfig:
    raw_videos_dir: str = str(RAW_VIDEOS_DIR)
    labels_csv: str = str(LABELS_CSV)
    classes_json: str = str(CLASSES_JSON)
    landmarks_dir: str = str(LANDMARKS_DIR)
    model_checkpoint: str = str(MODEL_CHECKPOINT)
    metrics_dir: str = str(METRICS_DIR)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["training"] = asdict(self.training)
        data["inference"] = asdict(self.inference)
        return data


CONFIG = ProjectConfig()
