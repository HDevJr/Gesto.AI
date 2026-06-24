from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.lstm_model import LSTMGestureClassifier


REQUIRED_CHECKPOINT_KEYS = {
    "state_dict",
    "class_to_idx",
    "idx_to_class",
    "sequence_length",
    "input_size",
    "hidden_size",
    "num_layers",
    "num_classes",
}


def load_lstm_checkpoint(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[LSTMGestureClassifier, dict[str, Any], dict[int, str]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint em formato antigo ou invalido. "
            "Execute novamente: python -m scripts.train_lstm"
        )

    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing_keys:
        raise ValueError(f"Checkpoint incompleto. Chaves ausentes: {sorted(missing_keys)}")

    model = LSTMGestureClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = {int(idx): class_name for idx, class_name in checkpoint["idx_to_class"].items()}
    return model, checkpoint, idx_to_class
