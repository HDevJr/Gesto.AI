from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_confusion_matrix_csv(
    matrix: list[list[int]],
    idx_to_class: dict[int, str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *labels])
        for idx, row in enumerate(matrix):
            writer.writerow([labels[idx], *row])


def save_summary_text(summary: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
