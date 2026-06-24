from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.padding import pad_or_truncate_sequence


class GestureDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        landmarks_dir: str | Path,
        classes_json: str | Path,
        sequence_length: int = 100,
        input_size: int = 258,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.landmarks_dir = Path(landmarks_dir)
        self.classes_json = Path(classes_json)
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.warnings: list[str] = []
        self.class_counts: dict[str, int] = {}

        self.defined_classes = self._load_defined_classes()
        self.samples = self._load_samples()
        self.class_to_idx = self._build_active_class_mapping()
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def _load_defined_classes(self) -> list[str]:
        if not self.classes_json.exists():
            raise FileNotFoundError(f"Arquivo de classes nao encontrado: {self.classes_json}")
        with self.classes_json.open("r", encoding="utf-8") as f:
            idx_to_class = json.load(f)

        return [idx_to_class[idx] for idx in sorted(idx_to_class, key=lambda value: int(value))]

    def _load_samples(self) -> list[tuple[Path, str]]:
        samples = []

        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Arquivo de labels nao encontrado: {self.labels_csv}")

        with self.labels_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or {"sample_id", "label"} - set(reader.fieldnames):
                raise ValueError("O CSV de labels precisa ter as colunas: sample_id,label")

            for row in reader:
                sample_id = row["sample_id"].strip()
                label = row["label"].strip()

                if label not in self.defined_classes:
                    raise ValueError(f"Label '{label}' existe em labels.csv, mas nao existe em classes.json.")

                npy_path = self.landmarks_dir / label / f"{sample_id}.npy"
                if not npy_path.exists():
                    self.warnings.append(f"Amostra ignorada sem arquivo .npy: {npy_path}")
                    continue

                self._validate_npy_shape(npy_path)
                samples.append((npy_path, label))
                self.class_counts[label] = self.class_counts.get(label, 0) + 1

        classes_without_samples = [
            class_name for class_name in self.defined_classes if self.class_counts.get(class_name, 0) == 0
        ]
        if classes_without_samples:
            self.warnings.append(
                "Classes definidas sem amostras .npy e excluidas do treino: "
                + ", ".join(classes_without_samples)
            )

        if not samples:
            raise ValueError("Nenhuma amostra .npy encontrada para o dataset.")

        return samples

    def _validate_npy_shape(self, npy_path: Path) -> None:
        sequence = np.load(npy_path, mmap_mode="r")
        if sequence.ndim != 2:
            raise ValueError(f"Arquivo {npy_path} deve ter shape 2D, encontrado {sequence.shape}.")
        if sequence.shape[1] != self.input_size:
            raise ValueError(
                f"Arquivo {npy_path} tem input_size {sequence.shape[1]}, esperado {self.input_size}."
            )

    def _build_active_class_mapping(self) -> dict[str, int]:
        active_classes = [class_name for class_name in self.defined_classes if self.class_counts.get(class_name, 0) > 0]
        return {class_name: idx for idx, class_name in enumerate(active_classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[index]

        sequence = np.load(npy_path).astype(np.float32)
        sequence = pad_or_truncate_sequence(sequence, self.sequence_length)

        x = torch.tensor(sequence, dtype=torch.float32)
        y = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        return x, y
