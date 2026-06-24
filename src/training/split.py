from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil

from src.datasets.dataset_loader import GestureDataset


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    val_indices: list[int]
    strategy: str
    warnings: list[str]


def create_train_val_split(
    dataset: GestureDataset,
    validation_fraction: float,
    seed: int,
) -> SplitResult:
    warnings: list[str] = []
    labels_by_index = [dataset.class_to_idx[label] for _, label in dataset.samples]
    indices_by_class: dict[int, list[int]] = {}

    for index, class_idx in enumerate(labels_by_index):
        indices_by_class.setdefault(class_idx, []).append(index)

    can_stratify = (
        len(dataset) >= 2
        and len(indices_by_class) >= 2
        and all(len(indices) >= 2 for indices in indices_by_class.values())
    )

    if not can_stratify:
        warnings.append(
            "Dataset pequeno demais para split treino/validacao independente e estratificado. "
            "Fallback: todos os dados serao usados em treino e validacao tecnica. "
            "As metricas de validacao nao devem ser interpretadas como generalizacao."
        )
        all_indices = list(range(len(dataset)))
        return SplitResult(
            train_indices=all_indices,
            val_indices=all_indices,
            strategy="fallback_all_data_train_and_validation",
            warnings=warnings,
        )

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_indices in indices_by_class.values():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, ceil(len(shuffled) * validation_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        strategy="stratified",
        warnings=warnings,
    )
