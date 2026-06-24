from __future__ import annotations

import numpy as np


def pad_or_truncate_sequence(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Padroniza a sequência para um tamanho fixo.

    - Se a sequência for menor: adiciona zeros no final
    - Se for maior: corta no tamanho alvo
    """
    current_length, num_features = sequence.shape

    if current_length == target_length:
        return sequence.astype(np.float32)

    if current_length > target_length:
        return sequence[:target_length].astype(np.float32)

    padded = np.zeros((target_length, num_features), dtype=np.float32)
    padded[:current_length] = sequence
    return padded