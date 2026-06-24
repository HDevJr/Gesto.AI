from __future__ import annotations

import torch
import torch.nn as nn


class LSTMGestureClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 258,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)

        # hidden shape: (num_layers, batch, hidden_size)
        last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)
        return logits