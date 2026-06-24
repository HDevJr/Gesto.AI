from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import CONFIG
from src.datasets.dataset_loader import GestureDataset
from src.evaluation.metrics import classification_metrics
from src.evaluation.reports import save_confusion_matrix_csv, save_metrics_json, save_summary_text
from src.models.lstm_model import LSTMGestureClassifier
from src.training.loop import run_epoch
from src.training.split import create_train_val_split


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_summary(
    *,
    started_at: str,
    dataset: GestureDataset,
    split_strategy: str,
    warnings: list[str],
    final_metrics: dict[str, object],
    report_paths: dict[str, str],
) -> str:
    lines = [
        "Gesto.AI - Resumo do Treinamento",
        f"Data/hora: {started_at}",
        f"Amostras validas: {len(dataset)}",
        f"Classes usadas: {dataset.class_to_idx}",
        f"Amostras por classe: {dataset.class_counts}",
        f"Estrategia de split: {split_strategy}",
        "",
        "Metricas finais de validacao:",
        f"- accuracy: {final_metrics.get('accuracy', 0.0):.4f}",
        f"- macro_precision: {final_metrics.get('macro_precision', 0.0):.4f}",
        f"- macro_recall: {final_metrics.get('macro_recall', 0.0):.4f}",
        f"- macro_f1_score: {final_metrics.get('macro_f1_score', 0.0):.4f}",
        "",
        "Relatorios:",
        f"- metricas_json: {report_paths['metrics_json']}",
        f"- confusion_matrix_csv: {report_paths['confusion_matrix_csv']}",
        f"- summary_txt: {report_paths['summary_txt']}",
    ]

    if warnings:
        lines.extend(["", "Avisos:"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines) + "\n"


def main() -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    training = CONFIG.training

    dataset = GestureDataset(
        labels_csv=CONFIG.labels_csv,
        landmarks_dir=CONFIG.landmarks_dir,
        classes_json=CONFIG.classes_json,
        sequence_length=training.sequence_length,
        input_size=training.input_size,
    )

    warnings = list(dataset.warnings)
    for warning in dataset.warnings:
        logger.warning(warning)

    num_classes = len(dataset.class_to_idx)
    if len(dataset) < training.min_samples_total:
        raise ValueError(
            f"Dataset insuficiente: {len(dataset)} amostras. "
            f"Minimo configurado: {training.min_samples_total}."
        )
    if num_classes < 2:
        raise ValueError("Treino requer pelo menos duas classes com amostras .npy validas.")

    for class_name, count in dataset.class_counts.items():
        if count < training.min_samples_per_class_warning:
            warning = f"Classe '{class_name}' tem apenas {count} amostra(s). Alto risco de overfitting."
            logger.warning(warning)
            warnings.append(warning)

    split = create_train_val_split(
        dataset=dataset,
        validation_fraction=training.validation_fraction,
        seed=training.random_seed,
    )
    warnings.extend(split.warnings)
    for warning in split.warnings:
        logger.warning(warning)

    logger.info("Amostras validas: %s", len(dataset))
    logger.info("Classes usadas no treino: %s", dataset.class_to_idx)
    logger.info("Split: %s | treino=%s | validacao=%s", split.strategy, len(split.train_indices), len(split.val_indices))
    logger.info("Device: %s", CONFIG.device)

    train_loader = DataLoader(
        Subset(dataset, split.train_indices),
        batch_size=training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, split.val_indices),
        batch_size=training.batch_size,
        shuffle=False,
    )

    model = LSTMGestureClassifier(
        input_size=training.input_size,
        hidden_size=training.hidden_size,
        num_layers=training.num_layers,
        num_classes=num_classes,
        dropout=training.dropout,
    ).to(CONFIG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    history: list[dict[str, float | int]] = []
    final_val_result: dict[str, object] = {}

    for epoch in range(training.epochs):
        train_result = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=CONFIG.device,
            optimizer=optimizer,
        )
        val_result = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=CONFIG.device,
        )
        final_val_result = val_result

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_result["loss"]),
            "train_accuracy": float(train_result["accuracy"]),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
        }
        history.append(epoch_metrics)

        logger.info(
            "Epoch %s/%s | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch + 1,
            training.epochs,
            epoch_metrics["train_loss"],
            epoch_metrics["train_accuracy"],
            epoch_metrics["val_loss"],
            epoch_metrics["val_accuracy"],
        )

    final_metrics = classification_metrics(
        y_true=final_val_result["y_true"],
        y_pred=final_val_result["y_pred"],
        idx_to_class=dataset.idx_to_class,
    )

    metrics_payload = {
        "run_type": "training",
        "started_at": started_at,
        "split_strategy": split.strategy,
        "train_indices": split.train_indices,
        "val_indices": split.val_indices,
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.idx_to_class,
        "class_counts": dataset.class_counts,
        "dataset_warnings": warnings,
        "history": history,
        "final_validation": final_metrics,
    }

    metrics_dir = Path(CONFIG.metrics_dir)
    report_paths = {
        "metrics_json": str(metrics_dir / "training_metrics.json"),
        "confusion_matrix_csv": str(metrics_dir / "training_confusion_matrix.csv"),
        "summary_txt": str(metrics_dir / "training_summary.txt"),
    }
    save_metrics_json(metrics_payload, report_paths["metrics_json"])
    save_confusion_matrix_csv(
        final_metrics["confusion_matrix"],
        dataset.idx_to_class,
        report_paths["confusion_matrix_csv"],
    )
    save_summary_text(
        build_summary(
            started_at=started_at,
            dataset=dataset,
            split_strategy=split.strategy,
            warnings=warnings,
            final_metrics=final_metrics,
            report_paths=report_paths,
        ),
        report_paths["summary_txt"],
    )

    checkpoint = {
        "checkpoint_version": 2,
        "state_dict": model.state_dict(),
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.idx_to_class,
        "sequence_length": training.sequence_length,
        "input_size": training.input_size,
        "hidden_size": training.hidden_size,
        "num_layers": training.num_layers,
        "num_classes": num_classes,
        "dropout": training.dropout,
        "config": CONFIG.to_dict(),
        "training_started_at": started_at,
        "class_counts": dataset.class_counts,
        "dataset_warnings": warnings,
        "split_strategy": split.strategy,
        "train_indices": split.train_indices,
        "val_indices": split.val_indices,
        "history": history,
        "final_metrics": final_metrics,
        "report_paths": report_paths,
    }

    output_path = Path(CONFIG.model_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    logger.info("Checkpoint completo salvo em: %s", output_path)
    logger.info("Relatorios salvos em: %s", metrics_dir)


if __name__ == "__main__":
    main()
