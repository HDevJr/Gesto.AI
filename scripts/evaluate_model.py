from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.datasets.dataset_loader import GestureDataset
from src.evaluation.metrics import classification_metrics
from src.evaluation.reports import save_confusion_matrix_csv, save_metrics_json, save_summary_text
from src.inference.checkpoint import load_lstm_checkpoint
from src.training.loop import run_epoch


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_class_mapping(
    checkpoint_class_to_idx: dict[str, int],
    dataset_class_to_idx: dict[str, int],
) -> None:
    if checkpoint_class_to_idx != dataset_class_to_idx:
        raise ValueError(
            "Mapeamento de classes do checkpoint difere do dataset atual. "
            f"checkpoint={checkpoint_class_to_idx} dataset={dataset_class_to_idx}"
        )


def build_summary(
    *,
    started_at: str,
    dataset: GestureDataset,
    metrics: dict[str, object],
    loss: float,
    warnings: list[str],
    report_paths: dict[str, str],
) -> str:
    lines = [
        "Gesto.AI - Resumo de Avaliacao",
        f"Data/hora: {started_at}",
        f"Amostras avaliadas: {len(dataset)}",
        f"Classes avaliadas: {dataset.class_to_idx}",
        f"Amostras por classe: {dataset.class_counts}",
        "",
        "Metricas:",
        f"- loss: {loss:.4f}",
        f"- accuracy: {metrics.get('accuracy', 0.0):.4f}",
        f"- macro_precision: {metrics.get('macro_precision', 0.0):.4f}",
        f"- macro_recall: {metrics.get('macro_recall', 0.0):.4f}",
        f"- macro_f1_score: {metrics.get('macro_f1_score', 0.0):.4f}",
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
    model, checkpoint, idx_to_class = load_lstm_checkpoint(CONFIG.model_checkpoint, CONFIG.device)

    dataset = GestureDataset(
        labels_csv=CONFIG.labels_csv,
        landmarks_dir=CONFIG.landmarks_dir,
        classes_json=CONFIG.classes_json,
        sequence_length=int(checkpoint["sequence_length"]),
        input_size=int(checkpoint["input_size"]),
    )
    validate_class_mapping(checkpoint["class_to_idx"], dataset.class_to_idx)

    warnings = list(dataset.warnings)
    if len(dataset) < CONFIG.training.min_samples_per_class_warning * len(dataset.class_to_idx):
        warnings.append(
            "Dataset pequeno. Esta avaliacao usa os dados disponiveis e nao substitui "
            "um conjunto de teste independente."
        )

    for warning in warnings:
        logger.warning(warning)

    logger.info("Amostras avaliadas: %s", len(dataset))
    logger.info("Classes avaliadas: %s", dataset.class_to_idx)
    logger.info("Device: %s", CONFIG.device)

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.training.batch_size,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    result = run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=CONFIG.device,
    )
    metrics = classification_metrics(
        y_true=result["y_true"],
        y_pred=result["y_pred"],
        idx_to_class=idx_to_class,
    )

    metrics_payload = {
        "run_type": "evaluation",
        "started_at": started_at,
        "checkpoint_path": CONFIG.model_checkpoint,
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": idx_to_class,
        "class_counts": dataset.class_counts,
        "dataset_warnings": warnings,
        "metrics": metrics,
    }

    metrics_dir = Path(CONFIG.metrics_dir)
    report_paths = {
        "metrics_json": str(metrics_dir / "evaluation_metrics.json"),
        "confusion_matrix_csv": str(metrics_dir / "evaluation_confusion_matrix.csv"),
        "summary_txt": str(metrics_dir / "evaluation_summary.txt"),
    }
    save_metrics_json(metrics_payload, report_paths["metrics_json"])
    save_confusion_matrix_csv(
        metrics["confusion_matrix"],
        idx_to_class,
        report_paths["confusion_matrix_csv"],
    )
    save_summary_text(
        build_summary(
            started_at=started_at,
            dataset=dataset,
            metrics=metrics,
            loss=float(result["loss"]),
            warnings=warnings,
            report_paths=report_paths,
        ),
        report_paths["summary_txt"],
    )

    logger.info(
        "Avaliacao concluida | loss=%.4f | acc=%.4f | macro_f1=%.4f",
        result["loss"],
        metrics["accuracy"],
        metrics["macro_f1_score"],
    )
    logger.info("Relatorios salvos em: %s", metrics_dir)


if __name__ == "__main__":
    main()
