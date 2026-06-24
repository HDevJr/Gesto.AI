from __future__ import annotations


def build_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_idx, pred_idx in zip(y_true, y_pred):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            matrix[true_idx][pred_idx] += 1
    return matrix


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
) -> dict[str, object]:
    num_classes = len(idx_to_class)
    matrix = build_confusion_matrix(y_true, y_pred, num_classes)
    per_class: dict[str, dict[str, float | int]] = {}

    for class_idx in range(num_classes):
        true_positive = matrix[class_idx][class_idx]
        false_positive = sum(matrix[row][class_idx] for row in range(num_classes) if row != class_idx)
        false_negative = sum(matrix[class_idx][col] for col in range(num_classes) if col != class_idx)
        support = sum(matrix[class_idx])

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_class[idx_to_class[class_idx]] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        }

    total = len(y_true)
    correct = sum(1 for true_idx, pred_idx in zip(y_true, y_pred) if true_idx == pred_idx)
    accuracy = correct / total if total > 0 else 0.0
    macro_precision = sum(item["precision"] for item in per_class.values()) / num_classes if num_classes else 0.0
    macro_recall = sum(item["recall"] for item in per_class.values()) / num_classes if num_classes else 0.0
    macro_f1 = sum(item["f1_score"] for item in per_class.values()) / num_classes if num_classes else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
