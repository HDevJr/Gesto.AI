from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path

import cv2

from src.config import CONFIG, PROJECT_ROOT


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
LABELS_FIELDNAMES = ["sample_id", "label", "path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coleta videos de gestos por webcam para o dataset Gesto.AI."
    )
    parser.add_argument("--label", required=True, help="Nome da classe/sinal. Exemplo: oi")
    parser.add_argument("--samples", type=int, default=1, help="Quantidade de videos a gravar.")
    parser.add_argument("--duration", type=float, default=3.0, help="Duracao de cada gravacao em segundos.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da camera OpenCV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(CONFIG.raw_videos_dir),
        help="Diretorio base de saida. Padrao: data/raw/videos",
    )
    parser.add_argument(
        "--create-label",
        action="store_true",
        help="Cria a classe em classes.json caso ela ainda nao exista.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Contagem regressiva antes de cada gravacao.",
    )
    return parser.parse_args()


def validate_label_name(label: str) -> str:
    normalized = label.strip()
    if not normalized:
        raise ValueError("Label nao pode ser vazio.")
    if not LABEL_PATTERN.match(normalized):
        raise ValueError("Label deve conter apenas letras, numeros, '_' ou '-'.")
    return normalized


def load_classes(classes_path: Path) -> dict[str, str]:
    if not classes_path.exists():
        raise FileNotFoundError(f"Arquivo de classes nao encontrado: {classes_path}")
    with classes_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_classes(classes_path: Path, classes: dict[str, str]) -> None:
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {str(idx): classes[str(idx)] for idx in sorted(map(int, classes.keys()))}
    with classes_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.write("\n")


def ensure_label_exists(label: str, classes_path: Path, create_label: bool) -> None:
    classes = load_classes(classes_path)
    if label in classes.values():
        return

    if not create_label:
        raise ValueError(
            f"Classe '{label}' nao existe em {classes_path}. "
            "Use --create-label para adiciona-la explicitamente."
        )

    next_idx = max((int(idx) for idx in classes.keys()), default=-1) + 1
    classes[str(next_idx)] = label
    save_classes(classes_path, classes)
    logger.info("Classe adicionada em %s: %s -> %s", classes_path, next_idx, label)


def read_labels(labels_csv: Path, raw_videos_dir: Path) -> list[dict[str, str]]:
    if not labels_csv.exists():
        return []

    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if {"sample_id", "label"} - set(reader.fieldnames):
            raise ValueError("labels.csv precisa conter pelo menos as colunas sample_id,label.")

        rows = []
        for row in reader:
            sample_id = row["sample_id"].strip()
            label = row["label"].strip()
            path = row.get("path", "").strip()
            if not path:
                path = str(Path("data/raw/videos") / label / f"{sample_id}.mp4")
            rows.append({"sample_id": sample_id, "label": label, "path": path})
        return rows


def write_labels(labels_csv: Path, rows: list[dict[str, str]]) -> None:
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABELS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def add_label_row(labels_csv: Path, row: dict[str, str], raw_videos_dir: Path) -> bool:
    rows = read_labels(labels_csv, raw_videos_dir)
    existing_sample_ids = {item["sample_id"] for item in rows}
    existing_paths = {item["path"] for item in rows}

    if row["sample_id"] in existing_sample_ids or row["path"] in existing_paths:
        logger.warning("Registro ja existe em labels.csv e sera ignorado: %s", row)
        return False

    rows.append(row)
    write_labels(labels_csv, rows)
    return True


def next_sample_index(label_dir: Path, label: str, labels_csv: Path, raw_videos_dir: Path) -> int:
    max_index = 0
    for path in label_dir.glob(f"{label}_*.mp4"):
        match = re.fullmatch(rf"{re.escape(label)}_(\d+)\.mp4", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    for row in read_labels(labels_csv, raw_videos_dir):
        if row["label"] != label:
            continue
        match = re.fullmatch(rf"{re.escape(label)}_(\d+)", row["sample_id"])
        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def put_centered_text(frame, text: str, y: int, scale: float, color: tuple[int, int, int]) -> None:
    thickness = 2
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (frame.shape[1] - size[0]) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def wait_for_start(cap: cv2.VideoCapture, label: str, sample_number: int, total_samples: int) -> bool:
    while True:
        success, frame = cap.read()
        if not success:
            raise RuntimeError("Falha ao capturar frame da camera.")

        put_centered_text(frame, f"Classe: {label}", 60, 0.9, (0, 255, 0))
        put_centered_text(frame, f"Amostra {sample_number}/{total_samples}", 105, 0.8, (255, 255, 0))
        put_centered_text(frame, "ESPACO inicia | ESC/Q cancela", 155, 0.7, (230, 230, 230))
        cv2.imshow("Gesto.AI - Coleta de Dados", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return True
        if key in (27, ord("q"), ord("Q")):
            return False


def run_countdown(cap: cv2.VideoCapture, seconds: int) -> bool:
    for remaining in range(seconds, 0, -1):
        start = time.monotonic()
        while time.monotonic() - start < 1.0:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Falha ao capturar frame da camera.")

            put_centered_text(frame, f"Prepare-se: {remaining}", 100, 1.3, (0, 255, 255))
            put_centered_text(frame, "ESC/Q cancela", 150, 0.7, (230, 230, 230))
            cv2.imshow("Gesto.AI - Coleta de Dados", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return False
    return True


def record_video(
    cap: cv2.VideoCapture,
    output_path: Path,
    duration: float,
    fps: float,
    frame_size: tuple[int, int],
    sample_number: int,
    total_samples: int,
) -> tuple[int, bool]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o arquivo de video: {output_path}")

    frames_written = 0
    cancelled = False
    start = time.monotonic()

    try:
        while time.monotonic() - start < duration:
            success, frame = cap.read()
            if not success:
                break

            elapsed = time.monotonic() - start
            remaining = max(0.0, duration - elapsed)
            writer.write(frame)
            frames_written += 1

            cv2.putText(frame, "GRAVANDO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Amostra {sample_number}/{total_samples}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Tempo restante: {remaining:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Gesto.AI - Coleta de Dados", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                cancelled = True
                break
    finally:
        writer.release()

    return frames_written, cancelled


def validate_saved_video(video_path: Path, min_frames: int) -> bool:
    if not video_path.exists() or video_path.stat().st_size == 0:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count >= min_frames


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    label = validate_label_name(args.label)

    if args.samples < 1:
        raise ValueError("--samples deve ser maior ou igual a 1.")
    if args.duration <= 0:
        raise ValueError("--duration deve ser maior que 0.")

    raw_videos_dir = args.output_dir
    labels_csv = Path(CONFIG.labels_csv)
    classes_path = Path(CONFIG.classes_json)

    ensure_label_exists(label, classes_path, args.create_label)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {args.camera}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_size = (width, height)
    min_frames = max(1, int(fps * args.duration * 0.5))

    label_dir = raw_videos_dir / label
    current_index = next_sample_index(label_dir, label, labels_csv, raw_videos_dir)
    saved_count = 0

    logger.info("Coleta iniciada | label=%s | samples=%s | duration=%.1fs", label, args.samples, args.duration)
    logger.info("Saida: %s", label_dir)

    try:
        for sample_offset in range(args.samples):
            sample_number = sample_offset + 1
            if not wait_for_start(cap, label, sample_number, args.samples):
                logger.info("Coleta cancelada antes da amostra %s.", sample_number)
                break
            if not run_countdown(cap, args.countdown):
                logger.info("Coleta cancelada durante a contagem regressiva.")
                break

            sample_id = f"{label}_{current_index:03d}"
            output_path = label_dir / f"{sample_id}.mp4"
            while output_path.exists():
                current_index += 1
                sample_id = f"{label}_{current_index:03d}"
                output_path = label_dir / f"{sample_id}.mp4"

            frames_written, cancelled = record_video(
                cap=cap,
                output_path=output_path,
                duration=args.duration,
                fps=fps,
                frame_size=frame_size,
                sample_number=sample_number,
                total_samples=args.samples,
            )

            if cancelled:
                logger.info("Coleta cancelada durante a gravacao. Amostra parcial descartada: %s", output_path)
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Nao foi possivel remover video parcial: %s", output_path)
                break

            if not validate_saved_video(output_path, min_frames):
                logger.error(
                    "Gravacao invalida: %s | frames=%s | minimo=%s. Registro nao sera adicionado.",
                    output_path,
                    frames_written,
                    min_frames,
                )
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Nao foi possivel remover video invalido: %s", output_path)
                continue

            relative_path = relative_project_path(output_path)
            row = {"sample_id": sample_id, "label": label, "path": relative_path}
            if add_label_row(labels_csv, row, raw_videos_dir):
                saved_count += 1
                logger.info("Video salvo e anotado: %s | frames=%s", output_path, frames_written)

            current_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info("Coleta finalizada. Videos validos adicionados: %s/%s", saved_count, args.samples)
    logger.info("Proximos comandos:")
    logger.info("python -m scripts.extract_landmarks")
    logger.info("python -m scripts.train_lstm")
    logger.info("python -m scripts.evaluate_model")


if __name__ == "__main__":
    main()
