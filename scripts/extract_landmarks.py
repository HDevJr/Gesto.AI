from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.config import CONFIG
from src.vision.landmark_extractor import LandmarkExtractor


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_labels(labels_csv_path: Path) -> list[dict[str, str]]:
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {labels_csv_path}")

    with labels_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if reader.fieldnames is None or {"sample_id", "label"} - set(reader.fieldnames):
        raise ValueError("O CSV precisa ter as colunas: sample_id,label")

    return rows


def main() -> None:
    rows = load_labels(Path(CONFIG.labels_csv))
    extractor = LandmarkExtractor(include_face=False)

    processed = 0
    failed = 0

    try:
        for row in rows:
            sample_id = row["sample_id"].strip()
            label = row["label"].strip()

            video_path = Path(CONFIG.raw_videos_dir) / label / f"{sample_id}.mp4"
            output_path = Path(CONFIG.landmarks_dir) / label / f"{sample_id}.npy"

            if not video_path.exists():
                logger.warning("Video nao encontrado: %s", video_path)
                failed += 1
                continue

            try:
                sequence = extractor.extract_from_video(
                    video_path=video_path,
                    max_frames=None,
                    frame_step=1,
                )
                if sequence.ndim != 2 or sequence.shape[1] != CONFIG.training.input_size:
                    raise ValueError(
                        f"Shape invalido {sequence.shape}. "
                        f"Esperado: (frames, {CONFIG.training.input_size})."
                    )

                extractor.save_sequence(sequence, output_path)
                logger.info("Processado: %s -> %s | shape=%s", sample_id, output_path, sequence.shape)
                processed += 1
            except Exception as exc:
                logger.error("Falha ao processar %s: %s", video_path, exc)
                failed += 1
    finally:
        extractor.close()

    logger.info("Resumo: processados=%s | falhas=%s", processed, failed)


if __name__ == "__main__":
    main()
