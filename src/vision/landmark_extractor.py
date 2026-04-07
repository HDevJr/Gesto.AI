from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.vision.mediapipe_detector import MediaPipeDetector


class LandmarkExtractor:
    """
    Extrai landmarks frame a frame de um vídeo e salva como sequência.
    """

    def __init__(self, include_face: bool = False) -> None:
        self.detector = MediaPipeDetector(include_face=include_face)

    def extract_from_video(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
    ) -> np.ndarray:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        sequence = []
        frame_index = 0
        kept_frames = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                result = self.detector.process_frame(frame)
                flattened = self.detector.flatten_result(result)
                sequence.append(flattened)

                kept_frames += 1
                frame_index += 1

                if max_frames is not None and kept_frames >= max_frames:
                    break
        finally:
            cap.release()

        if not sequence:
            raise ValueError(f"Nenhum frame processado em: {video_path}")

        return np.stack(sequence).astype(np.float32)

    def save_sequence(self, sequence: np.ndarray, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, sequence)

    def close(self) -> None:
        self.detector.close()