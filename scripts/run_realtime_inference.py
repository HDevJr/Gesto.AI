from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np
import torch

from src.config import CONFIG
from src.inference.checkpoint import load_lstm_checkpoint
from src.vision.mediapipe_detector import MediaPipeDetector


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    model, checkpoint, idx_to_class = load_lstm_checkpoint(CONFIG.model_checkpoint, CONFIG.device)
    sequence_length = int(checkpoint["sequence_length"])
    input_size = int(checkpoint["input_size"])
    confidence_threshold = CONFIG.inference.confidence_threshold

    logger.info("Checkpoint carregado: %s", CONFIG.model_checkpoint)
    logger.info("Classes disponiveis: %s", idx_to_class)
    logger.info("Device: %s", CONFIG.device)

    detector = MediaPipeDetector(include_face=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Nao foi possivel abrir a camera.")

    sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    predicted_label = "Aguardando..."
    predicted_confidence = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.error("Falha ao capturar frame da camera.")
                break

            result = detector.process_frame(frame)
            flattened = detector.flatten_result(result)
            if flattened.shape[0] != input_size:
                logger.error("Frame com input_size invalido: %s. Esperado: %s.", flattened.shape[0], input_size)
                break
            sequence_buffer.append(flattened)

            if len(sequence_buffer) == sequence_length:
                sequence_array = np.array(sequence_buffer, dtype=np.float32)
                x = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(CONFIG.device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs, dim=1).item())
                    predicted_confidence = float(probs[0, pred_idx].item())
                    if predicted_confidence < confidence_threshold:
                        predicted_label = "incerto"
                    else:
                        predicted_label = idx_to_class.get(pred_idx, f"classe_{pred_idx}")

            text_1 = f"Predicao: {predicted_label}"
            text_2 = f"Confianca: {predicted_confidence:.2%}"
            text_3 = f"Frames no buffer: {len(sequence_buffer)}/{sequence_length}"
            text_4 = "ESC para sair"

            cv2.putText(frame, text_1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, text_2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, text_3, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, text_4, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Inferencia em Tempo Real", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
