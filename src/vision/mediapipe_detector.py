from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DetectionResult:
    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray


class MediaPipeDetector:
    """
    Wrapper para MediaPipe Holistic.

    Extração inicial:
    - mãos esquerda e direita
    - pose
    - face opcional

    Saída:
    - arrays numéricos fixos
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        refine_face_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        include_face: bool = False,
    ) -> None:
        self.include_face = include_face
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.holistic.close()

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)
        pose = self._extract_pose(results.pose_landmarks)
        face = self._extract_face(results.face_landmarks) if self.include_face else np.array([], dtype=np.float32)

        return DetectionResult(
            left_hand=left_hand,
            right_hand=right_hand,
            pose=pose,
            face=face,
        )

    @staticmethod
    def _extract_hand(hand_landmarks: Optional[object]) -> np.ndarray:
        """
        21 landmarks * 3 coordenadas = 63 valores
        """
        if hand_landmarks is None:
            return np.zeros(21 * 3, dtype=np.float32)

        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _extract_pose(pose_landmarks: Optional[object]) -> np.ndarray:
        """
        33 landmarks * 4 valores = x, y, z, visibility
        """
        if pose_landmarks is None:
            return np.zeros(33 * 4, dtype=np.float32)

        coords = []
        for lm in pose_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _extract_face(face_landmarks: Optional[object]) -> np.ndarray:
        """
        Face completa é grande. Para começo, só salvamos se include_face=True.
        468 landmarks * 3 coordenadas.
        """
        if face_landmarks is None:
            return np.zeros(468 * 3, dtype=np.float32)

        coords = []
        for lm in face_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def flatten_result(self, result: DetectionResult) -> np.ndarray:
        """
        Junta tudo em um vetor único por frame.
        """
        parts = [result.left_hand, result.right_hand, result.pose]
        if self.include_face:
            parts.append(result.face)
        return np.concatenate(parts).astype(np.float32)