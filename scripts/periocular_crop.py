#!/usr/bin/env python3
"""Landmark-based periocular crops from MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
IRISES = list(range(468, 478))
UPPER_FACE = [10, 338, 297, 332, 284, 251, 389, 356, 127, 162, 21, 54, 103, 67, 109]
PERIOCULAR_LANDMARKS = LEFT_EYE + RIGHT_EYE + LEFT_BROW + RIGHT_BROW + IRISES + UPPER_FACE


@dataclass(frozen=True)
class PeriocularCrop:
    tensor: torch.Tensor
    bbox: tuple[int, int, int, int]
    source_size: tuple[int, int]
    landmark_count: int


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0 = max(0, min(width - 1, x0))
    x1 = max(1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(1, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_from_landmarks(
    landmarks: np.ndarray,
    image_size: tuple[int, int],
    landmark_indices: Sequence[int] = PERIOCULAR_LANDMARKS,
    horizontal_pad: float = 0.22,
    upper_pad: float = 0.34,
    lower_pad: float = 0.18,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    indices = [idx for idx in landmark_indices if idx < len(landmarks)]
    if not indices:
        return None

    points = landmarks[indices, :2].copy()
    points[:, 0] *= width
    points[:, 1] *= height

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    x0 = int(np.floor(min_xy[0] - span[0] * horizontal_pad))
    x1 = int(np.ceil(max_xy[0] + span[0] * horizontal_pad))
    y0 = int(np.floor(min_xy[1] - span[1] * upper_pad))
    y1 = int(np.ceil(max_xy[1] + span[1] * lower_pad))
    return clamp_bbox(x0, y0, x1, y1, width=width, height=height)


def image_to_model_tensor(image: Image.Image, output_size: tuple[int, int]) -> torch.Tensor:
    crop_width, crop_height = output_size
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(crop_height, crop_width), mode="bilinear", align_corners=False).squeeze(0)
    return (tensor - 0.5) / 0.5


class PeriocularCropper:
    """MediaPipe Face Mesh cropper for static face images."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        refine_landmarks: bool = True,
        landmark_indices: Iterable[int] = PERIOCULAR_LANDMARKS,
    ):
        import mediapipe as mp

        self._face_mesh_module = mp.solutions.face_mesh
        self._face_mesh = self._face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )
        self._landmark_indices = tuple(landmark_indices)

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "PeriocularCropper":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def landmarks(self, image: Image.Image) -> np.ndarray | None:
        rgb = np.asarray(image.convert("RGB"))
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        points = results.multi_face_landmarks[0].landmark
        return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32)

    def crop(
        self,
        image_or_path: Image.Image | str | Path,
        output_size: tuple[int, int] = (160, 96),
    ) -> PeriocularCrop | None:
        image = Image.open(image_or_path).convert("RGB") if not isinstance(image_or_path, Image.Image) else image_or_path.convert("RGB")
        landmarks = self.landmarks(image)
        if landmarks is None:
            return None
        bbox = bbox_from_landmarks(
            landmarks,
            image_size=image.size,
            landmark_indices=self._landmark_indices,
        )
        if bbox is None:
            return None
        crop = image.crop(bbox)
        return PeriocularCrop(
            tensor=image_to_model_tensor(crop, output_size=output_size),
            bbox=bbox,
            source_size=image.size,
            landmark_count=len(landmarks),
        )
