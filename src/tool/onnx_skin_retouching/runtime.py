# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope-free ONNXRuntime implementation of the skin retouching pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime

from .face import postprocess_retinaface, preprocess_retinaface
from .utils import (
    gen_diffuse_mask,
    get_crop_bbox,
    get_roi_without_padding,
    hwc_uint8_to_nchw_normalized,
    nchw_normalized_to_hwc01,
    pad_to_multiple_nchw,
    patch_aggregation_overlap,
    patch_partition_overlap,
    resize_nchw,
    resize_on_long_side,
    sigmoid,
    smooth_border_mg,
    whiten_img,
    ensure_bgr_uint8,
)


SKIN_MASK_MODEL = "skin_retouch_mask.onnx"
GENERATOR_MODEL = "retouch_generator.onnx"
LOCAL_DETECTION_MODEL = "local_detection.onnx"
LOCAL_INPAINTING_MODEL = "local_inpainting.onnx"
FACE_DETECTOR_MODEL = "face_detector.onnx"


@dataclass(frozen=True)
class FaceDetection:
    bbox: Sequence[float]
    score: float
    landmarks: Sequence[Sequence[float]]

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "score": float(self.score),
            "landmarks": [list(point) for point in self.landmarks],
        }


class OrtModel:
    """Small wrapper that records ONNX input/output names at startup."""

    def __init__(self, path: Path, providers: Sequence[str]):
        if not path.exists():
            raise FileNotFoundError(
                f"missing ONNX model: {path}. Export it first or pass a valid model_dir."
            )
        self.path = path
        self.session = onnxruntime.InferenceSession(str(path), providers=list(providers))
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]
        self.input_shapes = [node.shape for node in self.session.get_inputs()]
        self.output_shapes = [node.shape for node in self.session.get_outputs()]

    def run(self, *inputs: np.ndarray) -> List[np.ndarray]:
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"{self.path.name} expects {len(self.input_names)} inputs "
                f"{self.input_names}, got {len(inputs)}"
            )
        feed = {name: value.astype(np.float32, copy=False) for name, value in zip(self.input_names, inputs)}
        return self.session.run(self.output_names, feed)

    def run_dict(self, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        prepared = {name: value.astype(np.float32, copy=False) for name, value in feed.items()}
        return self.session.run(self.output_names, prepared)


def choose_providers(providers: Optional[Iterable[str]] = None) -> List[str]:
    """Choose ONNXRuntime providers.

    Default to CPU to avoid noisy CUDA provider load failures on machines that
    have onnxruntime-gpu installed but lack matching CUDA/cuDNN DLLs. Pass
    providers explicitly to opt in to GPU execution.
    """
    if providers is not None:
        selected = list(providers)
        if not selected:
            raise ValueError("providers cannot be empty")
        return selected
    return ["CPUExecutionProvider"]


class SkinRetoucher:
    """Complete ONNXRuntime skin-retouching pipeline without ModelScope.

    The bundled face detector exporter writes raw RetinaFace outputs
    ``loc/conf/landms``. This runtime decodes priors, landmarks and NMS in numpy,
    while still accepting a post-processed detector with boxes/scores/keypoints.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = ".",
        providers: Optional[Iterable[str]] = None,
        retouch_degree: float = 0.7,
        whitening_degree: float = 0.8,
        enable_local: bool = False,
        min_face_score: float = 0.5,
        input_size: int = 512,
        patch_size: int = 512,
        skin_mask_long_side: int = 800,
        face_detector_long_side: int = 1024,
        face_nms_threshold: float = 0.4,
    ):
        self.model_dir = Path(model_dir)
        self.providers = choose_providers(providers)
        self.retouch_degree = float(retouch_degree)
        self.whitening_degree = float(whitening_degree)
        self.enable_local = bool(enable_local)
        self.min_face_score = float(min_face_score)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.skin_mask_long_side = int(skin_mask_long_side)
        self.face_detector_long_side = int(face_detector_long_side)
        self.face_nms_threshold = float(face_nms_threshold)

        self.skin_mask_model = OrtModel(self.model_dir / SKIN_MASK_MODEL, self.providers)
        self.generator_model = OrtModel(self.model_dir / GENERATOR_MODEL, self.providers)
        self.face_detector_model = OrtModel(self.model_dir / FACE_DETECTOR_MODEL, self.providers)
        self.local_detection_model: Optional[OrtModel] = None
        self.local_inpainting_model: Optional[OrtModel] = None
        if self.enable_local:
            self.local_detection_model = OrtModel(self.model_dir / LOCAL_DETECTION_MODEL, self.providers)
            self.local_inpainting_model = OrtModel(self.model_dir / LOCAL_INPAINTING_MODEL, self.providers)

        self.diffuse_mask = gen_diffuse_mask()

    def retouch(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run the full retouching pipeline and return a cv2-compatible BGR image."""
        image_bgr = ensure_bgr_uint8(image_bgr)
        rgb_image = image_bgr[:, :, ::-1].copy()

        skin_mask = None
        if self.whitening_degree > 0:
            skin_mask = self._run_skin_mask(rgb_image)

        output_rgb = rgb_image.copy()
        face_results = [det.as_dict() for det in self._detect_faces(rgb_image)]
        crop_bboxes = get_crop_bbox(face_results, min_score=self.min_face_score)
        if not crop_bboxes:
            return output_rgb[:, :, ::-1]

        flag_big_kernel = False
        for bbox in crop_bboxes:
            try:
                roi_rgb, _, crop_tblr = get_roi_without_padding(rgb_image, bbox)
            except ValueError:
                continue
            if roi_rgb.shape[0] > 0.4 * rgb_image.shape[0]:
                flag_big_kernel = True

            roi_nchw = hwc_uint8_to_nchw_normalized(roi_rgb)
            if self.enable_local:
                roi_nchw = self._retouch_local(roi_nchw)

            roi_pred = self._predict_roi(roi_nchw)
            top, bottom, left, right = crop_tblr
            output_rgb[top:bottom, left:right] = roi_pred

        if skin_mask is not None and self.whitening_degree > 0:
            output_rgb = whiten_img(
                output_rgb,
                skin_mask,
                self.whitening_degree,
                flag_big_kernel=flag_big_kernel,
            )

        return output_rgb[:, :, ::-1]

    def _run_skin_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        small_rgb, _ = resize_on_long_side(rgb_image, self.skin_mask_long_side)
        return self.skin_mask_model.run(small_rgb.astype(np.float32))[0]

    def _detect_faces(self, rgb_image: np.ndarray) -> List[FaceDetection]:
        face_input, resized_shape, resize_scale = preprocess_retinaface(
            rgb_image,
            long_side=self.face_detector_long_side,
        )
        outputs = self.face_detector_model.run(face_input)
        boxes, scores, keypoints = self._parse_face_outputs(
            outputs,
            resized_shape=resized_shape,
            original_shape=rgb_image.shape[:2],
            resize_scale=resize_scale,
        )
        detections: List[FaceDetection] = []
        for box, score, points in zip(boxes, scores, keypoints):
            score_value = float(np.asarray(score).reshape(-1)[0])
            if score_value < self.min_face_score:
                continue
            detections.append(
                FaceDetection(
                    bbox=np.asarray(box, dtype=np.float32).reshape(4).tolist(),
                    score=score_value,
                    landmarks=np.asarray(points, dtype=np.float32).reshape(5, 2).tolist(),
                )
            )
        return detections

    def _parse_face_outputs(
        self,
        outputs: Sequence[np.ndarray],
        resized_shape: Tuple[int, int],
        original_shape: Tuple[int, int],
        resize_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        by_name = {name.lower(): value for name, value in zip(self.face_detector_model.output_names, outputs)}

        def pick(*tokens: str) -> Optional[np.ndarray]:
            for name, value in by_name.items():
                if any(token in name for token in tokens):
                    return value
            return None

        loc = pick("loc")
        conf = pick("conf")
        landms = pick("landm")
        if loc is not None and conf is not None and landms is not None:
            return postprocess_retinaface(
                loc,
                conf,
                landms,
                resized_shape=resized_shape,
                original_shape=original_shape,
                resize_scale=resize_scale,
                confidence_threshold=self.min_face_score,
                nms_threshold=self.face_nms_threshold,
            )

        boxes = pick("box", "bbox")
        scores = pick("score")
        keypoints = pick("keypoint", "landmark", "point")

        if boxes is None or scores is None or keypoints is None:
            if len(outputs) < 3:
                raise ValueError(
                    f"{FACE_DETECTOR_MODEL} must output raw loc/conf/landms or "
                    f"post-processed boxes/scores/keypoints; "
                    f"got outputs {self.face_detector_model.output_names}"
                )
            boxes, scores, keypoints = outputs[:3]

        boxes = np.asarray(boxes).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        keypoints = np.asarray(keypoints).reshape(-1, 5, 2)
        count = min(len(boxes), len(scores), len(keypoints))
        return boxes[:count], scores[:count], keypoints[:count]

    def _retouch_local(self, image_nchw: np.ndarray) -> np.ndarray:
        if self.local_detection_model is None or self.local_inpainting_model is None:
            raise RuntimeError("enable_local=True requires local_detection.onnx and local_inpainting.onnx")

        _, _, sub_h, sub_w = image_nchw.shape
        standard_image = resize_nchw(image_nchw, (768, 768), interpolation=cv2.INTER_LINEAR)
        mask_logits = self.local_detection_model.run(standard_image)[0]
        mask_pred = sigmoid(mask_logits)
        mask_pred = resize_nchw(mask_pred, (sub_h, sub_w), interpolation=cv2.INTER_NEAREST)

        hard_low = (mask_pred >= 0.35).astype(np.float32)
        hard_high = (mask_pred >= 0.5).astype(np.float32)
        mask_pred = mask_pred * (1.0 - hard_high) + hard_high
        mask_pred = mask_pred * hard_low
        mask_pred = 1.0 - mask_pred

        image_padded, (padded_h, padded_w) = pad_to_multiple_nchw(image_nchw, self.patch_size)
        mask_padded, _ = pad_to_multiple_nchw(mask_pred, self.patch_size)
        image_patches = patch_partition_overlap(image_padded, p1=self.patch_size, p2=self.patch_size)
        mask_patches = patch_partition_overlap(mask_padded, p1=self.patch_size, p2=self.patch_size)

        composed_patches = []
        for image_patch, mask_patch in zip(image_patches, mask_patches):
            image_patch = image_patch[None, ...]
            mask_patch = mask_patch[None, ...]
            masked_image = image_patch * mask_patch
            inpainted = self.local_inpainting_model.run(masked_image, mask_patch)[0]
            composed = masked_image + (1.0 - mask_patch) * inpainted
            composed_patches.append(composed[0])

        composed_stack = np.stack(composed_patches, axis=0)
        rows = int(round(padded_h / self.patch_size))
        cols = int(round(padded_w / self.patch_size))
        composed = patch_aggregation_overlap(composed_stack, h=rows, w=cols)
        return composed[:, :, :sub_h, :sub_w]

    def _predict_roi(self, roi_nchw: np.ndarray) -> np.ndarray:
        _, _, roi_h, roi_w = roi_nchw.shape
        image = resize_nchw(
            roi_nchw,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        pred_mg = self.generator_model.run(image)[0]
        pred_mg = (pred_mg - 0.5) * self.retouch_degree + 0.5
        pred_mg = np.clip(pred_mg, 0.0, 1.0)
        pred_mg = resize_nchw(pred_mg, (roi_h, roi_w), interpolation=cv2.INTER_LINEAR)
        pred_mg_hwc = pred_mg[0].transpose(1, 2, 0)
        if pred_mg_hwc.ndim == 2:
            pred_mg_hwc = pred_mg_hwc[..., None]
        pred_mg_hwc = smooth_border_mg(self.diffuse_mask, pred_mg_hwc)

        image_hwc = nchw_normalized_to_hwc01(roi_nchw)
        pred = (1.0 - 2.0 * pred_mg_hwc) * image_hwc * image_hwc + 2.0 * pred_mg_hwc * image_hwc
        return np.clip(pred * 255.0, 0, 255).astype(np.uint8)
