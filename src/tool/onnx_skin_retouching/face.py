# Copyright (c) Alibaba, Inc. and its affiliates.
"""RetinaFace numpy post-processing for the ONNXRuntime path."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np


RETINAFACE_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
RETINAFACE_STEPS = (8, 16, 32)
RETINAFACE_VARIANCE = (0.1, 0.2)


def preprocess_retinaface(
    rgb_image: np.ndarray,
    long_side: int = 1024,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Resize, pad and normalize an RGB image for RetinaFace."""
    original_h, original_w = rgb_image.shape[:2]
    scale = 1.0
    image = rgb_image
    if max(original_h, original_w) > long_side:
        scale = float(long_side) / float(max(original_h, original_w))
        new_w = max(1, int(round(original_w * scale)))
        new_h = max(1, int(round(original_h * scale)))
        image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    resized_h, resized_w = image.shape[:2]
    padded_h = int(math.ceil(resized_h / 32.0) * 32)
    padded_w = int(math.ceil(resized_w / 32.0) * 32)
    padded = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    bgr = image[:, :, ::-1].astype(np.float32)
    bgr /= 255.0
    padded[:resized_h, :resized_w] = bgr
    nchw = padded.transpose(2, 0, 1)[None, ...]
    return nchw, (resized_h, resized_w), scale


def prior_box(
    image_size: Tuple[int, int],
    min_sizes: Sequence[Sequence[int]] = RETINAFACE_MIN_SIZES,
    steps: Sequence[int] = RETINAFACE_STEPS,
    clip: bool = False,
) -> np.ndarray:
    """Generate RetinaFace priors for an image size in H,W order."""
    image_h, image_w = image_size
    anchors = []
    feature_maps = [
        [int(math.ceil(float(image_h) / step)), int(math.ceil(float(image_w) / step))]
        for step in steps
    ]
    for k, feature_map in enumerate(feature_maps):
        min_sizes_for_level = min_sizes[k]
        for i in range(feature_map[0]):
            for j in range(feature_map[1]):
                for min_size in min_sizes_for_level:
                    s_kx = float(min_size) / float(image_w)
                    s_ky = float(min_size) / float(image_h)
                    dense_cx = (j + 0.5) * steps[k] / float(image_w)
                    dense_cy = (i + 0.5) * steps[k] / float(image_h)
                    anchors.append([dense_cx, dense_cy, s_kx, s_ky])
    output = np.asarray(anchors, dtype=np.float32)
    if clip:
        output = np.clip(output, 0.0, 1.0)
    return output


def decode_boxes(loc: np.ndarray, priors: np.ndarray, variance: Sequence[float]) -> np.ndarray:
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variance[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre: np.ndarray, priors: np.ndarray, variance: Sequence[float]) -> np.ndarray:
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variance[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms


def nms(dets: np.ndarray, threshold: float) -> List[int]:
    if dets.size == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess_retinaface(
    loc: np.ndarray,
    conf: np.ndarray,
    landms: np.ndarray,
    resized_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
    resize_scale: float,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    keep_top_k: int = 750,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw RetinaFace outputs to boxes, scores and five landmarks."""
    loc = np.asarray(loc)[0]
    conf = np.asarray(conf)[0]
    landms = np.asarray(landms)[0]

    resized_h, resized_w = resized_shape
    padded_h = int(math.ceil(resized_h / 32.0) * 32)
    padded_w = int(math.ceil(resized_w / 32.0) * 32)
    priors = prior_box((padded_h, padded_w))
    count = min(len(priors), len(loc), len(conf), len(landms))
    priors = priors[:count]
    loc = loc[:count]
    conf = conf[:count]
    landms = landms[:count]

    boxes = decode_boxes(loc, priors, RETINAFACE_VARIANCE)
    boxes *= np.array([padded_w, padded_h, padded_w, padded_h], dtype=np.float32)
    landmarks = decode_landmarks(landms, priors, RETINAFACE_VARIANCE)
    landmarks *= np.array(
        [padded_w, padded_h, padded_w, padded_h, padded_w, padded_h, padded_w, padded_h, padded_w, padded_h],
        dtype=np.float32,
    )

    scores = conf[:, 1]
    valid = scores > confidence_threshold
    boxes = boxes[valid]
    landmarks = landmarks[valid]
    scores = scores[valid]
    if boxes.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 5, 2), dtype=np.float32),
        )

    order = scores.argsort()[::-1][:keep_top_k]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
    keep = nms(dets, nms_threshold)
    boxes = boxes[keep]
    landmarks = landmarks[keep]
    scores = scores[keep]

    inv_scale = 1.0 / max(resize_scale, 1e-12)
    boxes *= inv_scale
    landmarks *= inv_scale

    original_h, original_w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
    landmarks[:, 0::2] = np.clip(landmarks[:, 0::2], 0, original_w - 1)
    landmarks[:, 1::2] = np.clip(landmarks[:, 1::2], 0, original_h - 1)
    return boxes.astype(np.float32), scores.astype(np.float32), landmarks.reshape(-1, 5, 2).astype(np.float32)
