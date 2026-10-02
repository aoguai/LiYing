# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure numpy/opencv utilities for the ONNXRuntime skin retouching path."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


CropBBox = Tuple[int, int, int, int]
CropTLBR = Tuple[int, int, int, int]


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize cv2-style input to a three-channel BGR uint8 image."""
    if image is None:
        raise ValueError("input image is None")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxW, HxWx3, or HxWx4 image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_on_long_side(image: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """Resize image so its long side equals ``long_side``."""
    h, w = image.shape[:2]
    if max(h, w) == 0:
        raise ValueError("image has an empty spatial dimension")
    scale = float(long_side) / float(max(h, w))
    if scale == 1.0:
        return image.copy(), scale
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def get_crop_bbox(
    detections: Iterable[dict],
    min_score: float = 0.5,
    expand_ratio: float = 1.5,
) -> List[CropBBox]:
    """Convert face detector results to square-ish crop boxes.

    The original ModelScope utils are not shipped in this repository. This
    local version keeps the same contract: each output bbox is later clipped
    by ``get_roi_without_padding`` and can extend outside the image.
    """
    crop_bboxes: List[CropBBox] = []
    for det in detections:
        score = float(det.get("score", 1.0))
        if score < min_score:
            continue

        bbox = np.asarray(det["bbox"], dtype=np.float32).reshape(4)
        x1, y1, x2, y2 = bbox.tolist()
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        side = max(width, height) * expand_ratio
        left = cx - side * 0.5
        right = left + side
        top = cy - side * 0.5
        bottom = top + side
        crop_bboxes.append(
            (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
            )
        )
    return crop_bboxes


def get_roi_without_padding(
    rgb_image: np.ndarray,
    bbox: Sequence[int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int], CropTLBR]:
    """Clip a crop bbox to image bounds and return ROI plus paste coordinates."""
    h, w = rgb_image.shape[:2]
    left, top, right, bottom = [int(v) for v in bbox]
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(w, right)
    clipped_bottom = min(h, bottom)
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        raise ValueError(f"empty ROI after clipping bbox {bbox} to image shape {rgb_image.shape}")
    roi = rgb_image[clipped_top:clipped_bottom, clipped_left:clipped_right].copy()
    expand = (
        clipped_left - left,
        clipped_top - top,
        right - clipped_right,
        bottom - clipped_bottom,
    )
    crop_tblr = (clipped_top, clipped_bottom, clipped_left, clipped_right)
    return roi, expand, crop_tblr


def hwc_uint8_to_nchw_normalized(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 HWC image to NCHW float32 in [-1, 1]."""
    arr = rgb_image.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[None, ...]


def nchw_normalized_to_hwc01(image: np.ndarray) -> np.ndarray:
    """Convert NCHW float image in [-1, 1] to HWC float in [0, 1]."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"expected 1xCxHxW tensor, got {image.shape}")
    return ((image[0].transpose(1, 2, 0) + 1.0) * 0.5).clip(0.0, 1.0)


def resize_nchw(
    tensor: np.ndarray,
    size_hw: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize a NCHW tensor channel-by-channel using opencv."""
    if tensor.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got {tensor.shape}")
    new_h, new_w = size_hw
    batches = []
    for batch in tensor:
        channels = []
        for channel in batch:
            channels.append(cv2.resize(channel, (new_w, new_h), interpolation=interpolation))
        batches.append(np.stack(channels, axis=0))
    return np.stack(batches, axis=0).astype(tensor.dtype, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gen_diffuse_mask(size: int = 500, border: int = 20, out_channels: int = 3) -> np.ndarray:
    """Generate a soft square mask used to dampen blend-layer borders."""
    mask = np.ones((size, size), dtype=np.float32)
    if border <= 0:
        return np.dstack([mask] * out_channels)
    for i in range(size):
        for j in range(size):
            if border <= i <= size - border and border <= j <= size - border:
                mask[i, j] = 1.0
            elif i <= border:
                mask[i, j] = i * 1.0 / border
            elif i > size - border:
                mask[i, j] = (size - i) * 1.0 / border
    for i in range(size):
        for j in range(size):
            if j <= border:
                mask[i, j] = min(mask[i, j], j * 1.0 / border)
            elif j > size - border:
                mask[i, j] = min(mask[i, j], (size - j) * 1.0 / border)
    return np.dstack([mask] * out_channels)


def smooth_border_mg(diffuse_mask: np.ndarray, pred_mg: np.ndarray) -> np.ndarray:
    """Blend pred_mg toward neutral 0.5 near ROI borders."""
    h, w = pred_mg.shape[:2]
    mask = cv2.resize(diffuse_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.ndim == 2:
        mask = mask[..., None]
    return (pred_mg - 0.5) * mask + 0.5


def _as_hwc_mask(skin_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(skin_mask)
    if mask.ndim == 4:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.squeeze()
    if mask.ndim == 3 and mask.shape[0] in (1, 3) and mask.shape[-1] not in (1, 3):
        mask = mask.transpose(1, 2, 0)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.ndim != 3:
        raise ValueError(f"unsupported skin mask shape {skin_mask.shape}")
    return mask.astype(np.float32)


def whiten_img(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
    whitening_degree: float,
    flag_big_kernel: bool = False,
) -> np.ndarray:
    """Apply the same blend-layer whitening formula used by the original wrapper."""
    image = image_rgb.astype(np.float32) / 255.0
    mask = _as_hwc_mask(skin_mask)
    if mask.shape[2] >= 3:
        mask = mask[..., 2:3]

    mask = np.clip(mask / 255.0, 0.0, 1.0)
    kernel_size = 80 if flag_big_kernel else 30
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_2d = mask[..., 0]
    mask_2d = cv2.dilate(mask_2d, kernel, iterations=1)
    mask_2d = cv2.erode(mask_2d, kernel, iterations=1)
    mask_2d = cv2.blur(mask_2d, (20, 20))
    mask_2d = cv2.resize(
        mask_2d,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    whiten_mg = np.repeat(mask_2d[..., None], 3, axis=2)
    whiten_mg[..., 1:] *= 0.75
    whiten_mg = whiten_mg * 0.2 * float(whitening_degree) + 0.5
    pred = (1.0 - 2.0 * whiten_mg) * image * image + 2.0 * whiten_mg * image
    return np.clip(pred * 255.0, 0, 255).astype(np.uint8)


def pad_to_multiple_nchw(tensor: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad NCHW tensor on bottom/right to a spatial multiple."""
    if tensor.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got {tensor.shape}")
    h, w = tensor.shape[2:]
    padded_h = h if h % multiple == 0 else (h // multiple + 1) * multiple
    padded_w = w if w % multiple == 0 else (w // multiple + 1) * multiple
    result = np.zeros((tensor.shape[0], tensor.shape[1], padded_h, padded_w), dtype=tensor.dtype)
    result[:, :, :h, :w] = tensor
    return result, (padded_h, padded_w)


def patch_partition_overlap(
    tensor: np.ndarray,
    p1: int = 512,
    p2: int = 512,
    padding: int = 32,
) -> np.ndarray:
    """Partition a 1xCxHxW tensor into overlapped patches."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"expected 1xCxHxW tensor, got {tensor.shape}")
    _, _, h, w = tensor.shape
    padded = np.pad(
        tensor,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    patches = []
    for top in range(0, h, p1):
        for left in range(0, w, p2):
            patches.append(padded[:, :, top:top + p1 + padding * 2, left:left + p2 + padding * 2][0])
    return np.stack(patches, axis=0)


def patch_aggregation_overlap(
    patches: np.ndarray,
    h: int,
    w: int,
    padding: int = 32,
) -> np.ndarray:
    """Aggregate overlapped patches back to a 1xCxHxW tensor."""
    if patches.ndim != 4:
        raise ValueError(f"expected NxCxHxW patches, got {patches.shape}")
    _, channels, patch_h, patch_w = patches.shape
    tile_h = patch_h - padding * 2
    tile_w = patch_w - padding * 2
    result = np.zeros((1, channels, h * tile_h, w * tile_w), dtype=patches.dtype)
    idx = 0
    for row in range(h):
        for col in range(w):
            result[
                :,
                :,
                row * tile_h:(row + 1) * tile_h,
                col * tile_w:(col + 1) * tile_w,
            ] = patches[idx:idx + 1, :, padding:-padding, padding:-padding]
            idx += 1
    return result
