# -*- coding: utf-8 -*-
"""Skin retouching adapter around the bundled onnx_skin_retouching package.

Keeps the upstream package untouched and provides:
- a lazily created, cached SkinRetoucher per model directory (the retoucher
  loads three ONNX sessions at construction, so it must not be rebuilt per image);
- ONNXRuntime providers aligned with the project-wide deviceUtils policy;
- missing-model checks with actionable guidance and degree clamping.
"""

import os

import numpy as np

from .deviceUtils import ONNX_PROVIDER
from .onnx_skin_retouching import SkinRetoucher

# Same provider policy as deviceUtils.get_onnx_session: GPU first with CPU
# fallback when CUDA is available, plain CPU otherwise.
_SKIN_RETOUCH_PROVIDERS = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if ONNX_PROVIDER == 'CUDAExecutionProvider'
    else ['CPUExecutionProvider']
)

# Model files that must exist inside the skin-retouch model directory.
REQUIRED_MODELS = ('skin_retouch_mask.onnx', 'retouch_generator.onnx', 'face_detector.onnx')

# Cached retoucher instances, keyed by absolute model directory path.
_retoucher_cache = {}


def check_skin_retouch_models(model_dir):
    """Raise FileNotFoundError listing the missing model files and how to fix it."""
    missing = [name for name in REQUIRED_MODELS
               if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Skin retouching model(s) not found in '{}': {}. ".format(
                model_dir, ', '.join(missing))
            + "Place skin_retouch_mask.onnx, retouch_generator.onnx and "
            "face_detector.onnx in this directory to enable skin retouching."
        )


def get_retoucher(model_dir):
    """Return the cached SkinRetoucher for the directory, creating it on first use."""
    key = os.path.abspath(model_dir)
    if key not in _retoucher_cache:
        check_skin_retouch_models(model_dir)
        _retoucher_cache[key] = SkinRetoucher(
            model_dir=model_dir,
            providers=_SKIN_RETOUCH_PROVIDERS,
        )
    return _retoucher_cache[key]


def retouch_image(image_bgr, model_dir, retouch_degree=0.7, whitening_degree=0.8):
    """Retouch a BGR uint8 image; returns a new BGR uint8 image of the same size.

    Degrees are instance-level settings on the shared retoucher, so they are
    assigned per call instead of rebuilding the model sessions.
    """
    retoucher = get_retoucher(model_dir)
    retoucher.retouch_degree = float(np.clip(retouch_degree, 0.0, 1.0))
    retoucher.whitening_degree = float(np.clip(whitening_degree, 0.0, 1.0))
    return retoucher.retouch(image_bgr)
