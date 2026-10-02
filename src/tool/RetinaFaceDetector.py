# -*- coding: utf-8 -*-
"""RetinaFace face detector for the ID-photo layout pipeline.

Reuses the pure numpy pre/post-processing from the bundled
onnx_skin_retouching package (same face_detector.onnx file that the
skin-retouching feature uses) while exposing the same interface as the
YuNet-based FaceDetector, so PhotoEntity.detect_face() works unchanged.

Output rows follow the OpenCV FaceDetectorYN convention used by YuNet:
[x, y, w, h, score, lm0_x, lm0_y, ..., lm4_x, lm4_y] per detected face.
"""

import os

import numpy as np

from .deviceUtils import get_onnx_session
from .onnx_skin_retouching.face import postprocess_retinaface, preprocess_retinaface

# Cached detector instances keyed by (model_path, conf_threshold, nms_threshold, long_side).
_detector_cache = {}


class RetinaFaceDetector:
    """
    Face detector backed by RetinaFace via onnxruntime.

    :param model_path: Path to the face_detector.onnx model file
    :param conf_threshold: Confidence threshold for kept detections
    :param nms_threshold: Non-maximum suppression threshold
    :param long_side: Inference long side; larger inputs are scaled down first
    """

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4, long_side=1024):
        self.session = get_onnx_session(model_path)
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.long_side = long_side

    def process_array(self, image, origin_size=False):
        """
        Detect faces on an already-loaded BGR image.

        :param image: BGR image array to be processed
        :param origin_size: ignored; inference size is controlled by long_side
        :return: array with one [x, y, w, h, score, ...landmarks] row per face
        :rtype: numpy.ndarray
        """
        rgb_image = image[:, :, ::-1]
        nchw, resized_shape, resize_scale = preprocess_retinaface(
            rgb_image, long_side=self.long_side
        )
        feed = {name: nchw.astype(np.float32, copy=False) for name in self.input_names}
        outputs = self.session.run(self.output_names, feed)
        by_name = {name.lower(): value for name, value in zip(self.output_names, outputs)}

        def pick(*tokens):
            for name, value in by_name.items():
                if any(token in name for token in tokens):
                    return value
            return None

        loc = pick("loc")
        conf = pick("conf")
        landms = pick("landm")
        if loc is None or conf is None or landms is None:
            raise ValueError(
                "Unexpected face_detector.onnx outputs: {}".format(sorted(by_name))
            )
        boxes, scores, landmarks = postprocess_retinaface(
            loc,
            conf,
            landms,
            resized_shape=resized_shape,
            original_shape=rgb_image.shape[:2],
            resize_scale=resize_scale,
            confidence_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        rows = np.zeros((len(scores), 15), dtype=np.float32)
        rows[:, 0] = boxes[:, 0]
        rows[:, 1] = boxes[:, 1]
        rows[:, 2] = boxes[:, 2] - boxes[:, 0]
        rows[:, 3] = boxes[:, 3] - boxes[:, 1]
        rows[:, 4] = scores
        rows[:, 5:] = landmarks.reshape(len(scores), 10)
        return rows


def get_retinaface_detector(model_path, conf_threshold=0.5, nms_threshold=0.4, long_side=1024):
    """
    Return the cached RetinaFaceDetector for the given configuration.

    The ResNet50 model is too large to reload per image, so instances are cached.

    :param model_path: Path to the face_detector.onnx model file
    :return: cached or newly created RetinaFaceDetector
    :rtype: RetinaFaceDetector
    """
    key = (str(model_path), float(conf_threshold), float(nms_threshold), int(long_side))
    if key not in _detector_cache:
        if not os.path.exists(key[0]):
            raise FileNotFoundError(
                "Face detector model not found: {}. The face_detector.onnx file is shared "
                "by skin retouching and the optional RetinaFace layout detector; place it "
                "in the model directory to use this option.".format(key[0])
            )
        _detector_cache[key] = RetinaFaceDetector(*key)
    return _detector_cache[key]
