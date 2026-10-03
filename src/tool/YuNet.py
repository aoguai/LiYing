import cv2 as cv
import numpy as np
import os
import sys

class YuNet:
    """
    YuNet face detector class.

    :param model_path: Path to the model file
    :type model_path: str
    :param input_size: Size of the input image, in the form [w, h], default is [320, 320]
    :type input_size: list[int]
    :param conf_threshold: Confidence threshold, default is 0.6
    :type conf_threshold: float
    :param nms_threshold: Non-maximum suppression threshold, default is 0.3
    :type nms_threshold: float
    :param top_k: Number of top detections to keep, default is 5000
    :type top_k: int
    :param backend_id: ID of the backend to use, default is 0
    :type backend_id: int
    :param target_id: ID of the target device, default is 0
    :type target_id: int
    :return: None
    :rtype: None
    """

    def __init__(self, model_path=None, input_size=[320, 320], conf_threshold=0.6, nms_threshold=0.3, top_k=5000,
                 backend_id=0, target_id=0):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'model', 'face_detection_yunet_2023mar.onnx')
        assert model_path.endswith('.onnx'), f"invalid onnx model: {model_path}"
        assert os.path.exists(model_path), f"model not found: {model_path}"
        self._model_path = model_path
        self._input_size = tuple(input_size)  # [w, h]
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    @property
    def name(self):
        return self.__class__.__name__

    def set_backend_and_target(self, backend_id, target_id):
        """
        Set the backend ID and target ID.

        :param backend_id: Backend ID
        :type backend_id: int
        :param target_id: Target ID
        :type target_id: int
        :return: None
        :rtype: None
        """
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def set_input_size(self, input_size):
        """
        Set the size of the input image.

        :param input_size: Size of the input image, in the form [w, h]
        :type input_size: list[int]
        :return: None
        :rtype: None
        """
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        """
        Perform inference to detect faces in the image.

        :param image: The image to be processed
        :type image: numpy.ndarray
        :return: Detected face information, a numpy array of shape [n, 15], where each row represents a detected face with 15 elements: [x1, y1, x2, y2, score, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7]
        :rtype: numpy.ndarray
        """
        # Forward inference
        faces = self._model.detect(image)
        return faces[1]


class FaceDetector:
    """
    Face detector class.

    :param model_path: Path to the model file
    :type model_path: str
    :param conf_threshold: Minimum confidence threshold, default is 0.9
    :type conf_threshold: float
    :param nms_threshold: Non-maximum suppression threshold, default is 0.3
    :type nms_threshold: float
    :param top_k: Number of top detections to keep, default is 5000
    :type top_k: int
    :param backend_id: Backend ID, default is cv2.dnn.DNN_BACKEND_OPENCV
    :type backend_id: int
    :param target_id: Target ID, default is cv2.dnn.DNN_TARGET_CPU
    :type target_id: int
    :return: None
    :rtype: None
    """

    def __init__(self, model_path, conf_threshold=0.9, nms_threshold=0.3, top_k=5000,
                 backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU):
        self.model = YuNet(model_path=model_path,
                           input_size=[320, 320],
                           conf_threshold=conf_threshold,
                           nms_threshold=nms_threshold,
                           top_k=top_k,
                           backend_id=backend_id,
                           target_id=target_id)

    @staticmethod
    def _ensure_bgr_image(image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim == 2:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv.cvtColor(image[:, :, 0], cv.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        return image

    def process_image(self, image_path, origin_size=False):
        """
        Process the image for face detection.

        :param image_path: Path to the image file to be processed
        :type image_path: str
        :param origin_size: Whether to keep the original size
        :type origin_size: bool
        :return: Detected face information, a numpy array of shape [n, 15], where each row represents a detected face with 15 elements: [x1, y1, x2, y2, score, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7]
        :rtype: numpy.ndarray
        """
        image = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image from {image_path}")
        return self.process_array(image, origin_size=origin_size)

    def process_array(self, image, origin_size=False):
        """
        Process an already-loaded image for face detection.

        :param image: Image array to be processed
        :type image: numpy.ndarray
        :param origin_size: Whether to keep the original size
        :type origin_size: bool
        :return: Detected face information
        :rtype: numpy.ndarray
        """
        image = self._ensure_bgr_image(image)
        h, w, _ = image.shape
        target_size = 320
        max_size = 320
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize_factor = float(target_size) / float(im_size_min)

        if np.round(resize_factor * im_size_max) > max_size:
            resize_factor = float(max_size) / float(im_size_max)

        if origin_size:
            resize_factor = 1

        if resize_factor != 1:
            image = cv.resize(image, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_LINEAR)
            h, w, _ = image.shape

        self.model.set_input_size([w, h])
        results = self.model.infer(image)
        if results is not None:
            if resize_factor != 1:
                results = results[:, :15] / resize_factor
        else:
            results = []

        return results
