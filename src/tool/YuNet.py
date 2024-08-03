import cv2 as cv
import numpy as np


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

    def __init__(self, model_path, input_size=[320, 320], conf_threshold=0.6, nms_threshold=0.3, top_k=5000,
                 backend_id=0,
                 target_id=0):
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
