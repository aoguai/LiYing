import os
import sys
import cv2 as cv
from PIL import Image, ImageOps
import numpy as np

from .yolov8_detector import YOLOv8Detector
from .YuNet import FaceDetector
from .agpic import ImageCompressor


class PhotoEntity:
    def __init__(self, img_path, yolov8_model_path=None, yunet_model_path=None, y_b=False):
        """
        Initialize the PhotoEntity class.

        :param img_path: Path to the image
        :param yolov8_model_path: Path to the YOLOv8 model
        :param yunet_model_path: Path to the YuNet model
        :param y_b: Whether to compress the image, defaults to False
        """
        self.img_path = img_path
        self.image = self._load_image(img_path)
        
        # Use default model paths if not provided
        if yolov8_model_path is None:
            yolov8_model_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'model', 'yolov8n-pose.onnx')
        if yunet_model_path is None:
            yunet_model_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'model', 'face_detection_yunet_2023mar.onnx')
            
        self.yolov8_detector = YOLOv8Detector(yolov8_model_path)
        self.face_detector = FaceDetector(yunet_model_path)
        self.ImageCompressor_detector = ImageCompressor()
        if y_b:
            self._compress_image()

        # Initialize detection result attributes
        self.person_bbox = None
        self.person_label = None
        self.person_keypoints = None
        self.person_width = None
        self.person_height = None
        self.face_bbox = None
        self.face_width = None
        self.face_height = None
        self.print_size = None
        self.resolution = None
        self.detect()

    @staticmethod
    def _to_bgr_image(image_np):
        if image_np.ndim == 2:
            return cv.cvtColor(image_np, cv.COLOR_GRAY2BGR)
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            return cv.cvtColor(image_np, cv.COLOR_RGBA2BGR)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            return cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported image shape: {image_np.shape}")

    def _load_image(self, image_path):
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image_np = np.array(image)

        return self._to_bgr_image(image_np)

    def _compress_image(self):
        """
        Compress the image to reduce memory usage.
        """
        ext = os.path.splitext(self.img_path)[1].lower()
        encode_format = '.jpg' if ext in ['.jpg', '.jpeg'] else '.png'

        # Convert OpenCV image to byte format
        is_success, buffer = cv.imencode(encode_format, self.image)
        if not is_success:
            raise ValueError("Failed to encode the image to byte format")

        image_bytes = buffer.tobytes()

        # Call compress_image_from_bytes function to compress the image
        compressed_bytes = self.ImageCompressor_detector.compress_image_from_bytes(image_bytes)

        # Convert the compressed bytes back to OpenCV image format
        self.image = cv.imdecode(np.frombuffer(compressed_bytes, np.uint8), cv.IMREAD_COLOR)

    def get_print_info(self):
        """Get the print size and resolution information."""
        return {
            'print_size': self.print_size,
            'resolution': self.resolution
        }

    def detect(self, detect_person=True, detect_face=True):
        """
        Detect persons and faces in the image.

        :param detect_person: Whether to detect persons, defaults to True
        :param detect_face: Whether to detect faces, defaults to True
        """
        if detect_person:
            self.detect_person()
        if detect_face:
            self.detect_face()

    def detect_person(self):
        """
        Detect persons in the image.
        """
        person_result, _ = self.yolov8_detector.detect_person_image(self.image)
        if person_result:
            self.person_bbox = person_result['bbox_xyxy']
            self.person_label = person_result['bbox_label']
            self.person_keypoints = person_result['bbox_keypoints']
            self.person_width = self.person_bbox[2] - self.person_bbox[0]
            self.person_height = self.person_bbox[3] - self.person_bbox[1]
        else:
            self._reset_person_data()

    def detect_face(self):
        """
        Detect faces in the image.
        """
        face_results = self.face_detector.process_array(self.image)
        if not (face_results is None) and len(face_results) > 0:
            self.face_bbox = face_results[0][:4].astype('uint32')
            self.face_width = int(self.face_bbox[2]) - int(self.face_bbox[0])
            self.face_height = int(self.face_bbox[3]) - int(self.face_bbox[1])
        else:
            self._reset_face_data()

    def _reset_person_data(self):
        """
        Reset person detection data.
        """
        self.person_bbox = None
        self.person_label = None
        self.person_keypoints = None
        self.person_width = None
        self.person_height = None

    def _reset_face_data(self):
        """
        Reset face detection data.
        """
        self.face_bbox = None
        self.face_width = None
        self.face_height = None

    def set_img_path(self, img_path):
        """
        Set the image path and re-detect.

        :param img_path: New image path
        """
        self.img_path = img_path
        self.image = self._load_image(img_path)
        self.detect()

    def set_yolov8_model_path(self, model_path):
        """
        Set the YOLOv8 model path and re-detect.

        :param model_path: New YOLOv8 model path
        """
        self.yolov8_detector = YOLOv8Detector(model_path)
        self.detect()

    def set_yunet_model_path(self, model_path):
        """
        Set the YuNet model path and re-detect.

        :param model_path: New YuNet model path
        """
        self.face_detector = FaceDetector(model_path)
        self.detect()

    def manually_set_person_data(self, bbox, label, keypoints):
        """
        Manually set person detection data.

        :param bbox: Person bounding box
        :param label: Person label
        :param keypoints: Person keypoints
        """
        self.person_bbox = bbox
        self.person_label = label
        self.person_keypoints = keypoints
        self.person_width = self.person_bbox[2] - self.person_bbox[0]
        self.person_height = self.person_bbox[3] - self.person_bbox[1]

    def manually_set_face_data(self, bbox):
        """
        Manually set face detection data.

        :param bbox: Face bounding box
        """
        self.face_bbox = bbox
        self.face_width = self.face_bbox[2] - self.face_bbox[0]
        self.face_height = self.face_bbox[3] - self.face_bbox[1]
