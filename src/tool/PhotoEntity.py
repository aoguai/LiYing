import os
import cv2 as cv
from PIL import Image, ExifTags
import numpy as np

from yolov8_detector import YOLOv8Detector
from YuNet import FaceDetector
from agpic import ImageCompressor


class PhotoEntity:
    def __init__(self, img_path, yolov8_model_path, yunet_model_path, y_b=False):
        """
        Initialize the PhotoEntity class.

        :param img_path: Path to the image
        :param yolov8_model_path: Path to the YOLOv8 model
        :param yunet_model_path: Path to the YuNet model
        :param y_b: Whether to compress the image, defaults to False
        """
        self.img_path = img_path
        self.image = self._correct_image_orientation(img_path)
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
        self.detect()

    def _correct_image_orientation(self, image_path):
        # Open the image and read EXIF information
        image = Image.open(image_path)
        try:
            exif = image._getexif()
            if exif is not None:
                # Get EXIF tags
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS:
                        if ExifTags.TAGS[tag] == 'Orientation':
                            orientation = value
                            # Adjust the image based on orientation
                            if orientation == 3:
                                image = image.rotate(180, expand=True)
                            elif orientation == 6:
                                image = image.rotate(270, expand=True)
                            elif orientation == 8:
                                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError) as e:
            raise e

        # Convert Pillow image object to OpenCV image object
        image_np = np.array(image)
        # OpenCV defaults to BGR format, so convert to RGB
        image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

        return image_np

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
        person_result, original_img = self.yolov8_detector.detect_person(self.img_path)
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
        face_results = self.face_detector.process_image(self.img_path)
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
        self.image = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), cv.IMREAD_COLOR)
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
