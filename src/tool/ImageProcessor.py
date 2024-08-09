import os

import cv2 as cv
import numpy as np

from ImageSegmentation import ImageSegmentation
from PhotoEntity import PhotoEntity
from PhotoRequirements import PhotoRequirements


def get_model_file(filename):
    return os.path.join('model', filename)


class ImageProcessor:
    """
    Image processing class for cropping and correcting the human region in images.
    """

    def __init__(self, img_path,
                 yolov8_model_path=get_model_file('yolov8n-pose.onnx'),
                 yunet_model_path=get_model_file('face_detection_yunet_2023mar.onnx'),
                 RMBG_model_path=get_model_file('RMBG-1.4-model.onnx'),
                 rgb_list=None,
                 y_b=False):
        """
        Initialize ImageProcessor instance

        :param img_path: Path to the image
        :param yolov8_model_path: Path to the YOLOv8 model
        :param yunet_model_path: Path to the YuNet model
        :param RMBG_model_path: Path to the RMBG model
        :param rgb_list: List of rgb channel values for image composition
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        if not os.path.exists(yolov8_model_path):
            raise FileNotFoundError(f"YOLOv8 model path does not exist: {yolov8_model_path}")
        if not os.path.exists(yunet_model_path):
            raise FileNotFoundError(f"YuNet model path does not exist: {yunet_model_path}")
        if not os.path.exists(RMBG_model_path):
            raise FileNotFoundError(f"RMBG model path does not exist: {RMBG_model_path}")

        self.photo = PhotoEntity(img_path, yolov8_model_path, yunet_model_path, y_b)
        self.segmentation = ImageSegmentation(model_path=RMBG_model_path, model_input_size=[1024, 1024],
                                              rgb_list=rgb_list if rgb_list is not None else [255, 255, 255])
        self.photo_requirements_detector = PhotoRequirements()

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the image

        :param image: Original image (numpy.ndarray)
        :param angle: Rotation angle (degrees)
        :return: Rotated image (numpy.ndarray)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be of type numpy.ndarray")
        if not isinstance(angle, (int, float)):
            raise TypeError("The rotation angle must be of type int or float")

        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv.warpAffine(image, matrix, (width, height), flags=cv.INTER_CUBIC)
        return rotated_image

    @staticmethod
    def compute_rotation_angle(left_shoulder: tuple, right_shoulder: tuple, image_shape: tuple) -> float:
        """
        Compute the rotation angle to align the shoulders horizontally

        :param left_shoulder: Coordinates of the left shoulder keypoint (normalized or pixel coordinates)
        :param right_shoulder: Coordinates of the right shoulder keypoint (normalized or pixel coordinates)
        :param image_shape: Height and width of the image
        :return: Rotation angle (degrees)
        :rtype: float
        """
        if not (isinstance(left_shoulder, tuple) and len(left_shoulder) == 3):
            raise ValueError("The left shoulder keypoint format is incorrect")
        if not (isinstance(right_shoulder, tuple) and len(right_shoulder) == 3):
            raise ValueError("The right shoulder keypoint format is incorrect")
        if not (isinstance(image_shape, tuple) and len(image_shape) == 2):
            raise ValueError("The image size format is incorrect")

        height, width = image_shape

        # If coordinates are normalized, convert to pixel coordinates
        if left_shoulder[2] < 1.0 and right_shoulder[2] < 1.0:
            left_shoulder = (left_shoulder[0] * width, left_shoulder[1] * height)
            right_shoulder = (right_shoulder[0] * width, right_shoulder[1] * height)

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        angle = np.arctan2(dy, dx) * (180 / np.pi)  # Compute the angle
        return angle

    def crop_and_correct_image(self) -> PhotoEntity:
        """
        Crop and correct the human region in the image

        :return: Updated PhotoEntity instance
        :rtype: PhotoEntity
        :raises ValueError: If no single person is detected
        """
        if self.photo.person_bbox is not None:
            height, width = self.photo.image.shape[:2]

            # Get bounding box coordinates and keypoints
            bbox_xyxy = self.photo.person_bbox
            x1, y1, x2, y2 = bbox_xyxy
            bbox_keypoints = self.photo.person_keypoints
            bbox_height = y2 - y1

            # Get shoulder keypoints
            left_shoulder = (bbox_keypoints[18], bbox_keypoints[19],
                              bbox_keypoints[20]) # bbox_keypoints[5] right shoulder
            right_shoulder = (bbox_keypoints[15], bbox_keypoints[16], bbox_keypoints[17])   # bbox_keypoints[6] left shoulder
            # print(left_shoulder, right_shoulder)

            # Compute rotation angle
            angle = self.compute_rotation_angle(left_shoulder, right_shoulder, (height, width))

            # Rotate the image
            rotated_image = self.rotate_image(self.photo.image, angle) if abs(angle) > 5 else self.photo.image

            # Recalculate crop box position in the rotated image
            height, width = rotated_image.shape[:2]
            x1, y1, x2, y2 = int(x1 * width / width), int(y1 * height / height), int(x2 * width / width), int(
                y2 * height / height)

            # Adjust crop area to ensure the top does not exceed the image range
            top_margin = bbox_height / 5
            y1 = max(int(y1), 0) if y1 >= top_margin else 0

            # If y1 is less than 60 pixels from the top of the face detection box, adjust it
            if y1 != 0 and self.photo.face_bbox is not None:
                if y1 - self.photo.face_bbox[1] < max(int(height / 600 * 60), 60):
                    y1 = max(int(y1 - (int(height / 600 * 60))), 0)

            # Adjust the crop area to ensure the lower body is not too long
            shoulder_margin = y1 + bbox_height / max(int(height / 600 * 16), 16)
            y2 = min(y2, height - int(shoulder_margin)) if left_shoulder[1] > shoulder_margin or right_shoulder[
                1] > shoulder_margin else y2

            # Adjust the crop area to ensure the face is centered in the image
            left_eye = [bbox_keypoints[6], bbox_keypoints[7], bbox_keypoints[8]]  # bbox_keypoints[2]
            right_eye = [bbox_keypoints[3], bbox_keypoints[4], bbox_keypoints[5]]  # bbox_keypoints[1]
            # print(left_eye, right_eye)
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            crop_width = x2 - x1

            x1 = max(int(face_center_x - crop_width / 2), 0)
            x2 = min(int(face_center_x + crop_width / 2), width)

            # Ensure the crop area does not exceed the image range
            x1 = 0 if x1 < 0 else x1
            x2 = width if x2 > width else x2

            # print(x1,x2,y1,y2)

            # Crop the image
            cropped_image = rotated_image[y1:y2, x1:x2]

            # Update the PhotoEntity object's image and re-detect
            self.photo.image = cropped_image
            self.photo.detect()
            # Manually set the person bounding box to the full image range
            self.photo.person_bbox = [0, 0, cropped_image.shape[1], cropped_image.shape[0]]
            return self.photo
        else:
            raise ValueError('No single person detected.')

    def change_background(self, rgb_list=None) -> PhotoEntity:
        """
        Replace the background of the human region in the image

        :param rgb_list: New list of RGB channel values
        :return: Updated PhotoEntity instance
        :rtype: PhotoEntity
        """
        if rgb_list is not None:
            if not (isinstance(rgb_list, list) and len(rgb_list) == 3):
                raise ValueError("The RGB value format is incorrect")
            self.segmentation.rgb_list = rgb_list

        self.photo.image = self.segmentation.infer(self.photo.image)
        return self.photo

    def resize_image(self, photo_type):
        """
        Resize the image proportionally according to the specified photo type.

        :param photo_type: The type of the photo
        """
        # Get the target dimensions
        width, height, _ = self.photo_requirements_detector.get_resize_image_list(photo_type)
        # print(width, height)

        # Get the original image dimensions
        orig_height, orig_width = self.photo.image.shape[:2]
        # print(orig_width, orig_height)

        # Check if the dimensions are integer multiples
        is_width_multiple = (orig_width % width == 0) if orig_width >= width else (width % orig_width == 0)
        is_height_multiple = (orig_height % height == 0) if orig_height >= height else (height % orig_height == 0)

        if is_width_multiple and is_height_multiple:
            # Resize the image proportionally
            self.photo.image = cv.resize(self.photo.image, (width, height), interpolation=cv.INTER_AREA)
            return self.photo.image

        def get_crop_coordinates(original_size, aspect_ratio):
            original_width, original_height = original_size
            crop_width = original_width
            crop_height = int(crop_width / aspect_ratio)

            if crop_height > original_height:
                crop_height = original_height
                crop_width = int(crop_height * aspect_ratio)

            x_start = (original_width - crop_width) // 2
            y_start = 0

            return x_start, x_start + crop_width, y_start, y_start + crop_height

        x1, x2, y1, y2 = get_crop_coordinates((orig_width, orig_height), width / height)
        # print(x1, x2, y1, y2)

        cropped_image = self.photo.image[y1:y2, x1:x2]

        # Update the PhotoEntity object's image
        self.photo.image = cropped_image

        # Resize the image proportionally
        self.photo.image = cv.resize(self.photo.image, (width, height), interpolation=cv.INTER_AREA)
        return self.photo.image

    def save_photos(self, save_path: str, y_b=False) -> None:
        """
        Save the image to the specified path.

        :param save_path: The path to save the image
        :param y_b: Whether to compress the image
        """
        # Check the path length
        max_path_length = 200
        if len(save_path) > max_path_length:
            # Intercepts the filename and keeps the rest of the path
            dir_name = os.path.dirname(save_path)
            base_name = os.path.basename(save_path)
            ext = os.path.splitext(base_name)[1]
            base_name = base_name[:200] + ext  # Ensure that filenames do not exceed 200 characters
            save_path = os.path.join(dir_name, base_name)

        if y_b:
            ext = os.path.splitext(save_path)[1].lower()
            encode_format = '.jpg' if ext in ['.jpg', '.jpeg'] else '.png' if ext == '.png' else None
            if encode_format is None:
                raise ValueError(f"Unsupported file format: {ext}")

            is_success, buffer = cv.imencode(encode_format, self.photo.image)
            if not is_success:
                raise ValueError("Failed to encode the image to bytes")

            image_bytes = buffer.tobytes()

            compressed_bytes = self.photo.ImageCompressor_detector.compress_image_from_bytes(image_bytes)

            compressed_image = cv.imdecode(np.frombuffer(compressed_bytes, np.uint8), cv.IMREAD_COLOR)
            self.photo.image = compressed_image

        cv.imwrite(save_path, self.photo.image)

