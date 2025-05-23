import os
import warnings
from io import BytesIO

import cv2 as cv
import numpy as np
import piexif
from PIL import Image

from ImageSegmentation import ImageSegmentation
from PhotoEntity import PhotoEntity
from PhotoRequirements import PhotoRequirements
from agpic import ImageCompressor


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
                if int(y1) - int(self.photo.face_bbox[1]) < max(int(height / 600 * 60), 60):
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
            warnings.warn("No human face detected. Falling back to general object detection.", UserWarning)
            # No human subject detected, use YOLOv8 for basic object detection
            yolo_result, _ = self.photo.yolov8_detector.detect(self.photo.img_path)
            if yolo_result and yolo_result['boxes']:
                warnings.warn("Object detected. Using the first detected object for processing.", UserWarning)
                # Use the first detected object's bounding box
                bbox = yolo_result['boxes'][0]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Adjust crop area to include some margin
                height, width = self.photo.image.shape[:2]
                margin = min(height, width) // 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)
                
                # Crop the image
                cropped_image = self.photo.image[y1:y2, x1:x2]
                
                # Update the PhotoEntity object's image and re-detect
                self.photo.image = cropped_image
                self.photo.detect()
                
                # Set the bounding box to the full image range
                self.photo.person_bbox = [0, 0, cropped_image.shape[1], cropped_image.shape[0]]
            else:
                warnings.warn("No object detected. Using the entire image.", UserWarning)
                # If no object is detected, use the entire image
                self.photo.person_bbox = [0, 0, self.photo.image.shape[1], self.photo.image.shape[0]]
        
        return self.photo

    def change_background(self, rgb_list=None) -> PhotoEntity:
        """
        Replace the background of the human region in the image

        :param rgb_list: New list of RGB channel values
        :return: Updated PhotoEntity instance
        :rtype: PhotoEntity
        """
        if rgb_list is not None:
            if not (isinstance(rgb_list, (list, tuple)) and len(rgb_list) == 3):
                raise ValueError("The RGB value format is incorrect")
            self.segmentation.rgb_list = tuple(rgb_list)

        self.photo.image = self.segmentation.infer(self.photo.image)
        return self.photo

    def resize_image(self, photo_type):
        # Get the target dimensions and other info
        photo_info = self.photo_requirements_detector.get_resize_image_list(photo_type)
        width, height = photo_info['width'], photo_info['height']
        
        # Get the original image dimensions
        orig_height, orig_width = self.photo.image.shape[:2]
        
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
        cropped_image = self.photo.image[y1:y2, x1:x2]
        
        # Update the PhotoEntity object's image
        self.photo.image = cropped_image
        
        # Resize the image proportionally
        self.photo.image = cv.resize(self.photo.image, (width, height), interpolation=cv.INTER_AREA)
        
        # Store the actual print size and resolution in the PhotoEntity
        self.photo.print_size = photo_info['print_size']
        self.photo.resolution = photo_info['resolution']
        
        return self.photo.image

    def save_photos(self, save_path: str, y_b=False, target_size=None, size_range=None) -> None:
        """
        Save the image to the specified path.
        :param save_path: The path to save the image
        :param y_b: Whether to compress the image
        :param target_size: Target file size in KB. When specified, ignores quality.
        :param size_range: A tuple of (min_size, max_size) in KB for the output file.
        """
        # Parameter validation for target_size and size_range
        if target_size is not None and size_range is not None:
            warnings.warn("Both target_size and size_range provided. Using target_size and ignoring size_range.", 
                           UserWarning)
            size_range = None
            
        if target_size is not None and target_size <= 0:
            raise ValueError(f"Target size must be greater than 0, got {target_size}")
            
        if size_range is not None:
            if len(size_range) != 2:
                raise ValueError(f"Size range must be a tuple of (min_size, max_size), got {size_range}")
            min_size, max_size = size_range
            if min_size <= 0 or max_size <= 0:
                raise ValueError(f"Size range values must be greater than 0, got min_size={min_size}, max_size={max_size}")
            if min_size >= max_size:
                raise ValueError(f"Minimum size must be less than maximum size, got min_size={min_size}, max_size={max_size}")

        # Check the path length
        max_path_length = 200
        if len(save_path) > max_path_length:
            # Intercepts the filename and keeps the rest of the path
            dir_name = os.path.dirname(save_path)
            base_name = os.path.basename(save_path)
            ext = os.path.splitext(base_name)[1]
            base_name = base_name[:200] + ext  # Ensure that filenames do not exceed 200 characters
            save_path = os.path.join(dir_name, base_name)

        # Get the DPI from the photo entity
        if isinstance(self.photo.resolution, int):
            dpi = self.photo.resolution
        elif isinstance(self.photo.resolution, str):
            dpi = int(self.photo.resolution.replace('dpi', ''))
        else:
            dpi = 300  # Default DPI if resolution is not set

        # Check if we need to compress (either y_b flag is True or size parameters are provided)
        need_compression = y_b or target_size is not None or size_range is not None
        
        if need_compression:
            buffer = BytesIO()
            pil_image = Image.fromarray(cv.cvtColor(self.photo.image, cv.COLOR_BGR2RGB))
            pil_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            try:
                if target_size is not None:
                    compressed_bytes = ImageCompressor.compress_image_from_bytes(
                        image_bytes, quality=85, target_size=target_size
                    )
                elif size_range is not None:
                    compressed_bytes = ImageCompressor.compress_image_from_bytes(
                        image_bytes, quality=85, size_range=size_range
                    )
                else:
                    compressed_bytes = ImageCompressor.compress_image_from_bytes(
                        image_bytes, quality=85
                    )

                # Write compressed bytes directly to the file
                with open(save_path, 'wb') as f:
                    f.write(compressed_bytes)
                
                # Setting up DPI using piexif
                try:
                    # Converting DPI to EXIF resolution format
                    x_resolution = (dpi, 1)  # DPI value and unit
                    y_resolution = (dpi, 1)
                    resolution_unit = 2  # inches
                    
                    # Read existing EXIF data (if present)
                    exif_dict = piexif.load(save_path)
                    
                    # If '0th' does not exist, create it
                    if '0th' not in exif_dict:
                        exif_dict['0th'] = {}
                    
                    # Set resolution information
                    exif_dict['0th'][piexif.ImageIFD.XResolution] = x_resolution
                    exif_dict['0th'][piexif.ImageIFD.YResolution] = y_resolution
                    exif_dict['0th'][piexif.ImageIFD.ResolutionUnit] = resolution_unit
                    
                    # Write EXIF data back to file
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, save_path)
                except Exception as e:
                    warnings.warn(f"Failed to set DPI with piexif: {str(e)}. Image saved without DPI metadata.", UserWarning)
                
            except Exception as e:
                warnings.warn(f"Image compression failed: {str(e)}. Saving uncompressed image.", UserWarning)
                _, img_bytes = cv.imencode('.jpg', self.photo.image, [cv.IMWRITE_JPEG_QUALITY, 95])
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)

                try:
                    x_resolution = (dpi, 1)
                    y_resolution = (dpi, 1)
                    resolution_unit = 2

                    exif_dict = piexif.load(save_path)

                    if '0th' not in exif_dict:
                        exif_dict['0th'] = {}

                    exif_dict['0th'][piexif.ImageIFD.XResolution] = x_resolution
                    exif_dict['0th'][piexif.ImageIFD.YResolution] = y_resolution
                    exif_dict['0th'][piexif.ImageIFD.ResolutionUnit] = resolution_unit

                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, save_path)
                except Exception as ex:
                    warnings.warn(f"Failed to set DPI with piexif: {str(ex)}. Image saved without DPI metadata.", UserWarning)
        else:
            _, img_bytes = cv.imencode('.jpg', self.photo.image, [cv.IMWRITE_JPEG_QUALITY, 95])
            with open(save_path, 'wb') as f:
                f.write(img_bytes)

            try:
                x_resolution = (dpi, 1)
                y_resolution = (dpi, 1)
                resolution_unit = 2

                exif_dict = piexif.load(save_path)

                if '0th' not in exif_dict:
                    exif_dict['0th'] = {}

                exif_dict['0th'][piexif.ImageIFD.XResolution] = x_resolution
                exif_dict['0th'][piexif.ImageIFD.YResolution] = y_resolution
                exif_dict['0th'][piexif.ImageIFD.ResolutionUnit] = resolution_unit

                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, save_path)
            except Exception as ex:
                warnings.warn(f"Failed to set DPI with piexif: {str(ex)}. Image saved without DPI metadata.", UserWarning)

