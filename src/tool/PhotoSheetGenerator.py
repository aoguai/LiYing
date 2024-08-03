import cv2
from PIL import Image, ImageDraw
import numpy as np


class PhotoSheetGenerator:
    def __init__(self, five_inch_size=(1050, 1500)):
        self.five_inch_size = five_inch_size

    @staticmethod
    def cv2_to_pillow(cv2_image):
        """Convert OpenCV image data to Pillow image"""
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image_rgb)

    @staticmethod
    def pillow_to_cv2(pillow_image):
        """Convert Pillow image to OpenCV image data"""
        cv2_image_rgb = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
        return cv2_image_rgb

    def generate_photo_sheet(self, one_inch_photo_cv2, rows=3, cols=3, rotate=False):
        one_inch_height, one_inch_width = one_inch_photo_cv2.shape[:2]

        # Convert OpenCV image data to Pillow image
        one_inch_photo_pillow = self.cv2_to_pillow(one_inch_photo_cv2)

        # Rotate photo
        if rotate:
            one_inch_photo_pillow = one_inch_photo_pillow.rotate(90, expand=True)
            one_inch_height, one_inch_width = one_inch_width, one_inch_height

        # Create photo sheet (white background)
        five_inch_photo = Image.new('RGB', self.five_inch_size, 'white')

        # Create photo with black border
        bordered_one_inch_photo = Image.new('RGB', (one_inch_width + 2, one_inch_height + 2), 'black')
        bordered_one_inch_photo.paste(one_inch_photo_pillow, (1, 1))

        # Calculate positions for the photos on the sheet
        total_width = cols * (one_inch_width + 2)
        total_height = rows * (one_inch_height + 2)

        if total_width > self.five_inch_size[0] or total_height > self.five_inch_size[1]:
            raise ValueError("The specified layout exceeds the size of the photo sheet")

        start_x = (self.five_inch_size[0] - total_width) // 2
        start_y = (self.five_inch_size[1] - total_height) // 2

        # Arrange photos on the sheet in an n*m layout
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * (one_inch_width + 2)
                y = start_y + i * (one_inch_height + 2)
                five_inch_photo.paste(bordered_one_inch_photo, (x, y))

        # Draw alignment lines
        draw = ImageDraw.Draw(five_inch_photo)
        draw.rectangle([start_x, start_y, self.five_inch_size[0], self.five_inch_size[1]], outline="black")

        # Return the generated photo sheet as a Pillow image
        return self.pillow_to_cv2(five_inch_photo)

    def save_photo_sheet(self, photo_sheet_cv, output_path):
        """Save the generated photo sheet as an image file"""
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("output_path must be a valid image file path ending with .png, .jpg, or .jpeg")
        try:
            photo_sheet = self.cv2_to_pillow(photo_sheet_cv)
            photo_sheet.save(output_path)
        except Exception as e:
            raise IOError(f"Failed to save photo: {e}")
