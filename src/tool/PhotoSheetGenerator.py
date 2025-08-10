import cv2
from PIL import Image, ImageDraw
import numpy as np

class PhotoSheetGenerator:
    def __init__(self, five_inch_size=(1050, 1500), dpi=300):
        self.five_inch_size = five_inch_size
        self.dpi = dpi

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

    def generate_photo_sheet(self, one_inch_photo_cv2, rows=3, cols=3, rotate=False, add_crop_lines=True, layout_position=4, photos_spacing=0):
        """
        Generate a photo sheet with the specified layout.
        
        :param one_inch_photo_cv2: Input photo in OpenCV format
        :param rows: Number of rows in the layout
        :param cols: Number of columns in the layout
        :param rotate: Whether to rotate the photo 90 degrees
        :param add_crop_lines: Whether to add crop lines
        :param layout_position: Position of the layout (0-8):
                              0: Top-Left, 1: Top, 2: Top-Right
                              3: Middle-Left, 4: Center, 5: Middle-Right
                              6: Bottom-Left, 7: Bottom, 8: Bottom-Right
        :param photos_spacing: Pixel spacing between photos (applies to all sides)
        :return: Generated photo sheet in OpenCV format
        """
        one_inch_height, one_inch_width = one_inch_photo_cv2.shape[:2]

        # Convert OpenCV image data to Pillow image
        one_inch_photo_pillow = self.cv2_to_pillow(one_inch_photo_cv2)

        # Rotate photo
        if rotate:
            one_inch_photo_pillow = one_inch_photo_pillow.rotate(90, expand=True)
            one_inch_height, one_inch_width = one_inch_width, one_inch_height

        # Create photo sheet (white background)
        five_inch_photo = Image.new('RGB', self.five_inch_size, 'white')

        # Calculate content dimensions (photos keep original sizes) and total including spacing
        content_width = cols * one_inch_width
        content_height = rows * one_inch_height
        total_width = content_width + (cols - 1) * photos_spacing if cols > 1 else content_width
        total_height = content_height + (rows - 1) * photos_spacing if rows > 1 else content_height

        if total_width > self.five_inch_size[0] or total_height > self.five_inch_size[1]:
            raise ValueError("The specified layout exceeds the size of the photo sheet")

        # Calculate start positions based on layout_position
        if layout_position < 0 or layout_position > 8:
            layout_position = 4  # Default to center if invalid position

        # Calculate horizontal position
        if layout_position in [0, 3, 6]:  # Left
            start_x = 0
        elif layout_position in [2, 5, 8]:  # Right
            start_x = self.five_inch_size[0] - total_width
        else:  # Center
            start_x = (self.five_inch_size[0] - total_width) // 2

        # Calculate vertical position
        if layout_position in [0, 1, 2]:  # Top
            start_y = 0
        elif layout_position in [6, 7, 8]:  # Bottom
            start_y = self.five_inch_size[1] - total_height
        else:  # Middle
            start_y = (self.five_inch_size[1] - total_height) // 2

        # Arrange photos on the sheet in an n*m layout
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * (one_inch_width + photos_spacing)
                y = start_y + i * (one_inch_height + photos_spacing)
                five_inch_photo.paste(one_inch_photo_pillow, (x, y))

        # Draw crop lines if requested
        if add_crop_lines:
            draw = ImageDraw.Draw(five_inch_photo)
            
            # Draw crop rectangles around EACH photo
            for i in range(rows):
                for j in range(cols):
                    x = start_x + j * (one_inch_width + photos_spacing)
                    y = start_y + i * (one_inch_height + photos_spacing)
                    draw.rectangle([x, y, x + one_inch_width, y + one_inch_height], outline="black")
            # Necessary auxiliary lines, used to facilitate alignment during cutting
            draw.rectangle([start_x, start_y, self.five_inch_size[0], self.five_inch_size[1]], outline="black")

        # Set the DPI information
        five_inch_photo.info['dpi'] = (self.dpi, self.dpi)

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
            photo_sheet.save(output_path, dpi=(self.dpi, self.dpi))
        except Exception as e:
            raise IOError(f"Failed to save photo: {e}")