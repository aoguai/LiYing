import onnxruntime as ort
import numpy as np
from PIL import Image


def bgr_to_rgba(bgr):
    if not isinstance(bgr, (list, tuple)) or len(bgr) != 3:
        raise ValueError("bgr must be a list or tuple containing three elements")
    if any(not (0 <= color <= 1) for color in bgr):
        raise ValueError("bgr values must be in the range [0, 1]")

    blue, green, red = bgr

    # Normalize to the 255 range
    red = int(red * 255)
    green = int(green * 255)
    blue = int(blue * 255)

    # Add Alpha channel (fully opaque)
    alpha = 255

    # Return RGBA values
    return red, green, blue, alpha


class ImageSegmentation:
    def __init__(self, model_path: str, model_input_size: list, bgr_list: list):
        if not isinstance(model_path, str) or not model_path.endswith('.onnx'):
            raise ValueError("model_path must be a valid ONNX model file path")
        if not isinstance(model_input_size, list) or len(model_input_size) != 2:
            raise ValueError("model_input_size must be a list with two elements")
        if any(not isinstance(size, int) or size <= 0 for size in model_input_size):
            raise ValueError("model_input_size elements must be positive integers")

        # Initialize model path and input size
        self.model_path = model_path
        self.model_input_size = model_input_size
        try:
            self.ort_session = ort.InferenceSession(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        self.bgr_list = bgr_to_rgba(bgr_list)

    def preprocess_image(self, im: np.ndarray) -> np.ndarray:
        # If the image is grayscale, add a dimension to make it a color image
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # Resize the image to match the model input size
        try:
            im_resized = np.array(Image.fromarray(im).resize(self.model_input_size, Image.BILINEAR))
        except Exception as e:
            raise RuntimeError(f"Error resizing image: {e}")
        # Normalize image pixel values to the [0, 1] range
        image = im_resized.astype(np.float32) / 255.0
        # Further normalize image data
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        image = (image - mean) / std
        # Convert the image to the required shape
        image = image.transpose(2, 0, 1)  # Change dimension order (channels, height, width)
        return np.expand_dims(image, axis=0)  # Add batch dimension

    def postprocess_image(self, result: np.ndarray, im_size: list) -> np.ndarray:
        # Resize the result image to match the original image size
        result = np.squeeze(result)
        try:
            result = np.array(Image.fromarray(result).resize(im_size, Image.BILINEAR))
        except Exception as e:
            raise RuntimeError(f"Error resizing result image: {e}")
        # Normalize the result image data
        ma = result.max()
        mi = result.min()
        result = (result - mi) / (ma - mi)
        # Convert to uint8 image
        im_array = (result * 255).astype(np.uint8)
        return im_array

    def infer(self, image: np.ndarray) -> np.ndarray:
        # Prepare the input image
        orig_im_size = image.shape[0:2]
        image_preprocessed = self.preprocess_image(image)

        # Perform inference (image segmentation)
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_preprocessed}
        try:
            ort_outs = self.ort_session.run(None, ort_inputs)
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")
        result = ort_outs[0]

        # Post-process the result image
        result_image = self.postprocess_image(result, orig_im_size)

        # Save the result image
        try:
            pil_im = Image.fromarray(result_image).convert("L")
            orig_image = Image.fromarray(image).convert("RGBA")
            pil_im = pil_im.resize(orig_image.size)
        except Exception as e:
            raise RuntimeError(f"Error processing images: {e}")
        no_bg_image = Image.new("RGBA", orig_image.size, self.bgr_list)
        no_bg_image.paste(orig_image, mask=pil_im)

        # Convert to OpenCV image
        no_bg_image_cv = np.array(no_bg_image)
        return no_bg_image_cv
