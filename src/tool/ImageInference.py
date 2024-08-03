import cv2
import numpy as np
import onnxruntime as ort


class ImageInference:
    def __init__(self, model_path, bgr_list=None):
        """
        Initialize the image inference object.

        :param model_path: Path to the ONNX model
        :param bgr_list: List of BGR channel values for image composition
        """
        if bgr_list is None:
            bgr_list = [1.0, 0.0, 0.0]
        self.model_path = model_path
        try:
            self.sess = ort.InferenceSession(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {model_path}: {e}")
        self.rec = [np.zeros((1, 1, 1, 1), dtype=np.float32)] * 4
        self.downsample_ratio = np.array([0.25], dtype=np.float32)  # Ensure FP32
        self.bgr = np.array(bgr_list, dtype=np.float32).reshape((3, 1, 1))  # BGR

    def update_bgr(self, bgr_list):
        """
        Update BGR channel values.

        :param bgr_list: New list of BGR channel values
        """
        if bgr_list is None or not isinstance(bgr_list, (list, tuple)) or len(bgr_list) != 3:
            raise ValueError("bgr_list must be a list or tuple containing 3 elements.")
        self.bgr = np.array(bgr_list, dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """
        Normalize the image.

        :param frame: Input image (H, W) or (H, W, C)
        :return: Normalized image (B=1, C, H, W)
        :rtype: np.ndarray
        """
        if frame is None:
            raise ValueError("Input cannot be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if frame.ndim == 2:
            # If the image is grayscale, expand to 3 channels
            img = np.expand_dims(frame, axis=-1)
            img = np.repeat(img, 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            img = frame
        else:
            raise ValueError("Input shape must be (H, W) or (H, W, 3).")

        img = img.astype(np.float32) / 255.0
        img = img[:, :, ::-1]  # Convert from RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # Transpose to (C, H, W)
        img = np.expand_dims(img, axis=0)  # Expand to (B=1, C, H, W)
        return img

    def infer_rvm_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference on the image using the RVM model.

        :param image: Input image (H, W) or (H, W, C)
        :return: Inferred image (H, W, 3)
        :rtype: np.ndarray
        """
        src = self.normalize(image)

        # Perform inference
        try:
            fgr, pha, *self.rec = self.sess.run(None, {
                "src": src,
                "r1i": self.rec[0],
                "r2i": self.rec[1],
                "r3i": self.rec[2],
                "r4i": self.rec[3],
                "downsample_ratio": self.downsample_ratio
            })
        except Exception as e:
            raise RuntimeError(f"ONNX model inference failed: {e}")

        # Compose image
        merge_frame = fgr * pha + self.bgr * (1. - pha)  # (1, 3, H, W)
        merge_frame = merge_frame[0] * 255.0  # (3, H, W)
        merge_frame = merge_frame.astype(np.uint8)  # Convert to uint8
        merge_frame = np.transpose(merge_frame, (1, 2, 0))  # Transpose to (H, W, 3)
        merge_frame = cv2.cvtColor(merge_frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return merge_frame
