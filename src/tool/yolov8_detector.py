import logging
import os

import cv2 as cv
import numpy as np
from .deviceUtils import get_onnx_session


class YOLOv8Detector:
    def __init__(self, model_path, input_size=(640, 640), box_score=0.25, kpt_score=0.5, nms_thr=0.2):
        """
        Initialize the YOLOv8 detector

        Parameters:
        model_path (str): Path to the model file, can be an absolute or relative path.
        input_size (tuple): Input image size.
        box_score (float): Confidence threshold for detection boxes.
        kpt_score (float): Confidence threshold for keypoints.
        nms_thr (float): Non-Maximum Suppression (NMS) threshold.
        """
        assert model_path.endswith('.onnx'), f"invalid onnx model: {model_path}"
        assert os.path.exists(model_path), f"model not found: {model_path}"

        # Set log level to ERROR to disable default console info output
        logging.getLogger('ultralytics').setLevel(logging.ERROR)

        # Create ONNX Runtime session
        self.session = get_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.box_score = box_score
        self.kpt_score = kpt_score
        self.nms_thr = nms_thr

    def preprocess(self, img_path):
        # Read the image
        img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            raise ValueError(f"Failed to read image from {img_path}")

        input_w, input_h = self.input_size
        padded_img = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
        r = min(input_w / img.shape[1], input_h / img.shape[0])
        resized_img = cv.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                interpolation=cv.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
        return padded_img, r, img

    def postprocess(self, output, ratio):
        predict = output[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score, :]
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / ratio
        boxes = self.xywh2xyxy(boxes)
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3 * j + 2] < self.kpt_score:
                    kpts[i, 3 * j: 3 * (j + 1)] = [-1, -1, -1]
                else:
                    kpts[i, 3 * j] /= ratio
                    kpts[i, 3 * j + 1] /= ratio
        idxes = self.nms_process(boxes, scores)
        result = {'boxes': boxes[idxes, :].astype(int).tolist(), 'kpts': kpts[idxes, :].astype(float).tolist(),
                  'scores': scores[idxes].tolist()}
        return result

    def xywh2xyxy(self, box):
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def nms_process(self, boxes, scores):
        sorted_idx = np.argsort(scores)[::-1]
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
            rest_idx = np.where(ious < self.nms_thr)[0]
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area

    def detect(self, img_path):
        """
        Detect objects in an image

        Parameters:
        img_path (str): Path to the image file, can be an absolute or relative path.

        Returns:
        results: Detection results.
        """
        image, ratio, original_img = self.preprocess(img_path)
        ort_input = {self.input_name: image[None, :]}
        output = self.session.run(None, ort_input)
        result = self.postprocess(output, ratio)
        return result, original_img

    def detect_person(self, img_path):
        """
        Detect if there is only one person in the image

        Parameters:
        img_path (str): Path to the image file, can be an absolute or relative path.

        Returns:
        dict: Contains the coordinates of the box, predicted class, coordinates of all keypoints, and confidence scores.
              If more or fewer than one person is detected, returns None.
        """
        result, original_img = self.detect(img_path)
        boxes = result['boxes']
        # scores = result['scores']
        kpts = result['kpts']

        # Only handle cases where exactly one person is detected
        if len(boxes) == 1:
            bbox_xyxy = boxes[0]
            bbox_label = 0  # Assuming person class is 0
            bbox_keypoints = kpts[0]
            return {
                'bbox_xyxy': bbox_xyxy,
                'bbox_label': bbox_label,
                'bbox_keypoints': bbox_keypoints
            }, original_img
        return None, original_img

    def draw_result(self, img, result, with_label=False):
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        for box, kpt, score in zip(boxes, kpts, scores):
            x1, y1, x2, y2 = box
            label_str = "{:.0f}%".format(score * 100)
            label_size, baseline = cv.getTextSize(label_str, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if with_label:
                cv.rectangle(img, (x1, y1), (x1 + label_size[0], y1 + label_size[1] + baseline), (0, 0, 255), -1)
                cv.putText(img, label_str, (x1, y1 + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3 * idx: 3 * (idx + 1)]
                if score > 0:
                    cv.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        return img
