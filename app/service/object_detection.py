from typing import Any, List
import torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2
from pydantic import BaseModel


class Prediction(BaseModel):
    score: float
    mask: Any


class ObjectDetectionService:
    def get_predictions_over_threshold(
        self, image, threshold: float
    ) -> List[Prediction]:
        image_as_tensor = self.convert_image_to_tensor(image)
        model = self.load_mask_RCNN_model()
        predictions = self.execute_detection(model, image_as_tensor)
        predictions_filtered = self.filter_predictions_by_threshold(
            predictions, threshold
        )
        return predictions_filtered

    def get_prediction_histogram(self, image, prediction: Prediction):
        prediction_as_pixel_tensor = self.parse_prediction_from_image(image, prediction)
        hist, bin_edges = np.histogram(
            prediction_as_pixel_tensor,
            bins=256,
            range=(0, 256),
        )
        return hist, bin_edges

    def convert_image_to_tensor(self, image: Image) -> torch.Tensor:
        tensor = F.to_tensor(image).unsqueeze(0)
        return tensor

    def load_mask_RCNN_model(self) -> MaskRCNN:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        model.eval()
        return model

    def execute_detection(self, model, tensor):
        with torch.no_grad():
            predictions = model(tensor)
        return predictions

    def filter_predictions_by_threshold(
        self,
        predictions: List[dict],
        threshold: float,
    ) -> List[Prediction]:
        masks = predictions[0]["masks"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()

        prediction_count = len(masks)

        filtered_predictions = []
        for i in range(prediction_count):
            score = scores[i]
            if score > threshold:
                mask = masks[i][0]
                filtered_predictions.append(
                    Prediction(
                        mask=mask,
                        score=score,
                    )
                )

        return filtered_predictions

    def parse_prediction_from_image(self, image, prediction: Prediction):
        mask = prediction.mask

        mask_clipped = np.clip(mask, 0, 1)
        mask_uint8 = (mask_clipped * 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        image_gray = self.convert_image_to_gray(image)
        prediction_pixels = image_gray[mask_binary != 0]
        return prediction_pixels

    def convert_image_to_gray(self, image):
        image_np = np.array(image)
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        return image_gray
