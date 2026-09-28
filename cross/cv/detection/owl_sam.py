import torch

import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from cross.utils.profile import timeblock
from .utils import show_masks


class OwlSam:
    default_config = {
        "owl2_model": "google/owlv2-base-patch16-ensemble",
        "sam2_model": "facebook/sam2-hiera-tiny", # "facebook/sam2-hiera-base-plus", | "facebook/sam2-hiera-large",
        "detection_threshold": 0.1,
        "mask_threshold": 0.9,
    }
    def __init__(self, 
                 device='cuda:0', 
                 config=None,
                 ):

        self.default_config.update(config or {})
        self.device = device

        self.processor = Owlv2Processor.from_pretrained(self.default_config["owl2_model"], use_fast=True)
        self.owl = Owlv2ForObjectDetection.from_pretrained(self.default_config["owl2_model"])
        self.sam = SAM2ImagePredictor.from_pretrained(self.default_config["sam2_model"],device=self.device)

        self.owl.to(self.device)
        self.owl.eval()

    def predict(self, image, texts):
        """Detect objects in an image and return the detected objects with their confidence and location.
        args:
            image: PIL.Image.Image, or list of PIL.Image.Image
            texts: list of str, or list of list of str
        returns:
            boxes: np.ndarray, shape (N, 4), where N is the number of detected objects, and each row is (x0, y0, x1, y1)
            masks: np.ndarray, shape (N, H, W), where H, W is the height and width of the image
            scores: np.ndarray, shape (N,), the confidence of the detected objects
            labels: np.ndarray, shape (N,), the class of the detected objects
        """
        batch = isinstance(image, list)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if batch:
                self.sam.set_image_batch(image)
                target_sizes = torch.Tensor([img.size[::-1] for img in image])
            else:
                self.sam.set_image(image)
                target_sizes = torch.Tensor([image.size[::-1]])

            with timeblock("detection"):
                inputs = self.processor(text=texts, images=image, return_tensors="pt")
                
                outputs = self.owl(input_ids=inputs["input_ids"].to(self.device), 
                                pixel_values=inputs["pixel_values"].to(self.device),
                                attention_mask=inputs["attention_mask"].to(self.device))
                results = self.processor.post_process_grounded_object_detection(outputs=outputs, 
                                                                    target_sizes=target_sizes, 
                                                                    threshold=self.default_config["detection_threshold"])
            if batch:
                # TODO: batch sam
                raise NotImplementedError("Batch sam is not implemented")
                # boxes = [result["boxes"].cpu().numpy() for result in results]
                # masks, sam_scores, logits = self.sam.predict_batch(
                #     boxes=boxes,
                #     multimask_output=False,
                # )
                
            else:
                boxes = results[0]["boxes"].cpu()
                det_scores = results[0]["scores"].cpu()
                labels = results[0]["labels"].cpu()
                if len(boxes) == 0:
                    return {
                        "boxes":[],
                        "masks":[],
                        "scores":[],
                        "labels":[],
                    }
                with timeblock("sam"):
                    masks, sam_scores, logits = self.sam.predict(box=boxes,
                                                        multimask_output=False,)
                # select the top 1 mask for each box
                if len(masks) > 1:
                    top_mask_id = torch.argmax(torch.tensor(sam_scores), dim=1)
                    masks = torch.tensor(masks)[torch.arange(len(masks)), top_mask_id]
                    scores = det_scores * sam_scores.max(axis=1)
                else:
                    return {
                        "boxes":[],
                        "masks":[],
                        "scores":[],
                        "labels":[],
                    }

                # filter out low confidence detections

                mask_filter = sam_scores.max(axis=1) > self.default_config["mask_threshold"]
                boxes = boxes[mask_filter]
                masks = masks[mask_filter]
                scores = scores[mask_filter]
                labels = labels[mask_filter]

                res = {
                    "boxes": boxes,
                    "masks": masks,
                    "scores": scores,
                    "labels": labels,
                }
                return res

    def test_predict(self, image, texts):
        """Test the predict function."""
        res = self.predict(image, texts)
        self.visualize_results(image, res)

    @staticmethod
    def visualize_results(image, res, label_list=None):
        """Visualize the results."""
        if label_list is not None:
            idx_mask = [i for i, label in enumerate(res["labels"].numpy()) if label in label_list]
            show_masks(image, res["masks"].numpy()[idx_mask], res["boxes"].numpy()[idx_mask])
        else:
            show_masks(image, res["masks"].numpy(), res["boxes"].numpy())
