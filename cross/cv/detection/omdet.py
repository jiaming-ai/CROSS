import requests
from PIL import Image

from transformers import AutoProcessor, OmDetTurboForObjectDetection
import torch
import matplotlib.pyplot as plt
from .utils import show_box, show_masks
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cross.utils.profile import timeblock

class OmDet:
    default_config = {
        "model": "omlab/omdet-turbo-swin-tiny-hf",
        "sam2_model": "facebook/sam2-hiera-tiny",
        "detection_threshold": 0.3,
        "nms_threshold": 0.3,
        "mask_threshold": 0.9,
    }

    def __init__(self, device='cuda:0', config=None):
        self.default_config.update(config or {})
        self.device = device
        
        self.processor = AutoProcessor.from_pretrained(self.default_config["model"])
        self.model = OmDetTurboForObjectDetection.from_pretrained(self.default_config["model"])
        self.model.to(self.device)
        self.model.eval()
        self.sam = SAM2ImagePredictor.from_pretrained(self.default_config["sam2_model"], device=self.device)

    def predict(self, image, texts):
        """Detect objects in an image and return the detected objects with their confidence and location.
        args:
            image: PIL.Image.Image
            texts: list of str
        returns:
            dict containing:
                boxes: tensor shape (N, 4)
                scores: tensor shape (N,)
                labels: tensor shape (N,)
                masks: tensor shape (N, H, W)
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam.set_image(image)

            with timeblock("detection"):
                inputs = self.processor(image, text=texts, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                classes=texts,
                target_sizes=[image.size[::-1]],
                score_threshold=self.default_config["detection_threshold"],
                nms_threshold=self.default_config["nms_threshold"],
            )[0]

            if len(results["boxes"]) == 0:
                return {
                    "boxes": [],
                    "masks": [],
                    "scores": [],
                    "labels": [],
                }

            with timeblock("sam"):
                masks, sam_scores, logits = self.sam.predict(
                    box=results["boxes"],
                    multimask_output=False,
                )

            # Select top mask for each box and filter by confidence
            if len(masks) > 1:
                top_mask_id = torch.argmax(torch.tensor(sam_scores), dim=1)
                masks = torch.tensor(masks)[torch.arange(len(masks)), top_mask_id]
                scores = results["scores"].cpu().float() * sam_scores.max(axis=1)
            else:
                return {
                    "boxes": [],
                    "masks": [],
                    "scores": [],
                    "labels": [],
                }

            # Filter by mask confidence
            mask_filter = sam_scores.max(axis=1) > self.default_config["mask_threshold"]
            boxes = results["boxes"][mask_filter]
            masks = masks[mask_filter]
            scores = scores[mask_filter]
            label_indices = torch.tensor([texts.index(class_name) for class_name in results["classes"]])[mask_filter]

            return {
                "boxes": boxes,
                "masks": masks,
                "scores": scores,
                "labels": label_indices,
            }

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
        
         
# processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
# model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# classes = ["cat", "remote"]
# inputs = processor(image, text=classes, return_tensors="pt")

# outputs = model(**inputs)

# # convert outputs (bounding boxes and class logits)
# results = processor.post_process_grounded_object_detection(
#     outputs,
#     classes=classes,
#     target_sizes=[image.size[::-1]],
#     score_threshold=0.3,
#     nms_threshold=0.3,
# )[0]
# for score, class_name, box in zip(
#     results["scores"], results["classes"], results["boxes"]
# ):
#     box = [round(i, 1) for i in box.tolist()]
#     print(
#         f"Detected {class_name} with confidence "
#         f"{round(score.item(), 2)} at location {box}"
#     )