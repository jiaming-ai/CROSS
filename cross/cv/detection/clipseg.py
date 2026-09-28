from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from cross.utils.profile import timeblock
from .utils import show_masks

class ClipSeg:
    default_config = {
        "model": "CIDAS/clipseg-rd64-refined",
        "sam2_model": "facebook/sam2-hiera-tiny",
        "detection_threshold": 0.5,
        "mask_threshold": 0.5,
    }

    def __init__(self, device='cuda:0', config=None):
        self.default_config.update(config or {})
        self.device = device
        
        self.processor = CLIPSegProcessor.from_pretrained(self.default_config["model"])
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.default_config["model"])
        self.model.to(self.device)
        self.model.eval()

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
        w, h = image.size
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with timeblock("detection"):
                inputs = self.processor(
                    images=[image] * len(texts),
                    text=texts,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits  # (num_texts, height, width)
                
                # Apply sigmoid to get probabilities
                masks = torch.sigmoid(logits).float()  # (num_texts, height, width)
                
                # Resize masks to original image dimensions
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Get confidence scores as mean probability in each mask
                scores = masks.view(masks.shape[0], -1).max(dim=1)[0].cpu()
                
                # Filter by confidence threshold
                mask_filter = scores > self.default_config["detection_threshold"]
                masks = masks[mask_filter]
                scores = scores[mask_filter]
                labels = torch.arange(len(texts))[mask_filter]
                
                # Convert masks to binary using threshold
                binary_masks = (masks > self.default_config["mask_threshold"]).cpu()
                
                # Calculate bounding boxes from masks
                boxes = []
                # for mask in binary_masks:
                #     y_indices, x_indices = torch.where(mask > 0)
                #     if len(y_indices) > 0:
                #         x_min, x_max = x_indices.min(), x_indices.max()
                #         y_min, y_max = y_indices.min(), y_indices.max()
                #         boxes.append([x_min, y_min, x_max, y_max])
                #     else:
                #         boxes.append([0, 0, 0, 0])
                # boxes = torch.tensor(boxes)

                if len(binary_masks) == 0:
                    return {
                        "boxes": [],
                        "masks": [],
                        "scores": [],
                        "labels": [],
                    }

                return {
                    "boxes": boxes,
                    "masks": binary_masks,
                    "scores": scores,
                    "labels": labels,
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