"""SSDLite with batched class postprocessing for the >=0.5 mask consumer.

The released detector weights, box decoding, class-specific NMS and global
detection limit are retained. Top-k is computed for all classes together,
avoiding ninety Python/CUDA synchronization rounds per detector call.
Detections below 0.5 are unused by the monocular person-mask consumers.
"""

from types import MethodType

import torch
from torchvision.ops import boxes as box_ops


def high_confidence_postprocess(self, head_outputs, image_anchors, image_shapes):
    probabilities = head_outputs["cls_logits"].softmax(dim=-1)
    output = []
    for regression, scores, anchors, shape in zip(head_outputs["bbox_regression"], probabilities,
                                                 image_anchors, image_shapes):
        boxes = box_ops.clip_boxes_to_image(self.box_coder.decode_single(regression, anchors), shape)
        # Class 0 is background. Low-score entries cannot suppress a higher
        # score during NMS, or outrank it in the final global detection cap.
        scores, indices = scores[:, 1:].T.topk(min(self.topk_candidates, scores.shape[0]), dim=1)
        labels = torch.arange(1, probabilities.shape[-1], device=scores.device)[:, None].expand_as(indices)
        keep = scores >= .5
        scores, labels, boxes = scores[keep], labels[keep], boxes[indices[keep]]
        selected = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
        output.append(dict(boxes=boxes[selected], scores=scores[selected], labels=labels[selected]))
    return output


def make_person_detector(device):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
    detector = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).to(device).eval()
    detector.postprocess_detections = MethodType(high_confidence_postprocess, detector)
    return detector
