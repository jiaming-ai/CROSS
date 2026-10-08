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


class GraphedPersonDetector:
    """The detector of `make_person_detector` with its backbone and heads replayed as a CUDA graph.

    SSDLite resizes every image to a fixed 320 x 320 input, so the backbone and the heads always see the same shapes:
    they are captured once and replayed; the input transform, the anchors (constant for the fixed size), the
    high-confidence postprocessing and the box rescaling run as in the eager model.  The same kernels run on the same
    data, so the detections equal the eager call's (checked on real frames); the replay removes the Python and launch
    overhead of ~100 small layers (13 ms -> ~1-2 ms per call on an RTX 5090).  Inputs on the CPU, or any failure to
    capture, fall back to the eager model."""

    def __init__(self, detector):
        self.model = detector
        self.graph = None
        self.static_in = None
        self.static_out = None
        self.anchors = None
        self.failed = False

    def _capture(self, image_list):
        det = self.model
        self.static_in = image_list.tensors.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                  # warm-up off the default stream, as graph capture requires
            for _ in range(3):
                feats = list(det.backbone(self.static_in).values())
                det.head(feats)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            feats = list(det.backbone(self.static_in).values())
            self.static_out = det.head(feats)
        self.anchors = det.anchor_generator(image_list, list(det.backbone(self.static_in).values()))
        self.graph = graph
        # person-only graph: the class-1 scores' top-k and their boxes, as high_confidence_postprocess computes them
        k = min(det.topk_candidates, self.anchors[0].shape[0])
        size = image_list.image_sizes[0]
        pgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pgraph):
            feats = list(det.backbone(self.static_in).values())
            out = det.head(feats)
            prob = out["cls_logits"][0].softmax(dim=-1)
            boxes = box_ops.clip_boxes_to_image(det.box_coder.decode_single(out["bbox_regression"][0], self.anchors[0]), size)
            p_scores, p_idx = prob[:, 1].topk(k)
            self.person_out = (p_scores, boxes[p_idx])
        self.person_graph = pgraph

    def persons(self, image: torch.Tensor, score: float = 0.5) -> torch.Tensor:
        """Person boxes (n, 4) in the image's pixel coordinates with score >= `score` (0.5 at most: the patched
        postprocessing keeps nothing below it), as the full model's class-1 detections: same scores, class-specific
        NMS (batched_nms with one label), box rescaling; the other 89 classes are not post-processed."""
        det = self.model
        if self.failed or not image.is_cuda:
            res = det([image])[0]
            return res["boxes"][(res["labels"] == 1) & (res["scores"] >= score)]
        image_list, _ = det.transform([image])
        if self.graph is None:
            try:
                self._capture(image_list)
            except Exception:                          # noqa: BLE001
                self.failed = True
                return self.persons(image, score)
        if image_list.tensors.shape != self.static_in.shape:
            res = det([image])[0]
            return res["boxes"][(res["labels"] == 1) & (res["scores"] >= score)]
        self.static_in.copy_(image_list.tensors)
        self.person_graph.replay()
        p_scores, p_boxes = self.person_out
        keep = p_scores >= max(score, 0.5)
        scores, boxes = p_scores[keep], p_boxes[keep]
        selected = box_ops.batched_nms(boxes, scores, torch.ones_like(scores, dtype=torch.int64), det.nms_thresh)
        selected = selected[:det.detections_per_img]
        out = det.transform.postprocess([{"boxes": boxes[selected]}], image_list.image_sizes, [tuple(image.shape[-2:])])
        return out[0]["boxes"]

    def __call__(self, images):
        det = self.model
        if self.failed or not images[0].is_cuda:
            return det(images)
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, _ = det.transform(images)
        if len(images) != 1 or (self.static_in is not None and image_list.tensors.shape != self.static_in.shape):
            return det(images)
        if self.graph is None:
            try:
                self._capture(image_list)
            except Exception:                          # noqa: BLE001  e.g. capture unsupported: eager from now on
                self.failed = True
                return det(images)
        self.static_in.copy_(image_list.tensors)
        self.graph.replay()
        detections = det.postprocess_detections(self.static_out, self.anchors, image_list.image_sizes)
        return det.transform.postprocess(detections, image_list.image_sizes, original_sizes)


def make_person_detector(device):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
    detector = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).to(device).eval()
    detector.postprocess_detections = MethodType(high_confidence_postprocess, detector)
    return detector
