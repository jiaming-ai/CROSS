"""Released XFeat LighterGlue adapter for low-rate retrieved-pair geometry.

Architecture/checkpoint: https://github.com/verlab/accelerated_features
The default local tracker continues to use descriptor matching.
"""
import hashlib
import inspect
from pathlib import Path

import torch

LIGHTERGLUE_SHA256 = "766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"


def load_lighterglue(extractor, device):
    from kornia.feature.lightglue import LightGlue

    # Use the checkpoint distributed with the same XFeat source already loaded
    # by torch.hub; avoid a second implicit download or mutable model revision.
    weights = Path(inspect.getfile(type(extractor))).parent.parent / "weights/xfeat-lighterglue.pt"
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != LIGHTERGLUE_SHA256:
        raise RuntimeError("The tested XFeat LighterGlue checkpoint is missing or has changed")
    network = LightGlue(features=None, name="xfeat", input_dim=64, descriptor_dim=96,
                        add_scale_ori=False, add_laf=False, scale_coef=1., n_layers=6,
                        num_heads=1, flash=True, mp=False, depth_confidence=-1,
                        width_confidence=.95, filter_threshold=.1, weights=None)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    for i in range(6):
        for old, new in [(f"self_attn.{i}", f"transformers.{i}.self_attn"),
                         (f"cross_attn.{i}", f"transformers.{i}.cross_attn")]:
            state = {k.replace(old, new): v for k, v in state.items()}
    state = {k.removeprefix("matcher."): v for k, v in state.items()}
    expected = network.state_dict()
    extra, missing = set(state) - set(expected), set(expected) - set(state)
    # Released training state includes an extractor. Kornia0.8 additionally
    # constructs a deterministic early-stopping buffer (disabled above).
    # Never silently accept a missing learned matcher parameter.
    if (any(not k.startswith("extractor.model.net.") for k in extra)
            or missing - {"confidence_thresholds"}
            or missing & set(dict(network.named_parameters()))):
        raise RuntimeError(f"Incompatible LighterGlue checkpoint: missing={missing}, extra={extra}")
    converted = {k: state[k] if k in state else expected[k] for k in expected}
    network.load_state_dict(converted, strict=True)
    return network.to(device).eval()


@torch.inference_mode()
def match_lighterglue(network, reference, current):
    def image(features):
        points = features["keypoints"]
        return dict(keypoints=points[None], descriptors=features["descriptors"][None],
                    image_size=points.new_tensor(features["shape"][::-1])[None])
    result = network(dict(image0=image(reference), image1=image(current)))
    indices = result["matches"][0]
    return (reference["keypoints"][indices[:, 0]].cpu().numpy(),
            current["keypoints"][indices[:, 1]].cpu().numpy())
