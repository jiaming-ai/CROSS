"""Released XFeat LighterGlue adapter for low-rate retrieved-pair geometry.

Architecture/checkpoint: https://github.com/verlab/accelerated_features
The default local tracker continues to use descriptor matching.
"""
import hashlib
import inspect
from pathlib import Path

import torch

LIGHTERGLUE_SHA256 = "766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"
SUPERPOINT_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
SUPERPOINT_LIGHTGLUE_SHA256 = "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"


def load_superpoint_lightglue(device, keypoints=1600):
    """Optional official SuperPoint/LightGlue backend, with verified weights.

    Install cvg/LightGlue at eb42fee2d71449efb0aa5c10549752b5d75384d8.
    Both networks keep the release defaults; extraction keeps image coordinates.
    """
    try:
        from lightglue import LightGlue, SuperPoint
    except ImportError as exc:
        raise RuntimeError("SuperPoint retrieval requires the optional cvg/LightGlue package; see README") from exc
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    for release_name, cache_name, digest in (
        ("superpoint_v1.pth", "superpoint_v1.pth", SUPERPOINT_SHA256),
        ("superpoint_lightglue.pth", "superpoint_lightglue_v0-1_arxiv.pth", SUPERPOINT_LIGHTGLUE_SHA256),
    ):
        path = cache / cache_name
        if not path.exists():
            torch.hub.download_url_to_file(
                "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/" + release_name,
                str(path), hash_prefix=digest)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Unexpected retrieval checkpoint contents: {cache_name}")
    network = SuperPoint(max_num_keypoints=keypoints).to(device).eval()

    class Extractor:
        @torch.inference_mode()
        def detectAndCompute(self, tensor):
            features = network.extract(tensor, resize=None)
            return [dict(keypoints=features["keypoints"][i], descriptors=features["descriptors"][i],
                         scores=features["keypoint_scores"][i]) for i in range(len(tensor))]

    return Extractor(), LightGlue(features="superpoint").to(device).eval()


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
