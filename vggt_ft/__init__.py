"""Fine-tuning of VGGT-Omega for CROSS: a metric-scale head, a pairwise covisibility head and multi-session training.

The released model (third_party/vggt-omega) is kept as it is: the fine-tuned checkpoint loads into `VGGTOmega` (its
extra heads are ignored there) and the heads can be loaded on their own on top of any VGGT-Omega checkpoint
(`vggt_ft.heads`).
"""


def load_yaml(path):
    """A training / evaluation config with ${VAR} in its strings replaced from the environment (the shipped configs put
    data, checkpoints and runs under ${VGGT_FT_ROOT}); an unset variable is an error."""
    import os
    import re

    import yaml

    def expand(v):
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        if isinstance(v, str):
            missing = [m for m in re.findall(r"\$\{(\w+)\}", v) if m not in os.environ]
            if missing:
                raise KeyError(f"{path}: environment variable(s) {missing} not set (used in {v!r})")
            return os.path.expandvars(v)
        return v

    with open(path) as f:
        return expand(yaml.safe_load(f))
