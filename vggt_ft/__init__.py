"""Fine-tuning of VGGT-Omega for CROSS: a metric-scale head, a pairwise covisibility head and multi-session training.

The released model (third_party/vggt-omega) is kept as it is: the fine-tuned checkpoint loads into `VGGTOmega` (its
extra heads are ignored there) and the heads can be loaded on their own on top of any VGGT-Omega checkpoint
(`vggt_ft.heads`).
"""
