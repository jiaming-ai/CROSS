"""CROSS Nav: a bird's-eye-view (BEV) traversability head on the frozen VGGT-Omega backbone.

labels.py   world BEV of a scene from the accumulated ground truth (gravity, ground, free / occupied / unknown)
bev.py      BEV frame of a camera (gravity-aligned, heading-aligned) and the crop of a world BEV into it
data.py     training windows with their BEV labels
head.py     the BEV head (multi-layer features lifted with VGGT's own depth, BEV transformer, decoder)
model.py    frozen backbone + scale head + BEV head
train.py    training;  evaluate.py  held-out metrics and figures
"""
