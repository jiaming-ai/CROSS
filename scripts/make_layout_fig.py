#!/usr/bin/env python3
"""Schematic of the observation layout (report/figures/layout.pdf)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt

FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis("off")

def box(x, y, w, h, text, fc, ec="k", fs=8, lw=1.0):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

# context row
ax.text(0.2, 5.1, "views of one forward pass (shared gauge)", fontsize=9, va="center")
ax.add_patch(mp.Rectangle((0.15, 2.25), 9.7, 2.55, fc="#f8fafc", ec="#94a3b8", lw=0.8, ls="--"))
box(0.4, 3.6, 1.3, 0.9, "current\nleft $I^L_c$\n(reference)", "#bfdbfe")
box(2.0, 3.6, 1.1, 0.9, "kf $\\mathcal{K}_1$\nleft", "#e2e8f0")
box(3.25, 3.6, 1.1, 0.9, "kf $\\mathcal{K}_2$\nleft", "#e2e8f0")
ax.text(4.75, 4.05, "$\\cdots$", fontsize=12, ha="center", va="center")
box(5.1, 3.6, 1.1, 0.9, "kf $\\mathcal{K}_B$\nleft", "#e2e8f0")
box(6.5, 3.6, 1.2, 0.9, "prev. obs.\n$I^L_p$ (opt.)", "#fde68a")
box(0.4, 2.4, 1.3, 0.9, "current\nright $I^R_c$", "#bbf7d0")
box(2.0, 2.4, 1.1, 0.9, "kf $\\mathcal{K}_1$\nright", "#bbf7d0")
box(3.25, 2.4, 1.1, 0.9, "kf $\\mathcal{K}_2$\nright", "#bbf7d0")
ax.text(4.75, 2.85, "$A$ anchors", fontsize=8, ha="center", va="center", color="#166534")

# anchor arrows (known transforms)
for x in (1.05, 2.55, 3.8):
    ax.annotate("", xy=(x, 3.3), xytext=(x, 3.6), arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=1.6))
ax.text(1.15, 3.42, "$\\mathbf{B}$", color="#16a34a", fontsize=9)
ax.text(2.65, 3.42, "$\\mathbf{B}$", color="#16a34a", fontsize=9)
ax.text(3.9, 3.42, "$\\mathbf{B}$", color="#16a34a", fontsize=9)
ax.annotate("", xy=(6.5, 4.05), xytext=(1.7, 4.05), arrowprops=dict(arrowstyle="<->", color="#ea580c", lw=1.4, connectionstyle="arc3,rad=-0.35"))
ax.text(4.0, 4.95, "$\\Delta_{p\\to c}$ (odometry)", color="#ea580c", fontsize=8, ha="center")

# outputs
ax.annotate("", xy=(2.55, 1.9), xytext=(2.55, 2.25), arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.8))
box(0.4, 0.9, 2.9, 0.95, "scale anchors\n$s_j=\\|\\mathbf{t}^{known}_j\\|/\\|\\hat{\\mathbf{t}}_j\\|$\nrobust log-space fusion $\\to\\hat s,\\hat\\sigma_s$", "#dcfce7", fs=7.5)
box(3.6, 0.9, 2.9, 0.95, "metric relative poses\n$\\mathbf{T}_{\\mathcal{K}_i\\to c}=\\hat{\\mathbf{X}}_{\\mathcal{K}_i}^{-1}\\hat{\\mathbf{X}}_c$ (scaled)", "#dbeafe", fs=7.5)
box(6.8, 0.9, 2.9, 0.95, "covisibility $c_i$\nfrom $\\hat D_{\\mathcal{K}_i},\\hat D_c$\n(reprojection consistency)", "#fef3c7", fs=7.5)
for x in (1.85, 5.05, 8.25):
    ax.annotate("", xy=(x, 0.55), xytext=(x, 0.9), arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
box(0.4, 0.05, 9.3, 0.45, "CROSS back end (unchanged): proposal convolution, GMM filtering, hypothesis lifecycle, loop closure / PGO", "#f1f5f9", fs=8)
ax.annotate("", xy=(3.6, 1.35), xytext=(3.3, 1.35), arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
fig.tight_layout()
fig.savefig(FIG / "layout.pdf")
fig.savefig(FIG / "layout.png", dpi=180)
print("ok")
