#!/usr/bin/env python3
"""Example-frame grid for the SimChange benchmark figure (frames fetched into report/figures/sim_frames)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from pathlib import Path
F = Path(__file__).resolve().parents[1] / 'report/figures'
vars_ = ['map', 'light_morning', 'light_evening', 'light_night', 'light_overcast', 'move_50', 'remove_50', 'background', 'yaw_30', 'offset_1.0']
titles = ['map', 'morning', 'evening', 'night', 'overcast', 'move 50%', 'remove 50%', 'background', 'yaw 30°', 'offset 1 m']
fig, axs = plt.subplots(2, 10, figsize=(16, 2.9))
for r, (scene, idx) in enumerate([('classroom', '000060'), ('archiviz', '000020')]):
    for c, (v, t) in enumerate(zip(vars_, titles)):
        p = F / 'sim_frames' / scene / v / 'left' / f'{idx}.png'
        ax = axs[r, c]; ax.axis('off')
        if p.is_file():
            ax.imshow(mpimg.imread(p))
        if r == 0:
            ax.set_title(t, fontsize=9)
fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.005, right=0.995, top=0.9, bottom=0.01)
fig.savefig(F / 'sim_examples.pdf'); fig.savefig(F / 'sim_examples.png', dpi=110)
