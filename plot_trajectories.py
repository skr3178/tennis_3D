#!/usr/bin/env python3
"""Plot paper synthetic vs our Unity-game trajectories (side view + top-down)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── colour map: dark-red → orange → yellow ──
cmap = LinearSegmentedColormap.from_list("traj", ["#8B0000", "#CC3300", "#FF6600", "#FFaa00", "#FFdd44"])

# ── load paper synthetic trajectories ──
paper_dir = Path("/media/skr/storage/ten_bad/paper_npz/synthetic")
paper_trajs = []
for f in sorted(paper_dir.glob("seq_*.npz")):
    d = np.load(f, allow_pickle=True)
    xyz = d["xyz"]  # (N, 3) — x, y, z
    paper_trajs.append(xyz)

# ── load our game-based trajectories ──
ours_dir = Path("/media/skr/storage/ten_bad/TennisDataset_paperbot_10_test/test")
ours_trajs = []
for ep_dir in sorted(ours_dir.iterdir()):
    csv_path = ep_dir / "frames.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    xyz = df[["x", "y", "z"]].values  # x, y(height), z(along court)
    ours_trajs.append(xyz)

NET_HEIGHT = 1.07

def draw_net(ax, view):
    if view == "side":
        ax.plot([0, 0], [0, NET_HEIGHT], color="gray", lw=2.5, zorder=5)
    else:
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)

def plot_trajs(ax, trajs, view, title):
    n = len(trajs)
    for i, xyz in enumerate(trajs):
        colour = cmap(i / max(n - 1, 1))
        if view == "side":
            ax.plot(xyz[:, 2], xyz[:, 1], color=colour, lw=0.9, alpha=0.7)
        else:
            ax.plot(xyz[:, 0], xyz[:, 2], color=colour, lw=0.9, alpha=0.7)
    draw_net(ax, view)
    ax.set_title(title, fontsize=11)
    if view == "side":
        ax.set_xlabel("z (m, along court)")
        ax.set_ylabel("y (m, height)")
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 4)
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-15, 15)
    ax.set_aspect("auto")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("paper's Synthetic TrackNet (Tennis) vs our Unity-game trajectories", fontsize=14, y=0.98)

plot_trajs(axes[0, 0], paper_trajs, "side", f"paper synthetic ({len(paper_trajs)} trajectories)   side view z-y")
plot_trajs(axes[0, 1], ours_trajs,  "side", f"our game-based ({len(ours_trajs)} trajectories)   side view z-y")
plot_trajs(axes[1, 0], paper_trajs, "top",  "paper synthetic — top-down x-z")
plot_trajs(axes[1, 1], ours_trajs,  "top",  "our game-based — top-down x-z")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "/media/skr/storage/ten_bad/paper_vs_ours_new.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
