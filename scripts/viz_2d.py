"""2D visualization of the reconstructed 3D trajectory: top-down (XY) plus
side view (YZ), with the singles court drawn and bounce points marked.

Usage:
    python scripts/viz_2d.py data/game1_clip1 --out data/game1_clip1/traj_2d.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# Court (singles) in metres
DOUBLES_HALF_W = 5.485
SINGLES_HALF_W = 4.115
COURT_HALF_L = 11.885
SERVICE_LINE = 6.40
NET_HEIGHT = 0.914


def draw_court_xy(ax: plt.Axes) -> None:
    # Doubles outer rectangle (lighter)
    ax.add_patch(
        Rectangle(
            (-DOUBLES_HALF_W, -COURT_HALF_L),
            2 * DOUBLES_HALF_W,
            2 * COURT_HALF_L,
            linewidth=1.0,
            edgecolor="#999999",
            facecolor="#dbe9f4",
        )
    )
    # Singles court (slightly brighter)
    ax.add_patch(
        Rectangle(
            (-SINGLES_HALF_W, -COURT_HALF_L),
            2 * SINGLES_HALF_W,
            2 * COURT_HALF_L,
            linewidth=1.5,
            edgecolor="white",
            facecolor="#bcd6ec",
        )
    )
    # Net
    ax.plot([-DOUBLES_HALF_W, DOUBLES_HALF_W], [0, 0], color="black", lw=2)
    # Service lines
    for y in (+SERVICE_LINE, -SERVICE_LINE):
        ax.plot([-SINGLES_HALF_W, SINGLES_HALF_W], [y, y], color="white", lw=1.2)
    # Center service line (between service lines only)
    ax.plot([0, 0], [-SERVICE_LINE, SERVICE_LINE], color="white", lw=1.2)
    # Center marks at baselines
    for y in (-COURT_HALF_L, +COURT_HALF_L):
        ax.plot([-0.1, 0.1], [y, y], color="white", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-DOUBLES_HALF_W - 1.5, DOUBLES_HALF_W + 1.5)
    ax.set_ylim(-COURT_HALF_L - 1.5, COURT_HALF_L + 1.5)
    ax.set_xlabel("X (m, sideline)")
    ax.set_ylabel("Y (m, baseline-to-baseline)")
    ax.set_title("Top-down (XY)")


def draw_court_yz(ax: plt.Axes) -> None:
    # Court surface as horizontal line at z=0
    ax.plot([-COURT_HALF_L, +COURT_HALF_L], [0, 0], color="#888888", lw=2)
    # Net (vertical line at y=0, height ~0.914 m)
    ax.plot([0, 0], [0, NET_HEIGHT], color="black", lw=3)
    # Service lines markers
    for y in (-SERVICE_LINE, +SERVICE_LINE, -COURT_HALF_L, +COURT_HALF_L):
        ax.plot([y, y], [0, 0.05], color="white", lw=1.2)
    ax.set_xlim(-COURT_HALF_L - 1.5, COURT_HALF_L + 1.5)
    ax.set_ylim(-0.2, 5.5)
    ax.set_xlabel("Y (m, baseline-to-baseline)")
    ax.set_ylabel("Z (m, height)")
    ax.set_title("Side view (YZ)")
    ax.set_aspect("equal")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rdir: Path = args.rally_dir
    out = args.out or (rdir / "traj_2d.png")

    df = pd.read_csv(rdir / "ball_traj_3D.csv").sort_values("idx").reset_index(drop=True)
    bounces_csv = rdir / "bounces.csv"
    bounces = pd.read_csv(bounces_csv) if bounces_csv.exists() else None

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    # Top-down (XY)
    ax = axs[0]
    ax.set_facecolor("#f6f6f6")
    draw_court_xy(ax)
    sc = ax.scatter(df["x"], df["y"], c=df["idx"], cmap="viridis", s=12, zorder=3)
    if bounces is not None:
        ax.scatter(
            bounces["x"], bounces["y"],
            marker="X", s=110, edgecolors="red", facecolors="none",
            linewidths=2.0, zorder=4,
            label=f"bounces ({len(bounces)})",
        )
        for _, r in bounces.iterrows():
            ax.annotate(f"f{int(r['frame'])}", (r["x"], r["y"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="red", zorder=5)
        ax.legend(loc="lower right", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("frame index")

    # Side (YZ)
    ax = axs[1]
    ax.set_facecolor("#f6f6f6")
    draw_court_yz(ax)
    ax.scatter(df["y"], df["z"], c=df["idx"], cmap="viridis", s=12, zorder=3)
    if bounces is not None:
        ax.scatter(
            bounces["y"], bounces["z"],
            marker="X", s=110, edgecolors="red", facecolors="none",
            linewidths=2.0, zorder=4,
        )

    fig.suptitle(
        f"Tennis 3D ball trajectory  —  {len(df)} frames, "
        f"{0 if bounces is None else len(bounces)} bounces",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[viz_2d] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
