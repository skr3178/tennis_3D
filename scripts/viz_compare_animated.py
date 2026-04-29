"""Side-by-side animated 3D comparison: OURS (left) vs GT (right) for one
WITB clip. H.264 MP4 output (matplotlib's FFMpegWriter pipes to ffmpeg).

Shows per-frame ball position with a recent trail; bounces marked as red X;
court drawn as a low-saturation surface with line markings and net.

Usage:
    python scripts/viz_compare_animated.py data/witb_id00 \
        --gt-json /media/skr/storage/ten_bad/notes/screenshots/where_is_the_ball/clips_mp4/id00_game1_Clip1_f33-191.json \
        --out data/witb_id00/compare_anim.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DOUBLES_HALF_W = 5.485
SINGLES_HALF_W = 4.115
COURT_HALF_L = 11.885
SERVICE_LINE = 6.40
NET_HEIGHT = 0.914


def gt_to_our_frame(gt_their: np.ndarray) -> np.ndarray:
    return np.column_stack([gt_their[:, 0], gt_their[:, 2], gt_their[:, 1]])


def setup_3d_axes(ax, title: str) -> None:
    ax.set_xlim(-7, 7)
    ax.set_ylim(-COURT_HALF_L - 2, COURT_HALF_L + 2)
    ax.set_zlim(0, 5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect((14, 26, 5))

    # Court surface (light blue translucent quad)
    pts = [
        [-DOUBLES_HALF_W, -COURT_HALF_L, 0],
        [+DOUBLES_HALF_W, -COURT_HALF_L, 0],
        [+DOUBLES_HALF_W, +COURT_HALF_L, 0],
        [-DOUBLES_HALF_W, +COURT_HALF_L, 0],
    ]
    surf = Poly3DCollection([pts], facecolors="#bcd6ec", edgecolors="#888", alpha=0.35)
    ax.add_collection3d(surf)
    # Singles court outline
    for seg in [
        ([-SINGLES_HALF_W, -COURT_HALF_L, 0], [+SINGLES_HALF_W, -COURT_HALF_L, 0]),
        ([-SINGLES_HALF_W, +COURT_HALF_L, 0], [+SINGLES_HALF_W, +COURT_HALF_L, 0]),
        ([-SINGLES_HALF_W, -COURT_HALF_L, 0], [-SINGLES_HALF_W, +COURT_HALF_L, 0]),
        ([+SINGLES_HALF_W, -COURT_HALF_L, 0], [+SINGLES_HALF_W, +COURT_HALF_L, 0]),
        ([-SINGLES_HALF_W, +SERVICE_LINE, 0], [+SINGLES_HALF_W, +SERVICE_LINE, 0]),
        ([-SINGLES_HALF_W, -SERVICE_LINE, 0], [+SINGLES_HALF_W, -SERVICE_LINE, 0]),
        ([0, -SERVICE_LINE, 0], [0, +SERVICE_LINE, 0]),
    ]:
        a, b = seg
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="white", lw=1.2)
    # Net as thin black quad
    net_pts = [
        [-DOUBLES_HALF_W, 0, 0],
        [+DOUBLES_HALF_W, 0, 0],
        [+DOUBLES_HALF_W, 0, NET_HEIGHT],
        [-DOUBLES_HALF_W, 0, NET_HEIGHT],
    ]
    net = Poly3DCollection([net_pts], facecolors="#222222", edgecolors="black", alpha=0.55)
    ax.add_collection3d(net)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--trail", type=int, default=12,
                   help="Number of past frames to draw as trail")
    args = p.parse_args()

    rdir: Path = args.rally_dir
    out_path = args.out or (rdir / "compare_anim.mp4")

    # GT
    with open(args.gt_json) as fh:
        d = json.load(fh)
    gt = gt_to_our_frame(np.array(d["gt"], dtype=np.float64))     # (N, 3)
    n_frames = len(gt)
    title_clip = f"clip {d['id']:02d}  ({d['g_name']}/{d['c_name']})"

    # OURS — sparse, indexed by frame
    ours_csv = rdir / "ball_traj_3D.csv"
    ours_lookup: dict[int, np.ndarray] = {}
    if ours_csv.exists():
        df = pd.read_csv(ours_csv)
        for _, r in df.iterrows():
            ours_lookup[int(r["idx"])] = np.array([r["x"], r["y"], r["z"]])
    bounces_csv = rdir / "bounces.csv"
    bounces = pd.read_csv(bounces_csv) if bounces_csv.exists() else None
    print(f"[viz] {n_frames} GT frames, {len(ours_lookup)} OURS frames, "
          f"{0 if bounces is None else len(bounces)} bounces")

    fig = plt.figure(figsize=(18, 7), facecolor="white")
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    setup_3d_axes(ax_l, "OURS (single-cam physics)")
    setup_3d_axes(ax_r, "GT (paper, multi-cam)")

    # Persistent artists per side: trail line + current ball + bounces
    ours_trail, = ax_l.plot([], [], [], "-", color="#d62728", lw=1.6, alpha=0.7)
    ours_ball = ax_l.scatter([], [], [], color="#d62728", s=70, depthshade=False)
    gt_trail, = ax_r.plot([], [], [], "-", color="#1f77b4", lw=1.6, alpha=0.7)
    gt_ball = ax_r.scatter([], [], [], color="#1f77b4", s=70, depthshade=False)

    # Bounces (static once placed)
    if bounces is not None and len(bounces) > 0:
        ax_l.scatter(bounces["x"], bounces["y"], bounces["z"],
                     marker="X", s=140, color="red", depthshade=False, zorder=5)

    fig.suptitle(f"WITB {title_clip}  —  ball trajectory comparison",
                 y=0.98, fontsize=13)

    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=110):
        for fi in range(n_frames):
            # OURS trail (most recent `trail` frames including fi, only those present)
            ours_pts = []
            for k in range(max(0, fi - args.trail + 1), fi + 1):
                if k in ours_lookup:
                    ours_pts.append(ours_lookup[k])
            if ours_pts:
                arr = np.asarray(ours_pts)
                ours_trail.set_data(arr[:, 0], arr[:, 1])
                ours_trail.set_3d_properties(arr[:, 2])
                cur = arr[-1] if fi in ours_lookup else None
                if cur is not None:
                    ours_ball._offsets3d = ([cur[0]], [cur[1]], [cur[2]])
                else:
                    ours_ball._offsets3d = ([], [], [])
            else:
                ours_trail.set_data([], [])
                ours_trail.set_3d_properties([])
                ours_ball._offsets3d = ([], [], [])

            # GT trail
            lo = max(0, fi - args.trail + 1)
            seg = gt[lo:fi + 1]
            gt_trail.set_data(seg[:, 0], seg[:, 1])
            gt_trail.set_3d_properties(seg[:, 2])
            cur = gt[fi]
            gt_ball._offsets3d = ([cur[0]], [cur[1]], [cur[2]])

            # Frame counter + ours/gt deltas
            ours_have = fi in ours_lookup
            if ours_have:
                err = float(np.linalg.norm(ours_lookup[fi] - gt[fi]))
                ax_l.set_title(f"OURS  frame {fi:3d}  err={err:.2f} m")
            else:
                ax_l.set_title(f"OURS  frame {fi:3d}  (no fit)")
            ax_r.set_title(f"GT    frame {fi:3d}  pos=({gt[fi,0]:+.1f},"
                           f"{gt[fi,1]:+.1f},{gt[fi,2]:.2f})")

            writer.grab_frame()
    print(f"[viz] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
