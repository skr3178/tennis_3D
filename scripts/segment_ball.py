"""M3 step 2: run the (unchanged) tt3d trajectory segmenter on the tennis
ball_traj_2D.csv. Sweeps a few values of L (segment-creation penalty) and
saves a diagnostic plot per L overlaying segment breakpoints on (t, u) and
(t, v).

Usage:
    python scripts/segment_ball.py data/game1_clip1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d.traj_seg.segmenter import basic_segmenter
from tennis_3d.traj_seg.utils import read_traj


def run_one(t, traj, L: float, out_path: Path, frame_idx, label: str) -> int:
    q_sol = basic_segmenter(t, traj, deg=2, L=L, use_blur=False)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axs[0].scatter(t, traj[:, 0], s=8, c="tab:blue")
    axs[1].scatter(t, traj[:, 1], s=8, c="tab:orange")
    for q in q_sol:
        axs[0].axvline(t[q], c="r", lw=0.7, zorder=0)
        axs[1].axvline(t[q], c="r", lw=0.7, zorder=0)
    axs[0].set_ylabel("u (pixels)")
    axs[1].set_ylabel("v (pixels)")
    axs[1].set_xlabel("time (s)")
    axs[0].set_title(f"{label}  L={L}  segments={len(q_sol) + 1}  breaks={len(q_sol)}")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    # Also print frame indices of breakpoints
    breaks_frames = [int(frame_idx[q]) for q in q_sol]
    print(f"  L={L:>6}  breaks at frames: {breaks_frames}")
    return len(q_sol)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument(
        "--Ls", type=float, nargs="+", default=[50, 100, 200, 500, 1000, 2000]
    )
    p.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        default=None,
        help="Restrict to a subset of frames [start end] inclusive",
    )
    args = p.parse_args()

    csv_path = args.rally_dir / "ball_traj_2D.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run make_ball_csv.py first.",
              file=sys.stderr)
        return 2
    traj = read_traj(csv_path)

    if args.frame_range is not None:
        a, b = args.frame_range
        keep = (traj[:, 0] >= a) & (traj[:, 0] <= b)
        traj = traj[keep]

    # Drop invisible frames before segmenting
    visible = traj[:, 3] != 0
    traj = traj[visible]

    frame_idx = traj[:, 0].astype(np.int64)
    t = traj[:, 0] / args.fps
    obs = traj[:, [1, 2]]
    label = f"frames {int(frame_idx.min())}-{int(frame_idx.max())} ({len(t)} visible)"
    print(f"[segment] {label}, fps={args.fps}")

    out_dir = args.rally_dir / "seg_diagnostic"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for L in args.Ls:
        out_path = out_dir / f"seg_L{int(L)}.png"
        n_breaks = run_one(t, np.column_stack([obs, traj[:, 3:5]]), L, out_path,
                           frame_idx, label)
        summary.append((L, n_breaks, str(out_path)))

    print("\n[segment] summary (L, n_breaks, plot):")
    for row in summary:
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
