"""Compare our reconstructed `ball_traj_3D.csv` against the Where-Is-The-Ball
paper's ground-truth 3D trajectory.

GT axis convention is (X across, Y up, Z baseline-to-baseline). Our pipeline
uses (X across, Y baseline-to-baseline, Z up). So we map their (x, z, y) → our
(x, y, z) before comparing.

Usage:
    python scripts/compare_witb.py data/witb_id00 \
        --gt-json /media/skr/storage/ten_bad/notes/screenshots/where_is_the_ball/clips_mp4/id00_game1_Clip1_f33-191.json
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
from matplotlib.patches import Rectangle


def gt_to_our_frame(gt_xyz_their: np.ndarray) -> np.ndarray:
    """Their (X across, Y up, Z length) → our (X across, Y length, Z up)."""
    x = gt_xyz_their[:, 0]
    y = gt_xyz_their[:, 2]   # their Z (length) → our Y
    z = gt_xyz_their[:, 1]   # their Y (up)     → our Z
    return np.column_stack([x, y, z])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rdir: Path = args.rally_dir
    out_png = args.out or (rdir / "compare_witb.png")

    with open(args.gt_json) as fh:
        d = json.load(fh)
    gt_their = np.array(d["gt"], dtype=np.float64)        # (N, 3)
    gt = gt_to_our_frame(gt_their)                        # (N, 3) in our axes
    n_frames = d["num_frames"]
    print(f"[compare] GT: {n_frames} frames  has_real_gt={d['has_real_gt']}")
    print(f"[compare] GT axes (our convention) range: "
          f"x [{gt[:,0].min():+.2f}, {gt[:,0].max():+.2f}]  "
          f"y [{gt[:,1].min():+.2f}, {gt[:,1].max():+.2f}]  "
          f"z [{gt[:,2].min():+.2f}, {gt[:,2].max():+.2f}]")

    ours = pd.read_csv(rdir / "ball_traj_3D.csv").sort_values("idx").reset_index(drop=True)
    print(f"[compare] OURS: {len(ours)} frames")

    # Index alignment: GT index i corresponds to video frame i (0-based);
    # our `idx` column also uses video-frame indexing.
    gt_lookup = {i: gt[i] for i in range(len(gt))}
    matched = []
    for _, r in ours.iterrows():
        f = int(r["idx"])
        if 0 <= f < len(gt):
            matched.append((f, np.array([r["x"], r["y"], r["z"]]), gt_lookup[f]))

    if not matched:
        print("[compare] no overlap between OURS and GT frames")
        return 1

    frames = np.array([m[0] for m in matched])
    o = np.array([m[1] for m in matched])
    g = np.array([m[2] for m in matched])
    err_xyz = o - g
    err_3d = np.linalg.norm(err_xyz, axis=1)
    err_xy = np.linalg.norm(err_xyz[:, :2], axis=1)
    err_z = np.abs(err_xyz[:, 2])

    print("\n[compare] error metrics over overlapping frames:")
    print(f"  3D    err (m):  mean={err_3d.mean():.3f}  med={np.median(err_3d):.3f}  "
          f"p90={np.quantile(err_3d, 0.9):.3f}  max={err_3d.max():.3f}")
    print(f"  X-Y   err (m):  mean={err_xy.mean():.3f}  med={np.median(err_xy).round(3)}  "
          f"p90={np.quantile(err_xy, 0.9):.3f}  max={err_xy.max():.3f}")
    print(f"  Z     err (m):  mean={err_z.mean():.3f}   med={np.median(err_z):.3f}   "
          f"p90={np.quantile(err_z, 0.9):.3f}   max={err_z.max():.3f}")
    print(f"  N matched: {len(matched)}/{len(ours)} ours, "
          f"{len(matched)}/{len(gt)} gt")

    # Plot: top-down + side view, GT vs OURS
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")

    ax = axs[0]
    ax.set_title("Top-down (XY)")
    ax.add_patch(Rectangle((-4.115, -11.885), 2 * 4.115, 2 * 11.885,
                           edgecolor="#888", facecolor="#dbe9f4", lw=1))
    ax.plot([-4.115, 4.115], [0, 0], color="black", lw=2)
    ax.plot([-4.115, 4.115], [6.40, 6.40], color="white", lw=1)
    ax.plot([-4.115, 4.115], [-6.40, -6.40], color="white", lw=1)
    ax.plot([0, 0], [-6.40, 6.40], color="white", lw=1)
    ax.scatter(gt[:, 0], gt[:, 1], c="#1f77b4", s=10, label="GT (full)")
    ax.scatter(o[:, 0], o[:, 1], c="#d62728", s=14,
               marker="x", label=f"OURS ({len(o)})")
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7); ax.set_ylim(-13, 13)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axs[1]
    ax.set_title("Side (YZ)")
    ax.plot([-11.885, 11.885], [0, 0], color="#888", lw=2)
    ax.plot([0, 0], [0, 0.914], color="black", lw=3)
    ax.scatter(gt[:, 1], gt[:, 2], c="#1f77b4", s=10, label="GT")
    ax.scatter(o[:, 1], o[:, 2], c="#d62728", s=14, marker="x", label="OURS")
    ax.set_xlim(-13, 13); ax.set_ylim(-0.2, 5.0)
    ax.set_xlabel("Y (m, length)"); ax.set_ylabel("Z (m, height)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    ax = axs[2]
    ax.set_title("Per-frame 3D error (m)")
    ax.plot(frames, err_3d, "o-", color="#d62728", ms=4, label="3D")
    ax.plot(frames, err_xy, ".-", color="#1f77b4", ms=3, label="XY")
    ax.plot(frames, err_z,  ".-", color="#2ca02c", ms=3, label="Z")
    ax.set_xlabel("frame"); ax.set_ylabel("err (m)")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle(
        f"WITB clip {d['id']:02d}  ({d['g_name']}/{d['c_name']})  "
        f"GT vs OURS  —  med 3D err = {np.median(err_3d):.2f} m, "
        f"matched {len(matched)} frames",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[compare] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
