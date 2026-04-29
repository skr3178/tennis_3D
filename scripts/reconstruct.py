"""M4: 3D ball-trajectory reconstruction for a tennis rally.

Reads camera.yaml, ball_traj_2D.csv from <rally_dir>, runs the segmenter at
L=2000, attempts a physics fit at every interior break, accepts those that
reproject within --reproj-threshold, and writes ball_traj_3D.csv plus a
summary text file.

Usage:
    python scripts/reconstruct.py data/game1_clip1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d.calibration.utils import get_K, read_camera_info
from tennis_3d.rally.rally import reconstruct_arcs, stitch_3d_csv
from tennis_3d.traj_seg.segmenter import basic_segmenter
from tennis_3d.traj_seg.utils import read_traj


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--seg-L", type=float, default=2000.0)
    p.add_argument("--reproj-threshold", type=float, default=8.0,
                   help="Max median reprojection error (px) to accept a bounce")
    p.add_argument("--window-seconds", type=float, default=0.4,
                   help="Half-window of frames around each bounce to fit")
    p.add_argument("--no-chain-arcs", action="store_true",
                   help="Disable extending output to half-time of neighbouring "
                        "bounces (default: chained, with per-frame reproj gate)")
    p.add_argument("--chain-reproj-gate", type=float, default=100.0,
                   help="Per-frame reproj gate (px) when chaining arcs")
    p.add_argument("--min-vz-down", type=float, default=1.0,
                   help="Min |v_z| (m/s, downward) at bounce to accept")
    p.add_argument("--enable-spin", action="store_true",
                   help="Allow Magnus during pre-bounce fit (default: locked off)")
    p.add_argument("--frame-range", type=int, nargs=2, default=None)
    args = p.parse_args()

    cam_path = args.rally_dir / "camera.yaml"
    csv_path = args.rally_dir / "ball_traj_2D.csv"
    if not cam_path.exists() or not csv_path.exists():
        print(f"ERROR: need {cam_path} and {csv_path}", file=sys.stderr)
        return 2

    rvec, tvec, f, h, w = read_camera_info(cam_path)
    K = get_K(f, h, w)
    print(f"[reconstruct] camera f={f:.1f}px  rvec={rvec.flatten()}  "
          f"tvec={tvec.flatten()}")

    traj = read_traj(csv_path)
    if args.frame_range is not None:
        a, b = args.frame_range
        traj = traj[(traj[:, 0] >= a) & (traj[:, 0] <= b)]
    visible = traj[:, 3] != 0
    traj = traj[visible]

    frame_idx = traj[:, 0].astype(np.int64)
    t = traj[:, 0] / args.fps
    obs = traj[:, 1:6]   # X, Y, Visibility, L, Theta

    print(f"[reconstruct] segmenting {len(t)} visible frames at L={args.seg_L}")
    t0 = time.time()
    q_sol = basic_segmenter(t, obs, deg=2, L=args.seg_L, use_blur=False)
    dt = time.time() - t0
    print(f"[reconstruct] segmenter found {len(q_sol)} breaks in {dt:.1f}s")
    print(f"[reconstruct] break frames: "
          f"{[int(frame_idx[q]) for q in q_sol]}")

    # Pack a (n, 2+) "traj_2d" with the frame number as col 2 (used only for
    # reporting; reconstruct_arcs doesn't need it). Actually rally.py reads
    # only [u, v]; we attach frame as a separate vector through fits below.
    pts_2d = np.column_stack([obs[:, 0], obs[:, 1], frame_idx])
    print(f"[reconstruct] fitting bounces (reproj threshold "
          f"{args.reproj_threshold}px, spin locked off)")
    fits = reconstruct_arcs(
        t, pts_2d, q_sol, K, rvec, tvec,
        fps=args.fps, reproj_threshold=args.reproj_threshold,
        window_seconds=args.window_seconds,
        chain_arcs=not args.no_chain_arcs,
        chain_reproj_gate=args.chain_reproj_gate,
        min_vz_down=args.min_vz_down,
        lock_spin=not args.enable_spin,
        verbose=True,
    )
    print(f"[reconstruct] accepted {len(fits)} bounces:")
    for f_ in fits:
        print(f"  q={f_.q_idx} frame={f_.q_frame}  "
              f"bp=({f_.bounce_pt_world[0]:+.2f}, "
              f"{f_.bounce_pt_world[1]:+.2f}, "
              f"{f_.bounce_pt_world[2]:+.3f})  "
              f"v_pre=({f_.v_pre[0]:+.1f}, {f_.v_pre[1]:+.1f}, "
              f"{f_.v_pre[2]:+.1f})  "
              f"med_px={f_.median_px:.2f}")

    out_csv = args.rally_dir / "ball_traj_3D.csv"
    out = stitch_3d_csv(fits, args.fps, str(out_csv))
    if out is None:
        print("[reconstruct] no arcs accepted — nothing written")
        return 1
    print(f"[reconstruct] wrote {len(out)} rows -> {out_csv}")

    # Bounces metadata for downstream visualisation.
    bounces_csv = args.rally_dir / "bounces.csv"
    import csv as _csv
    with open(bounces_csv, "w", newline="") as fh:
        wr = _csv.writer(fh)
        wr.writerow(["frame", "x", "y", "z", "vx", "vy", "vz", "median_px"])
        for fit in fits:
            wr.writerow([
                fit.q_frame,
                f"{fit.bounce_pt_world[0]:.4f}",
                f"{fit.bounce_pt_world[1]:.4f}",
                f"{fit.bounce_pt_world[2]:.4f}",
                f"{fit.v_pre[0]:.3f}",
                f"{fit.v_pre[1]:.3f}",
                f"{fit.v_pre[2]:.3f}",
                f"{fit.median_px:.2f}",
            ])
    print(f"[reconstruct] wrote {len(fits)} bounces -> {bounces_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
