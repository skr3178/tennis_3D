"""Export everything the browser viewer needs as a single JSON:
  - Court dimensions
  - Per-frame ball position in court frame (from ball_traj_3D_pose.csv)
  - Per-frame, per-player skeleton joints in court frame
    (PromptHMR world -> phmr cam -> our cam (per-player scale) -> court)
  - SMPL body bone connectivity

Usage:
    python scripts/export_for_viewer.py data/game1_clip1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_3d.calibration.utils import get_K, read_camera_info
from tennis_3d.rally.geometry import get_transform


# SMPL body parents for the first 22 joints
PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]
BONES = [(c, p) for c, p in enumerate(PARENTS) if p != -1]
J_LANKLE = 7
J_RANKLE = 8


def ray_at_height(K, rvec, tvec, u, v, court_z):
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
    tvec = np.asarray(tvec).reshape(3)
    Kinv = np.linalg.inv(K)
    ray_cam = Kinv @ np.array([u, v, 1.0])
    Rt = R.T
    o_court = -Rt @ tvec
    d_court = Rt @ ray_cam
    if abs(d_court[2]) < 1e-6:
        return None
    lam = (court_z - o_court[2]) / d_court[2]
    if lam <= 0:
        return None
    return o_court + lam * d_court


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=float, default=50.0)
    args = p.parse_args()

    rd: Path = args.rally_dir
    out_json = args.out or (rd / "viewer_data.json")

    rvec, tvec, focal_ours, h, w = read_camera_info(rd / "camera.yaml")
    T_court_to_cam = get_transform(rvec.flatten(), tvec.flatten())
    T_cam_to_court = np.linalg.inv(T_court_to_cam)
    R_c2c = T_cam_to_court[:3, :3]
    t_c2c = T_cam_to_court[:3, 3]

    pose = np.load(rd / "pose_joints.npz", allow_pickle=True)
    joints_w = pose["joints_3d_world"]              # (P, N, 55, 3)
    pids = list(pose["pids"])
    Rcw = pose["Rcw"]                               # (N, 3, 3)
    Tcw = pose["Tcw"]                               # (N, 3)
    P, N, J, _ = joints_w.shape

    # Per-player scale: median over (frame, ankle) of the s that puts that
    # ankle on the floor (court z = 0).
    scales = []
    for p_idx in range(P):
        s_obs = []
        for f in range(N):
            for j in (J_LANKLE, J_RANKLE):
                a_w = joints_w[p_idx, f, j]
                a_c = Rcw[f] @ a_w + Tcw[f]
                denom = float(R_c2c[2] @ a_c)
                if abs(denom) < 1e-3:
                    continue
                s_obs.append(-float(t_c2c[2]) / denom)
        s = float(np.median(s_obs))
        scales.append(s)
        print(f"[viewer] pid={pids[p_idx]}  scale={s:.3f}  (n={len(s_obs)})")

    # Transform body joints (first 22) to court frame
    joints_court = np.zeros((P, N, 22, 3), dtype=np.float32)
    for p_idx in range(P):
        s = scales[p_idx]
        for f in range(N):
            jw = joints_w[p_idx, f, :22]            # (22, 3)
            jc_phmr = (Rcw[f] @ jw.T).T + Tcw[f]    # (22, 3) PromptHMR cam
            jc_ours = jc_phmr * s                   # our cam frame
            jcourt = (R_c2c @ jc_ours.T).T + t_c2c  # court frame
            joints_court[p_idx, f] = jcourt

    # Ball trajectory in court frame
    df = pd.read_csv(rd / "ball_traj_3D_pose.csv")
    has_src = "src" in df.columns
    by_frame = {int(r["idx"]): (float(r["x"]), float(r["y"]), float(r["z"]),
                                r["src"] if has_src else "phys")
                for _, r in df.iterrows()}
    ball_pos = []
    ball_src = []
    for f in range(N):
        if f in by_frame:
            x, y, z, src = by_frame[f]
            ball_pos.append([x, y, z])
            ball_src.append(src)
        else:
            ball_pos.append(None)
            ball_src.append(None)

    # ALSO transform ball positions into PromptHMR world frame, so the viewer
    # can overlay it on the world4d.glb scene. Per-frame transform:
    #   ball_court -> ball_our_cam (court->cam via rvec, tvec)
    #   ball_our_cam -> ball_phmr_cam (divide by median scale)
    #   ball_phmr_cam -> ball_phmr_world (apply Rwc[i], Twc[i] per frame)
    R_court_to_cam = T_court_to_cam[:3, :3]
    t_court_to_cam = T_court_to_cam[:3, 3]
    median_scale = float(np.median([s for s in scales]))    # avg over players
    print(f"[viewer] median scale used for ball xform: {median_scale:.3f}")
    # Inverse of Rcw/Tcw is Rwc/Twc
    Rwc = np.transpose(Rcw, (0, 2, 1))
    Twc = -np.einsum("nij,nj->ni", Rwc, Tcw)

    ball_phmr = []
    for f in range(N):
        bp_court = ball_pos[f]
        if bp_court is None:
            ball_phmr.append(None)
            continue
        bp_court = np.array(bp_court, dtype=np.float64)
        bp_our_cam = R_court_to_cam @ bp_court + t_court_to_cam
        bp_phmr_cam = bp_our_cam / median_scale
        bp_phmr_world = Rwc[f] @ bp_phmr_cam + Twc[f]
        ball_phmr.append(bp_phmr_world.tolist())

    out = {
        "fps": args.fps,
        "n_frames": int(N),
        "court": {
            "doubles_half_w": 5.485,
            "singles_half_w": 4.115,
            "half_l": 11.885,
            "service_line": 6.40,
            "net_height": 0.914,
        },
        "bones": BONES,
        "median_scale": median_scale,
        "players": [
            {
                "pid": int(pids[p_idx]),
                "scale": float(scales[p_idx]),
                "joints": joints_court[p_idx].tolist(),     # (N, 22, 3)
            }
            for p_idx in range(P)
        ],
        "ball": ball_pos,                                   # court frame
        "ball_phmr": ball_phmr,                             # PromptHMR world
        "ball_src": ball_src,
    }
    with open(out_json, "w") as fh:
        json.dump(out, fh)
    print(f"[viewer] wrote {out_json}  size_mb={out_json.stat().st_size/1e6:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
