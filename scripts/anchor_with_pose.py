"""Use PromptHMR wrist positions as additional 3D anchors to fill gaps in
ball_traj_3D.csv that the bounce-anchored physics couldn't reach.

Pipeline:
  1. Read impacts.csv (candidate racket-impact frames + wrist pid/joint)
  2. For each impact, compute wrist 3D in our court frame:
       a. Take wrist 3D in PromptHMR world (FK already done)
       b. Transform to PromptHMR camera frame
       c. Scale by focal_ours/focal_phmr (corrects depth bias from
          PromptHMR's wider-FOV intrinsics estimate vs our calibration)
       d. Transform from our camera frame to court frame (inverse of rvec,tvec)
  3. Stitch bounces + impact anchors + linear interpolation between adjacent
     anchors over WASB-visible frames into ball_traj_3D_pose.csv

Usage:
    python scripts/anchor_with_pose.py data/game1_clip1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_3d.calibration.utils import get_K, read_camera_info
from tennis_3d.rally.geometry import get_transform


J_LWRIST = 20
J_RWRIST = 21
J_LANKLE = 7
J_RANKLE = 8


def ray_at_height(K, rvec, tvec, u, v, court_z):
    """Cast a ray from camera through pixel (u, v); return the 3D court-frame
    point on that ray at given court z height. None if ray is ~parallel to z."""
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
    tvec = np.asarray(tvec).reshape(3)
    Kinv = np.linalg.inv(K)
    ray_cam = Kinv @ np.array([u, v, 1.0])
    # Camera origin in court frame: o_court = -R.T @ tvec
    Rt = R.T
    o_court = -Rt @ tvec
    # Ray direction in court frame: d_court = R.T @ ray_cam
    d_court = Rt @ ray_cam
    if abs(d_court[2]) < 1e-6:
        return None
    # Find lambda s.t. o_court + lambda * d_court has z = court_z
    lam = (court_z - o_court[2]) / d_court[2]
    if lam <= 0:
        return None
    return o_court + lam * d_court


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--max-anchor-gap", type=int, default=80,
                   help="Max frame gap between two anchors to interpolate over")
    args = p.parse_args()

    rd: Path = args.rally_dir
    out_csv = args.out_csv or (rd / "ball_traj_3D_pose.csv")

    # Our camera (court frame)
    rvec, tvec, focal_ours, h, w = read_camera_info(rd / "camera.yaml")
    K_ours = get_K(focal_ours, h, w)
    T_court_to_cam = get_transform(rvec.flatten(), tvec.flatten())  # court->cam
    T_cam_to_court = np.linalg.inv(T_court_to_cam)

    # PromptHMR pose
    pose = np.load(rd / "pose_joints.npz", allow_pickle=True)
    joints_3d_world = pose["joints_3d_world"]    # (P, N, 55, 3)
    pids = list(pose["pids"])
    Rcw = pose["Rcw"]                            # (N, 3, 3) world->phmr-cam
    Tcw = pose["Tcw"]                            # (N, 3)

    impacts_pids = pd.read_csv(rd / "impacts.csv")["pid"].tolist()

    impacts = pd.read_csv(rd / "impacts.csv")
    print(f"[anchor] {len(impacts)} candidate impacts")

    # For each impact: ball 3D = ray from our camera through (ball_u, ball_v)
    # at height z = (wrist - ankle) above ground (scale-invariant).
    impact_anchors = []
    for _, r in impacts.iterrows():
        f = int(r["frame"])
        pid = int(r["pid"])
        is_left = r["joint"] == "L_wrist"
        jw = J_LWRIST if is_left else J_RWRIST
        ja = J_LANKLE if is_left else J_RANKLE
        p_idx = pids.index(pid)
        wrist_w = joints_3d_world[p_idx, f, jw]
        ankle_w = joints_3d_world[p_idx, f, ja]
        # Same-side ankle defines floor reference. PromptHMR world axes don't
        # match ours, so use distance-along-PromptHMR-vertical: their scene
        # has *some* up axis; the magnitude of the (wrist-ankle) vector
        # along that axis is what we need. Without knowing PromptHMR's up,
        # use the largest-magnitude component as a proxy.
        diff = wrist_w - ankle_w
        # Better: use the camera-frame Y axis (downward in image) as proxy
        # for height: cam Y is roughly -world up.
        diff_cam = Rcw[f] @ diff
        h_above_ground = float(-diff_cam[1])      # cam Y down -> negate for up
        # Sanity clamp: tennis racket reaches ~ 0.5 .. 3.5 m
        h_above_ground = float(np.clip(h_above_ground, 0.3, 4.0))
        u = float(r["ball_u"]); v = float(r["ball_v"])
        ball_court = ray_at_height(K_ours, rvec, tvec, u, v, h_above_ground)
        if ball_court is None:
            continue
        impact_anchors.append({
            "frame": f, "x": float(ball_court[0]), "y": float(ball_court[1]),
            "z": float(ball_court[2]), "kind": "racket"
        })

    # Existing bounce anchors
    bounces = pd.read_csv(rd / "bounces.csv")
    bounce_anchors = []
    for _, r in bounces.iterrows():
        bounce_anchors.append({
            "frame": int(r["frame"]), "x": float(r["x"]),
            "y": float(r["y"]), "z": float(r["z"]), "kind": "bounce"
        })

    anchors = sorted(impact_anchors + bounce_anchors, key=lambda a: a["frame"])
    print(f"[anchor] total anchors: {len(anchors)} "
          f"(bounces={len(bounce_anchors)}, racket={len(impact_anchors)})")
    print(f"[anchor] anchor frames + kind:")
    for a in anchors:
        print(f"  f={a['frame']:3d}  ({a['x']:+.2f}, {a['y']:+.2f}, "
              f"{a['z']:+.2f})  {a['kind']}")

    # Existing physics-fit 3D rows (from reconstruct.py)
    have_phys = pd.read_csv(rd / "ball_traj_3D.csv")
    phys_idx = set(have_phys["idx"].tolist())

    # WASB visible frames
    wasb = pd.read_csv(rd / "wasb_full.csv")
    visible_frames = wasb[wasb["visible"] == 1]["frame"].tolist()

    out = {}
    for _, r in have_phys.iterrows():
        out[int(r["idx"])] = (float(r["x"]), float(r["y"]),
                              float(r["z"]), "phys")

    # Interpolate between adjacent anchors over visible WASB frames in the gap
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        gap = b["frame"] - a["frame"]
        if gap > args.max_anchor_gap or gap < 2:
            continue
        for f in visible_frames:
            if not (a["frame"] < f < b["frame"]):
                continue
            if f in out:
                continue                              # physics already covered
            t = (f - a["frame"]) / gap
            x = (1 - t) * a["x"] + t * b["x"]
            y = (1 - t) * a["y"] + t * b["y"]
            z = (1 - t) * a["z"] + t * b["z"]
            out[f] = (x, y, z, "interp")

    # Add anchor frames themselves
    for a in anchors:
        if a["frame"] not in out:
            out[a["frame"]] = (a["x"], a["y"], a["z"], a["kind"])

    rows = sorted(out.items())
    df = pd.DataFrame(
        [{"idx": k, "x": v[0], "y": v[1], "z": v[2], "src": v[3]}
         for k, v in rows])
    df.to_csv(out_csv, index=False)
    n_phys = (df["src"] == "phys").sum()
    n_interp = (df["src"] == "interp").sum()
    n_anchor = len(df) - n_phys - n_interp
    print(f"[anchor] wrote {out_csv}  rows={len(df)}  "
          f"(phys={n_phys}, interp={n_interp}, anchor={n_anchor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
