"""Run SMPL-X FK on PromptHMR output to get per-frame 3D joint positions for
the main 2 players, project wrists to image, and detect candidate racket-impact
frames where the WASB ball 2D position is close to a player's wrist 2D.

Saves:
  <rally_dir>/pose_joints.npz   - per-frame 3D joints in camera frame
  <rally_dir>/impacts.csv       - candidate impact frames

Usage:
    python scripts/extract_pose_anchors.py data/game1_clip1 \
        --phmr-pkl /media/skr/storage/ten_bad/PromptHMR/results/S_Original_HL_clip_cropped/results.pkl \
        --smplx-model /home/skr/Downloads/video2robot/body_models/smplx/SMPLX_NEUTRAL.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import smplx
import torch


# SMPL-X joint indices for body (first 22):
#   0: pelvis,    1-2: left/right hip,   3: spine1,
#   4-5: left/right knee, 6: spine2, 7-8: l/r ankle, 9: spine3,
#   10-11: l/r foot, 12: neck, 13-14: l/r collar, 15: head,
#   16-17: l/r shoulder, 18-19: l/r elbow, 20-21: l/r wrist
J_LWRIST = 20
J_RWRIST = 21
J_LANKLE = 7
J_RANKLE = 8


def pick_main_players(people: dict, im_w: int = 1280, im_h: int = 720) -> list[int]:
    """Pick near and far player. Near = biggest bbox in lower half of frame.
    Far = bbox in upper half closest to horizontal centre (court central column)."""
    near_cands, far_cands = [], []
    for pid, p in people.items():
        bb = p["bboxes"]
        det = p["detected"].astype(bool)
        if det.sum() < 100:
            continue
        bb = bb[det]
        h = (bb[:, 3] - bb[:, 1]).mean()
        cx = ((bb[:, 0] + bb[:, 2]) / 2).mean()
        cy = ((bb[:, 1] + bb[:, 3]) / 2).mean()
        if cy > im_h / 2:
            near_cands.append((h, pid, cx, cy))            # bigger = closer
        else:
            far_cands.append((-abs(cx - im_w / 2), pid, h, cx, cy))   # central = main
    near_cands.sort(reverse=True)
    far_cands.sort(reverse=True)
    print("[pose] near candidates:", [(p, f"h={h:.0f}", f"cx={cx:.0f}", f"cy={cy:.0f}")
                                       for h, p, cx, cy in near_cands[:3]])
    print("[pose] far candidates:", [(p, f"h={h:.0f}", f"cx={cx:.0f}", f"cy={cy:.0f}")
                                      for _, p, h, cx, cy in far_cands[:3]])
    return [near_cands[0][1], far_cands[0][1]]


def run_fk(smplx_world: dict, betas_per_frame: np.ndarray, model) -> np.ndarray:
    """Run SMPL-X forward kinematics. Returns joints (N, 55, 3) in world frame."""
    pose = smplx_world["pose"]                      # (N, 165)
    trans = smplx_world["trans"]                    # (N, 3)
    N = pose.shape[0]

    pose_t = torch.from_numpy(pose).float()
    trans_t = torch.from_numpy(trans).float()
    betas_t = torch.from_numpy(betas_per_frame).float()

    # SMPL-X axis-angle layout (from PromptHMR pipeline/world.py):
    # [0:3] global orient
    # [3:66] body pose (21 joints)
    # [66:69] jaw
    # [69:72] leye
    # [72:75] reye
    # [75:120] left hand (15 joints)
    # [120:165] right hand
    out = model(
        global_orient=pose_t[:, :3],
        body_pose=pose_t[:, 3:66],
        jaw_pose=pose_t[:, 66:69],
        leye_pose=pose_t[:, 69:72],
        reye_pose=pose_t[:, 72:75],
        left_hand_pose=pose_t[:, 75:120],
        right_hand_pose=pose_t[:, 120:165],
        betas=betas_t,
        transl=trans_t,
        return_full_pose=False,
    )
    return out.joints[:, :55, :].detach().cpu().numpy()       # (N, 55, 3)


def world_to_image(joints_world: np.ndarray, Rcw: np.ndarray, Tcw: np.ndarray,
                   focal: float, center: np.ndarray) -> np.ndarray:
    """Project world-frame joints to image. joints_world: (N, J, 3),
    Rcw: (N, 3, 3) world->camera rotation, Tcw: (N, 3) translation."""
    Xc = np.einsum("nij,nkj->nki", Rcw, joints_world) + Tcw[:, None, :]
    u = focal * Xc[..., 0] / Xc[..., 2] + center[0]
    v = focal * Xc[..., 1] / Xc[..., 2] + center[1]
    return np.stack([u, v], axis=-1)              # (N, J, 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--phmr-pkl", type=Path, required=True)
    p.add_argument("--smplx-model", type=Path, required=True)
    p.add_argument("--impact-px", type=float, default=40.0,
                   help="Max distance (px) from ball to wrist for an impact")
    args = p.parse_args()

    rd: Path = args.rally_dir
    print(f"[pose] loading PromptHMR: {args.phmr_pkl}")
    d = joblib.load(args.phmr_pkl)
    cw = d["camera_world"]
    Rcw = cw["Rcw"]                                # (N, 3, 3) world->camera
    Tcw = cw["Tcw"]                                # (N, 3)
    focal = float(np.atleast_1d(cw["img_focal"]).flatten()[0])
    center = np.atleast_1d(cw["img_center"]).flatten()[:2]
    print(f"[pose] PromptHMR cam: focal={focal:.1f}  center={center}")

    main_pids = pick_main_players(d["people"])
    print(f"[pose] using players {main_pids}")

    print(f"[pose] loading SMPL-X model: {args.smplx_model}")
    # smplx.create expects model_path = parent of 'smplx/' subdir, OR direct file
    model = smplx.create(
        model_path=str(args.smplx_model.parent.parent),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=767,
    )

    all_joints_2d = {}                             # pid -> (N, 55, 2)
    all_joints_3d_world = {}                       # pid -> (N, 55, 3)
    for pid in main_pids:
        p_data = d["people"][pid]
        betas = p_data["smplx_betas"]              # (767, 10)
        sw = p_data["smplx_world"]
        joints_3d = run_fk(sw, betas, model)
        joints_2d = world_to_image(joints_3d, Rcw, Tcw, focal, center)
        all_joints_3d_world[pid] = joints_3d
        all_joints_2d[pid] = joints_2d
        print(f"[pose] pid={pid}: 3D joints shape={joints_3d.shape}  "
              f"l_wrist sample0=({joints_3d[0, J_LWRIST]})  "
              f"r_wrist 2D sample0=({joints_2d[0, J_RWRIST]})")

    np.savez(rd / "pose_joints.npz",
             pids=np.array(main_pids),
             joints_3d_world=np.stack([all_joints_3d_world[p] for p in main_pids]),
             joints_2d=np.stack([all_joints_2d[p] for p in main_pids]),
             Rcw=Rcw, Tcw=Tcw, focal=focal, center=center)

    # For each visible frame, find the closest wrist (across both players,
    # both hands). An impact is a LOCAL MINIMUM of that distance, below the
    # `impact-px` threshold, with a refractory window to avoid duplicates.
    from scipy.signal import find_peaks

    ball = pd.read_csv(rd / "wasb_full.csv")
    n_frames = len(ball)
    closest = np.full(n_frames, np.inf)
    closest_pid = np.full(n_frames, -1, dtype=int)
    closest_joint = np.full(n_frames, "", dtype=object)
    closest_wrist = np.full((n_frames, 2), np.nan)
    visible = ball["visible"].to_numpy() == 1
    bx_all = ball["x"].to_numpy()
    by_all = ball["y"].to_numpy()
    for f in range(n_frames):
        if not visible[f]:
            continue
        bx, by = bx_all[f], by_all[f]
        for pid in main_pids:
            for jname, jidx in (("L_wrist", J_LWRIST), ("R_wrist", J_RWRIST)):
                wx, wy = all_joints_2d[pid][f, jidx]
                if not (np.isfinite(wx) and np.isfinite(wy)):
                    continue
                d = float(np.hypot(bx - wx, by - wy))
                if d < closest[f]:
                    closest[f] = d
                    closest_pid[f] = pid
                    closest_joint[f] = jname
                    closest_wrist[f] = [wx, wy]

    finite = np.isfinite(closest)
    # Find local minima in -closest (so negative-prominence becomes positive)
    minus = np.where(finite, -closest, -1e9)
    peaks, _ = find_peaks(minus, distance=20, height=-args.impact_px,
                          prominence=20.0)
    print(f"[pose] {len(peaks)} local-minimum impact candidates "
          f"(threshold={args.impact_px}px)")

    out_rows = []
    for f in peaks:
        out_rows.append({
            "frame": int(f),
            "pid": int(closest_pid[f]),
            "joint": closest_joint[f],
            "ball_u": float(bx_all[f]), "ball_v": float(by_all[f]),
            "wrist_u": float(closest_wrist[f, 0]),
            "wrist_v": float(closest_wrist[f, 1]),
            "dist_px": float(closest[f]),
        })
    df = pd.DataFrame(out_rows)
    df.to_csv(rd / "impacts.csv", index=False)
    print(f"[pose] wrote {rd / 'impacts.csv'} with {len(df)} candidate impacts")
    if len(df):
        print(f"[pose] impact frames: {df['frame'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
