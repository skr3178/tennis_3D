#!/usr/bin/env python3
"""Paper-matched landing evaluation on Real TrackNet.

Difference vs eval_tracknet.py:
    * Landing error = L2 distance in the x-z (ground) plane between
      predicted (xyz_final.x, xyz_final.z) and the GT landing point obtained
      by back-projecting TrackNet's (u, v) label at status==1 frames through
      the calibrated camera to y=0. This is the metric reported in Table 4
      of "Where Is The Ball" (paper's 0.63 m on Real TrackNet).
    * Also reports landing accuracy T.acc at several thresholds (fraction of
      GT bounces whose predicted landing is within the threshold) and T.F1
      using the trained EoT head as the bounce/hit detector.
    * The height-only error from the original script is kept as a diagnostic
      under the name "landing_height_err" so we can see both at once.

Usage:
    /media/skr/storage/ten_bad/.venv/bin/python eval_tracknet_landing.py \
        --ckpt checkpoints_5k_v2/best.pt \
        --out_dir inference_output/tracknet_eval_5k_v2_landing
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _here)
sys.path.insert(0, _parent)
sys.path.insert(0, os.path.join(_parent, "TennisCourtDetector"))

from LSTM_Model.data.parameterization import pixel_to_plane_points
from LSTM_Model.pipeline import WhereIsTheBall
from eval_tracknet import (
    calibrate_camera_from_image,
    load_clip_labels,
    project_3d_to_2d,
    save_calibration_vis,
)


LANDING_THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0)


def tf1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints_5k_v2/best.pt")
    p.add_argument("--tracknet_dir", default="../TrackNet/datasets/trackNet/Dataset")
    p.add_argument("--court_model", default="../TennisCourtDetector/model_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="inference_output/tracknet_eval_5k_v2_landing")
    p.add_argument("--games", type=str, default=None,
                   help="Comma-separated game numbers to evaluate. Default: all.")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--eot_threshold", type=float, default=0.5,
                   help="Probability threshold for EoT -> predicted bounce/hit flag.")
    p.add_argument("--landing_threshold_m", type=float, default=0.5,
                   help="T.acc / T.F1 radius in metres (paper-style reporting).")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] Loading LSTM model from {args.ckpt} ...")
    net = WhereIsTheBall(hidden=64).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state"])
    net.eval()
    print(f"    Loaded epoch {state.get('epoch', '?')}")

    games = sorted([d for d in os.listdir(args.tracknet_dir)
                    if d.startswith("game") and
                    os.path.isdir(os.path.join(args.tracknet_dir, d))])
    if args.games:
        selected = set(f"game{g}" for g in args.games.split(","))
        games = [g for g in games if g in selected]
    print(f"    Games to evaluate: {games}")

    clip_results = []
    all_reproj = []
    all_landing_xz = []       # paper-style: x-z plane distance at GT bounces
    all_landing_height = []   # |y_pred| at GT bounces (old metric, kept)
    # Detection counters at the chosen landing threshold for T.F1
    tp = fp = fn = 0
    per_game_landing = {}
    total_clips = 0
    skipped_clips = 0

    print("\n[2/3] Processing games ...")
    for game in games:
        game_dir = os.path.join(args.tracknet_dir, game)
        clips = sorted([d for d in os.listdir(game_dir)
                        if d.startswith("Clip") and
                        os.path.isdir(os.path.join(game_dir, d))])
        if not clips:
            continue

        first_clip = os.path.join(game_dir, clips[0])
        first_frame = os.path.join(first_clip, "0000.jpg")
        if not os.path.exists(first_frame):
            frames_in = sorted(f for f in os.listdir(first_clip) if f.endswith(".jpg"))
            if not frames_in:
                skipped_clips += len(clips)
                continue
            first_frame = os.path.join(first_clip, frames_in[0])

        print(f"\n  {game}: calibrating from {os.path.basename(first_frame)} ...")
        cam = calibrate_camera_from_image(first_frame, args.court_model, args.device)
        if cam is None:
            print(f"  {game}: camera calibration failed, skipping")
            skipped_clips += len(clips)
            continue
        print(f"    Camera: fx={cam['fx']:.0f}, "
              f"reproj={cam['reprojection_error']:.2f}px, "
              f"keypoints={cam['num_keypoints']}")

        if args.vis:
            vis_dir = os.path.join(args.out_dir, "calibration_vis")
            os.makedirs(vis_dir, exist_ok=True)
            save_calibration_vis(first_frame, cam,
                                 os.path.join(vis_dir, f"{game}_calibration.png"))

        intrinsics = cam["intrinsics"]
        extrinsic = cam["extrinsic"]

        game_landing_errs = []

        for clip_name in clips:
            total_clips += 1
            clip_dir = os.path.join(game_dir, clip_name)
            labels = load_clip_labels(clip_dir)
            if labels is None:
                skipped_clips += 1
                continue

            uv = []
            statuses = []
            for lbl in labels:
                if lbl["visible"] and lbl["x"] > 0 and lbl["y"] > 0:
                    uv.append([lbl["x"], lbl["y"]])
                else:
                    uv.append([np.nan, np.nan])
                statuses.append(lbl["status"])
            uv = np.array(uv, dtype=np.float32)
            statuses = np.array(statuses, dtype=np.int64)

            valid_mask = ~np.isnan(uv[:, 0])
            if valid_mask.sum() < 10:
                skipped_clips += 1
                continue
            valid_idx = np.where(valid_mask)[0]
            uv_valid = uv[valid_idx]
            statuses_valid = statuses[valid_idx]

            P = pixel_to_plane_points(uv_valid, intrinsics, extrinsic,
                                      convention="opengl")
            if np.isnan(P).any():
                nan_frac = np.isnan(P).any(axis=1).mean()
                if nan_frac > 0.5:
                    skipped_clips += 1
                    continue
                good = ~np.isnan(P).any(axis=1)
                P = P[good]
                uv_valid = uv_valid[good]
                statuses_valid = statuses_valid[good]
            if len(P) < 10:
                skipped_clips += 1
                continue

            P_t = torch.from_numpy(P).unsqueeze(0).to(device)
            lengths = torch.tensor([len(P)], dtype=torch.long)
            with torch.no_grad():
                out = net(P_t, lengths=lengths)
            xyz = out["xyz_final"][0].cpu().numpy()      # (L, 3)
            eps = out["eps"][0, :, 0].cpu().numpy()      # (L,)

            # Reprojection error (diagnostic)
            uv_proj = project_3d_to_2d(xyz, intrinsics, extrinsic)
            reproj_err = np.sqrt(((uv_proj - uv_valid) ** 2).sum(axis=1))
            all_reproj.extend(reproj_err.tolist())

            # Paper-style landing metric at GT bounce frames (status==1)
            bounce_mask = statuses_valid == 1
            clip_landing = []
            if bounce_mask.any():
                pred_xz = xyz[bounce_mask][:, [0, 2]]             # (B, 2)
                gt_xz = P[bounce_mask][:, [0, 1]]                 # (p_g.x, p_g.z)
                dist_xz = np.sqrt(((pred_xz - gt_xz) ** 2).sum(axis=1))
                all_landing_xz.extend(dist_xz.tolist())
                all_landing_height.extend(np.abs(xyz[bounce_mask][:, 1]).tolist())
                clip_landing = dist_xz.tolist()
                game_landing_errs.extend(dist_xz.tolist())

                # Detection-style T.F1 at args.landing_threshold_m
                eps_bounce = eps[bounce_mask]
                for d, e in zip(dist_xz, eps_bounce):
                    if e >= args.eot_threshold and d <= args.landing_threshold_m:
                        tp += 1
                    else:
                        fn += 1
                # False positives: EoT fires but not at a GT bounce
                eot_pos = (eps >= args.eot_threshold)
                gt_pos = bounce_mask
                fp += int((eot_pos & ~gt_pos).sum())
            else:
                # Whole-clip false positives (no GT bounces at all)
                fp += int((eps >= args.eot_threshold).sum())

            clip_results.append({
                "game": game,
                "clip": clip_name,
                "num_frames": int(len(P)),
                "reproj_mean": float(reproj_err.mean()),
                "reproj_median": float(np.median(reproj_err)),
                "num_bounces": int(bounce_mask.sum()),
                "landing_xz_mean": (float(np.mean(clip_landing))
                                    if clip_landing else None),
                "landing_xz_median": (float(np.median(clip_landing))
                                      if clip_landing else None),
                "h_max": float(xyz[:, 1].max()),
                "h_min": float(xyz[:, 1].min()),
            })
            land_txt = (f"land_xz={np.mean(clip_landing):.2f}m"
                        if clip_landing else "land_xz=-")
            print(f"    {clip_name}: n={len(P)}, "
                  f"reproj={reproj_err.mean():.1f}px, "
                  f"bounces={bounce_mask.sum()}, {land_txt}")

        if game_landing_errs:
            per_game_landing[game] = {
                "n": len(game_landing_errs),
                "mean": float(np.mean(game_landing_errs)),
                "median": float(np.median(game_landing_errs)),
            }

    # Summary
    print(f"\n{'=' * 60}")
    print("[3/3] EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total clips: {total_clips}, evaluated: {len(clip_results)}, "
          f"skipped: {skipped_clips}")
    print(f"Bounces (GT status==1): {len(all_landing_xz)}")

    if all_reproj:
        r = np.array(all_reproj)
        print(f"\nReprojection error ({len(r)} frames):")
        print(f"  mean={r.mean():.2f}px  median={np.median(r):.2f}px  "
              f"<5px={100*(r<5).mean():.1f}%  <10px={100*(r<10).mean():.1f}%")

    summary = {
        "model_epoch": state.get("epoch"),
        "ckpt": args.ckpt,
        "total_clips": total_clips,
        "evaluated_clips": len(clip_results),
        "skipped_clips": skipped_clips,
        "num_bounces": len(all_landing_xz),
        "reproj_mean_px": float(np.mean(all_reproj)) if all_reproj else None,
        "reproj_median_px": float(np.median(all_reproj)) if all_reproj else None,
        "landing_xz_mean_m": None,
        "landing_xz_median_m": None,
        "landing_xz_std_m": None,
        "landing_height_mean_m": None,
        "landing_height_median_m": None,
        "t_acc": {},
        "t_f1": None,
        "eot_threshold": args.eot_threshold,
        "landing_threshold_m": args.landing_threshold_m,
        "per_game": per_game_landing,
        "clips": clip_results,
    }

    if all_landing_xz:
        lx = np.array(all_landing_xz)
        lh = np.array(all_landing_height)
        print(f"\nLanding error on x-z ground plane (paper metric):")
        print(f"  mean  ={lx.mean():.3f} m")
        print(f"  median={np.median(lx):.3f} m")
        print(f"  std   ={lx.std():.3f} m")
        print(f"  paper Real TrackNet avg = 0.63 m  (Table 4)")
        print(f"\nLanding-height error (diagnostic, |y_pred| at GT bounce):")
        print(f"  mean  ={lh.mean():.3f} m   median={np.median(lh):.3f} m")
        print(f"\nT.acc at thresholds:")
        for th in LANDING_THRESHOLDS_M:
            frac = 100 * (lx <= th).mean()
            summary["t_acc"][f"<{th}m"] = float(frac)
            print(f"  <= {th:.2f} m : {frac:.1f}%")
        f1 = tf1_from_counts(tp, fp, fn)
        summary["t_f1"] = float(f1)
        summary["tp"], summary["fp"], summary["fn"] = tp, fp, fn
        print(f"\nT.F1 @ (eps>={args.eot_threshold}, dist<={args.landing_threshold_m}m):")
        print(f"  TP={tp}  FP={fp}  FN={fn}  F1={f1:.3f}")
        summary["landing_xz_mean_m"] = float(lx.mean())
        summary["landing_xz_median_m"] = float(np.median(lx))
        summary["landing_xz_std_m"] = float(lx.std())
        summary["landing_height_mean_m"] = float(lh.mean())
        summary["landing_height_median_m"] = float(np.median(lh))

        print(f"\nPer-game landing error:")
        for g in sorted(per_game_landing):
            v = per_game_landing[g]
            print(f"  {g}: n={v['n']:3d}  mean={v['mean']:.3f} m  "
                  f"median={v['median']:.3f} m")

    out_path = os.path.join(args.out_dir, "tracknet_eval_landing_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
