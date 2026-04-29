"""Milestone 1 CLI: estimate camera pose for a fixed-camera rally clip.

Usage:
    python scripts/calibrate.py data/game1_clip1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `tennis_3d` importable when run from repo root or scripts/.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d.calibration.court_keypoints_3d import COURT_KEYPOINTS_3D
from tennis_3d.calibration.court_pnp import (
    detect_keypoints,
    draw_calib_overlay,
    sample_frames,
    search_focal,
)
from tennis_3d.calibration.utils import get_K, write_camera_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rally_dir", type=Path)
    parser.add_argument(
        "--video", default=None, help="Override path; default <rally_dir>/rally.mp4"
    )
    parser.add_argument(
        "--tcd-model",
        default="/media/skr/storage/ten_bad/TennisCourtDetector/model_best.pt",
    )
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip TCD refine_kps post-processing",
    )
    parser.add_argument(
        "--no-homography",
        action="store_true",
        help="Skip TCD homography post-processing",
    )
    args = parser.parse_args()

    rally_dir: Path = args.rally_dir
    rally_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video) if args.video else rally_dir / "rally.mp4"
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        return 2

    print(f"[calibrate] sampling {args.n_frames} frames from {video}")
    frames = sample_frames(video, n=args.n_frames)
    h, w = frames[0].shape[:2]
    print(f"[calibrate] frame size: {w}x{h}")

    print(f"[calibrate] running TennisCourtDetector ({args.tcd_model})")
    kps, mask = detect_keypoints(
        frames,
        Path(args.tcd_model),
        use_refine=not args.no_refine,
        use_homography=not args.no_homography,
    )
    n_valid = int(mask.sum())
    print(f"[calibrate] valid keypoints: {n_valid}/14")
    if n_valid < 6:
        print("ERROR: need at least 6 valid keypoints for stable PnP", file=sys.stderr)
        return 3

    obj_pts = COURT_KEYPOINTS_3D[mask].astype(np.float64)
    img_pts = kps[mask].astype(np.float64)

    print("[calibrate] searching focal length + solving PnP")
    rvec, tvec, f, err = search_focal(obj_pts, img_pts, w=w, h=h)
    print(f"[calibrate] best f={f:.1f}px  median reproj err={err:.2f}px")
    print(f"[calibrate] rvec={rvec.flatten()}  tvec={tvec.flatten()}")

    yaml_path = rally_dir / "camera.yaml"
    write_camera_info(yaml_path, rvec, tvec, f, h, w)
    print(f"[calibrate] wrote {yaml_path}")

    K = get_K(f, h, w)
    overlay = draw_calib_overlay(frames[len(frames) // 2], obj_pts, img_pts, rvec, tvec, K)
    overlay_path = rally_dir / "calib_check.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"[calibrate] wrote {overlay_path}")

    if err > 5.0:
        print(f"WARN: median reproj error {err:.2f}px exceeds 5px target")
        return 1
    print("[calibrate] M1 success: median reproj error within target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
