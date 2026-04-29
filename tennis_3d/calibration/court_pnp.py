"""Detect court keypoints with TennisCourtDetector and solve for camera pose.

Pipeline:
    1. Pick a frame from the rally video (default: median sample of N frames
       to dodge motion blur on a single frame).
    2. Run TennisCourtDetector to get the 14 image-space keypoints.
    3. With principal point at image center, search over focal length f and
       solvePnP for each. Pick the f with minimum reprojection error, then
       refine pose with solvePnPRefineLM.
    4. Write camera.yaml in tt3d format and a calib_check.png overlay.

This file expects to be called via scripts/calibrate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tennis_3d.calibration.court_keypoints_3d import COURT_KEYPOINTS_3D
from tennis_3d.calibration.utils import get_K, write_camera_info

TCD_DIR = Path("/media/skr/storage/ten_bad/TennisCourtDetector")


def _import_tcd():
    """Import TCD modules without permanently shadowing project namespaces."""
    sys.path.insert(0, str(TCD_DIR))
    try:
        from tracknet import BallTrackerNet
        from postprocess import postprocess, refine_kps
        from homography import get_trans_matrix, refer_kps
    finally:
        sys.path.pop(0)
    return BallTrackerNet, postprocess, refine_kps, get_trans_matrix, refer_kps


def sample_frames(video_path: Path, n: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")
    idxs = np.linspace(0, total - 1, num=n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Failed to sample any frames")
    return frames


def detect_keypoints(
    frames: List[np.ndarray],
    model_path: Path,
    use_refine: bool = True,
    use_homography: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (kps, mask) where kps is (14, 2) median-of-frames image
    coordinates and mask is (14,) bool indicating valid detections.
    """
    BallTrackerNet, postprocess, refine_kps, get_trans_matrix, refer_kps = _import_tcd()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BallTrackerNet(out_channels=15).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    OUT_W, OUT_H = 640, 360
    per_frame: List[List[Tuple[Optional[float], Optional[float]]]] = []
    with torch.no_grad():
        for image in frames:
            img = cv2.resize(image, (OUT_W, OUT_H))
            inp = (img.astype(np.float32) / 255.0)
            inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(device)
            out = model(inp.float())[0]
            pred = torch.sigmoid(out).cpu().numpy()

            points: List[Tuple[Optional[float], Optional[float]]] = []
            for k in range(14):
                heat = (pred[k] * 255).astype(np.uint8)
                x_pred, y_pred = postprocess(heat, low_thresh=170, max_radius=25)
                if use_refine and k not in (8, 12, 9) and x_pred and y_pred:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
                points.append((x_pred, y_pred))

            if use_homography:
                M = get_trans_matrix(points)
                if M is not None:
                    proj = cv2.perspectiveTransform(refer_kps, M)
                    points = [tuple(np.squeeze(p)) for p in proj]
            per_frame.append(points)

    # Stack into (N, 14, 2) with NaN for missing
    arr = np.full((len(per_frame), 14, 2), np.nan, dtype=np.float64)
    for fi, pts in enumerate(per_frame):
        for k, (x, y) in enumerate(pts):
            if x is not None and y is not None:
                arr[fi, k, 0] = float(x)
                arr[fi, k, 1] = float(y)

    kps = np.nanmedian(arr, axis=0)              # (14, 2)
    mask = ~np.any(np.isnan(kps), axis=1)         # (14,)
    return kps, mask


def reprojection_error(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> Tuple[float, np.ndarray]:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.median(err)), err


def solve_pose_for_f(
    obj_pts: np.ndarray, img_pts: np.ndarray, f: float, w: int, h: int
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    K = get_K(f, h, w)
    flag = cv2.SOLVEPNP_SQPNP if obj_pts.shape[0] >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None, flags=flag)
    if not ok:
        return None
    med, _ = reprojection_error(obj_pts, img_pts, rvec, tvec, K)
    return rvec, tvec, med


def search_focal(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    w: int,
    h: int,
    f_range: Tuple[float, float] = (600.0, 3500.0),
    coarse_step: float = 50.0,
    fine_window: float = 100.0,
    fine_step: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # Coarse sweep
    best = None
    for f in np.arange(f_range[0], f_range[1] + 1e-6, coarse_step):
        out = solve_pose_for_f(obj_pts, img_pts, float(f), w, h)
        if out is None:
            continue
        rvec, tvec, err = out
        if best is None or err < best[2]:
            best = (rvec, tvec, err, float(f))
    if best is None:
        raise RuntimeError("PnP failed across all focal lengths in coarse sweep")
    # Fine sweep around the coarse best
    f0 = best[3]
    lo = max(f_range[0], f0 - fine_window)
    hi = min(f_range[1], f0 + fine_window)
    for f in np.arange(lo, hi + 1e-6, fine_step):
        out = solve_pose_for_f(obj_pts, img_pts, float(f), w, h)
        if out is None:
            continue
        rvec, tvec, err = out
        if err < best[2]:
            best = (rvec, tvec, err, float(f))
    rvec, tvec, err, f_best = best

    # Final LM refinement
    K = get_K(f_best, h, w)
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, None, rvec, tvec)
    err_final, _ = reprojection_error(obj_pts, img_pts, rvec, tvec, K)
    return rvec, tvec, f_best, err_final


def draw_calib_overlay(
    frame: np.ndarray,
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    img = frame.copy()
    proj, _ = cv2.projectPoints(COURT_KEYPOINTS_3D, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    # Draw all 14 reprojections (yellow)
    for p in proj:
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 6, (0, 255, 255), 2)
    # Draw detected (green) and connect to reprojection (red line) for the
    # subset actually used in PnP.
    proj_used, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    proj_used = proj_used.reshape(-1, 2)
    for det, rep in zip(img_pts, proj_used):
        cv2.circle(img, (int(round(det[0])), int(round(det[1]))), 4, (0, 255, 0), -1)
        cv2.line(
            img,
            (int(round(det[0])), int(round(det[1]))),
            (int(round(rep[0])), int(round(rep[1]))),
            (0, 0, 255),
            1,
        )
    return img
