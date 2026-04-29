"""Reproject the 3D ball trajectory onto rally.mp4 to validate M4 visually.

Draws on every frame:
  * green dot   : 2D ball detection from ball_traj_2D.csv
  * cyan ring   : 3D-then-projected ball position from ball_traj_3D.csv
  * yellow X    : reconstructed bounce points (z=0 in world)
  * red lines   : court keypoint reprojection (M1 sanity)

H.264 output (avc1).

Usage:
    python scripts/render_overlay.py data/game1_clip1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d.calibration.court_keypoints_3d import COURT_KEYPOINTS_3D
from tennis_3d.calibration.utils import get_K, read_camera_info
from tennis_3d.traj_seg.utils import read_traj


def project_points(pts3d, rvec, tvec, K):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    return proj.reshape(-1, 2)


def draw_court_lines(img, rvec, tvec, K):
    # Singles court rectangle, net, both service lines, center service line
    pts3d_pairs = [
        # baselines
        ((-4.115, +11.885, 0.0), (+4.115, +11.885, 0.0)),
        ((-4.115, -11.885, 0.0), (+4.115, -11.885, 0.0)),
        # singles sidelines
        ((-4.115, +11.885, 0.0), (-4.115, -11.885, 0.0)),
        ((+4.115, +11.885, 0.0), (+4.115, -11.885, 0.0)),
        # service lines
        ((-4.115, +6.40, 0.0), (+4.115, +6.40, 0.0)),
        ((-4.115, -6.40, 0.0), (+4.115, -6.40, 0.0)),
        # center service line
        ((0.0, +6.40, 0.0), (0.0, -6.40, 0.0)),
        # net
        ((-5.485, 0.0, 0.0), (+5.485, 0.0, 0.0)),
    ]
    pts3d = np.array([p for pair in pts3d_pairs for p in pair], dtype=np.float64)
    proj = project_points(pts3d, rvec, tvec, K)
    for i in range(0, len(proj), 2):
        a, b = proj[i].astype(int), proj[i + 1].astype(int)
        cv2.line(img, tuple(a), tuple(b), (0, 0, 200), 1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--csv-3d", default="ball_traj_3D.csv",
                   help="Filename of the 3D CSV inside rally_dir")
    args = p.parse_args()

    rdir: Path = args.rally_dir
    video = Path(args.video) if args.video else rdir / "rally.mp4"
    out_path = Path(args.out) if args.out else rdir / "ball_3d_overlay.mp4"
    cam_yaml = rdir / "camera.yaml"
    csv_2d = rdir / "ball_traj_2D.csv"
    csv_3d = rdir / args.csv_3d
    for q in (video, cam_yaml, csv_2d, csv_3d):
        if not q.exists():
            print(f"ERROR: missing {q}", file=sys.stderr)
            return 2

    rvec, tvec, f, h, w = read_camera_info(cam_yaml)
    K = get_K(f, h, w)

    traj2 = read_traj(csv_2d)
    by_frame_2d = {int(r[0]): (float(r[1]), float(r[2]), int(r[3]))
                   for r in traj2}

    df3 = pd.read_csv(csv_3d)
    has_src = "src" in df3.columns
    by_frame_3d = {int(r["idx"]): (np.array([r["x"], r["y"], r["z"]]),
                                   r["src"] if has_src else "phys")
                   for _, r in df3.iterrows()}

    cap = cv2.VideoCapture(str(video))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # OpenCV's bundled ffmpeg often lacks avc1, so write an MJPG AVI in a
    # temp dir then transcode to H.264 with the system ffmpeg.
    tmp_dir = Path(tempfile.mkdtemp(prefix="tennis3d_overlay_"))
    tmp_avi = tmp_dir / "overlay.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(tmp_avi), fourcc, in_fps, (in_w, in_h))
    if not writer.isOpened():
        print(f"ERROR: failed to open intermediate writer at {tmp_avi}",
              file=sys.stderr)
        return 3

    bounce_pts_world = df3.iloc[
        np.argsort(np.abs(df3["z"].values))[: max(1, len(df3) // 30)]
    ]  # heuristic: smallest |z| samples are at/near the bounce events

    n_drawn_2d = 0
    n_drawn_3d = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_court_lines(frame, rvec, tvec, K)

        if fi in by_frame_2d:
            x, y, vis = by_frame_2d[fi]
            if vis:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                n_drawn_2d += 1

        if fi in by_frame_3d:
            p3, src = by_frame_3d[fi]
            proj = project_points(p3.reshape(1, 3), rvec, tvec, K)[0]
            color = {"phys": (255, 200, 0),
                     "interp": (180, 180, 180),
                     "racket": (0, 200, 255),
                     "bounce": (0, 200, 0)}.get(src, (255, 200, 0))
            cv2.circle(frame, (int(proj[0]), int(proj[1])), 9, color, 2)
            n_drawn_3d += 1
            cv2.putText(
                frame,
                f"({p3[0]:+.1f},{p3[1]:+.1f},{p3[2]:+.1f})",
                (int(proj[0]) + 12, int(proj[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # banner
        cv2.putText(
            frame,
            f"frame {fi:4d}/{n_frames}  green=det  cyan=3D-reproj  red=court",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()

    # Transcode AVI -> H.264 MP4
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_avi),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if res.returncode != 0:
        print(f"ERROR: ffmpeg transcode failed:\n{res.stderr}", file=sys.stderr)
        return 4

    print(f"[overlay] wrote {out_path}  ({fi} frames, "
          f"{n_drawn_2d} 2D dots, {n_drawn_3d} 3D rings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
