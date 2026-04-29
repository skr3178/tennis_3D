"""Run TrackNetV2 ball detection on a clip and write a WASB-schema CSV plus
an H.264 overlay video.

CSV columns: frame,x,y,visible — same shape `make_ball_csv.py` already
accepts, so the rest of the pipeline (segmenter, reconstruct, render) is a
drop-in.

Usage:
    python scripts/run_tracknet.py --video data/game1_clip1/rally.mp4 \
        --model /media/skr/storage/ten_bad/tracknetv2_tennis_best.tar \
        --csv  data/game1_clip1/tracknet_full.csv \
        --out  data/game1_clip1/tracknet_overlay.mp4
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

TN_DIR = Path("/media/skr/storage/ten_bad/TrackNet")


def _import_tracknet():
    sys.path.insert(0, str(TN_DIR))
    try:
        from model import BallTrackerNet
        from general import postprocess
    finally:
        sys.path.pop(0)
    return BallTrackerNet, postprocess


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    BallTrackerNet, postprocess = _import_tracknet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallTrackerNet().to(device)
    state = torch.load(str(args.model), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[tracknet] model loaded on {device}")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[tracknet] video: {in_w}x{in_h} @ {fps} fps, {n_frames} frames")

    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    print(f"[tracknet] read {len(frames)} frames")

    OUT_W, OUT_H = 640, 360

    ball: list[tuple[float | None, float | None]] = [(None, None)] * 2
    with torch.no_grad():
        for k in tqdm(range(2, len(frames)), desc="[tracknet] infer"):
            stack = np.concatenate(
                (
                    cv2.resize(frames[k], (OUT_W, OUT_H)),
                    cv2.resize(frames[k - 1], (OUT_W, OUT_H)),
                    cv2.resize(frames[k - 2], (OUT_W, OUT_H)),
                ),
                axis=2,
            ).astype(np.float32) / 255.0
            inp = torch.from_numpy(np.rollaxis(stack, 2, 0)).unsqueeze(0).to(device)
            out = model(inp.float())
            out_arg = out.argmax(dim=1).detach().cpu().numpy().astype(np.float32)
            x_pred, y_pred = postprocess(out_arg)
            if x_pred is None or y_pred is None:
                ball.append((None, None))
            else:
                ball.append((float(x_pred), float(y_pred)))

    visible = sum(1 for b in ball if b[0] is not None)
    print(f"[tracknet] visible: {visible}/{len(ball)} frames")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "visible", "interpolated", "score"])
        for fi, (x, y) in enumerate(ball):
            if x is None or y is None:
                w.writerow([fi, "", "", 0, 0, 0.0])
            else:
                w.writerow([fi, x, y, 1, 0, 1.0])
    print(f"[tracknet] wrote {args.csv}")

    # Overlay video: detected ball as red dot, recent trail behind it.
    tmp_dir = Path(tempfile.mkdtemp(prefix="tn_overlay_"))
    tmp_avi = tmp_dir / "ov.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(tmp_avi), fourcc, fps, (in_w, in_h))
    TRAIL = 6
    for fi, fr in enumerate(frames):
        for back in range(TRAIL):
            j = fi - back
            if j < 0 or ball[j][0] is None:
                continue
            r = max(2, 8 - back)
            cv2.circle(fr, (int(ball[j][0]), int(ball[j][1])),
                       r, (0, 0, 255), -1 if back == 0 else 1)
        cv2.putText(fr, f"frame {fi}/{len(frames)}  TrackNetV2  red=det",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(fr)
    writer.release()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_avi),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20",
        str(args.out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if res.returncode != 0:
        print(f"ERROR: ffmpeg transcode failed:\n{res.stderr}", file=sys.stderr)
        return 4
    print(f"[tracknet] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
