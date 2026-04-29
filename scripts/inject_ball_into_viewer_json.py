"""Inject our reconstructed 3D ball trajectory (court frame) into the
existing PromptHMR viewer JSON (Y-up, ball field per frame).

Coordinate map court -> viewer:
    viewer_x = court_x
    viewer_y = court_z   (height stays as height)
    viewer_z = -court_y  (sign flip on along-court axis so existing
                          player positions and ours line up)

Frames where our pipeline has no 3D fit get null (viewer hides ball).

Usage:
    python scripts/inject_ball_into_viewer_json.py \
        --src /media/skr/storage/ten_bad/tennis_match_3d_prompthmr.json \
        --ball data/game1_clip1/ball_traj_3D_pose.csv \
        --out /media/skr/storage/ten_bad/tennis_match_3d_prompthmr_balltrack.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--ball", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"[inject] loading viewer JSON: {args.src}")
    with open(args.src) as fh:
        d = json.load(fh)
    n = d["total_frames"]
    print(f"[inject] {n} frames in JSON")

    print(f"[inject] loading ball CSV: {args.ball}")
    ball = pd.read_csv(args.ball)
    by_frame = {int(r["idx"]): (float(r["x"]), float(r["y"]), float(r["z"]),
                                r["src"] if "src" in ball.columns else "phys")
                for _, r in ball.iterrows()}
    print(f"[inject] {len(by_frame)} reconstructed 3D rows")

    # Replace ball per frame
    n_set = 0
    for fi, frame_obj in enumerate(d["frames"]):
        if fi in by_frame:
            cx, cy, cz, src = by_frame[fi]
            frame_obj["ball"] = [cx, cz, -cy]
            frame_obj["ball_src"] = src
            n_set += 1
        else:
            frame_obj["ball"] = None
            frame_obj["ball_src"] = None
    print(f"[inject] set {n_set} frames with real ball, {n - n_set} nulled")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(d, fh)
    print(f"[inject] wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
