"""M3 step 1: convert raw 2D ball-track CSV to ball_traj_2D.csv.

Usage:
    python scripts/make_ball_csv.py \
        --src /media/skr/storage/ten_bad/wasb_game1_clip1_50fps.csv \
        --dst data/game1_clip1/ball_traj_2D.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d.ball_io.to_traj import convert


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n = convert(args.src, args.dst)
    print(f"[ball_csv] wrote {n} rows -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
