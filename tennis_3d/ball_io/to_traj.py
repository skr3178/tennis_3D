"""Convert a ball-track CSV (WASB or TrackNet flavour) into the schema the
segmenter consumes."""
from __future__ import annotations

from pathlib import Path

from tennis_3d.traj_seg.utils import read_traj, write_traj


def convert(src_csv: Path, dst_csv: Path) -> int:
    traj = read_traj(src_csv)
    write_traj(dst_csv, traj)
    return len(traj)
