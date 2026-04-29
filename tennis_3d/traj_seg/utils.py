"""Trajectory I/O for tennis. Output schema mirrors tt3d's so the segmenter
runs unchanged, but `L` and `Theta` are zeroed because tennis 2D detectors
(WASB / TrackNet) don't emit blur features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wrap_angles(angles):
    angles = np.asarray(angles)
    angles = np.where(angles > 90, angles - 180, angles)
    angles = np.where(angles < -90, angles + 180, angles)
    return angles


def read_traj(csv_path, frame_dir=None):
    """Read a tennis ball-track CSV and return ndarray (n, 6) with columns
    [Frame, X, Y, Visibility, L, Theta]. L and Theta are zero-filled if absent.

    Accepts either tt3d-style capitalised columns (`Frame, X, Y, Visibility,
    L, Theta`) or WASB lowercase (`frame, x, y, visible`). NaN x/y rows are
    treated as not-visible (Visibility=0).
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    def col(name, *alts):
        for n in (name, *alts):
            if n in df.columns:
                return df[n]
            if n.lower() in cols:
                return df[cols[n.lower()]]
        return None

    fid = col("Frame", "frame")
    x = col("X", "x")
    y = col("Y", "y")
    vis = col("Visibility", "visible")
    L = col("L")
    Th = col("Theta")

    if fid is None or x is None or y is None:
        raise ValueError(f"CSV {csv_path} missing required Frame/X/Y columns")

    fid = pd.to_numeric(fid, errors="coerce").astype("Int64")
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    if vis is None:
        vis = (~x.isna() & ~y.isna()).astype(int)
    else:
        vis = pd.to_numeric(vis, errors="coerce").fillna(0).astype(int)
        # Force visibility off where x or y is NaN regardless of source flag.
        vis = vis.where(~(x.isna() | y.isna()), 0)

    L = pd.to_numeric(L, errors="coerce").fillna(0.0) if L is not None else pd.Series(np.zeros(len(df)))
    Th = pd.to_numeric(Th, errors="coerce").fillna(0.0) if Th is not None else pd.Series(np.zeros(len(df)))

    out = np.column_stack(
        [
            fid.fillna(-1).to_numpy(dtype=np.int64),
            x.fillna(0.0).to_numpy(dtype=np.float64),
            y.fillna(0.0).to_numpy(dtype=np.float64),
            vis.to_numpy(dtype=np.int64),
            L.to_numpy(dtype=np.float64),
            Th.to_numpy(dtype=np.float64),
        ]
    )
    # Drop frames with negative ids
    out = out[out[:, 0] >= 0]
    return out


def write_traj(out_csv, traj):
    """Write a (n, 6) array as a tt3d-format ball_traj_2D.csv."""
    df = pd.DataFrame(
        {
            "Frame": traj[:, 0].astype(np.int64),
            "X": traj[:, 1],
            "Y": traj[:, 2],
            "Visibility": traj[:, 3].astype(np.int64),
            "L": traj[:, 4],
            "Theta": traj[:, 5],
        }
    )
    df.to_csv(out_csv, index=False)
