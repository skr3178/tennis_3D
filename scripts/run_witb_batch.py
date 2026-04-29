"""Batch the full pipeline + GT comparison across all WITB clips with real GT.

For each clip id in 0..max-id:
  1. Stage rally.mp4 in data/witb_id<NN>/
  2. Calibrate camera (TCD + PnP)
  3. Run WASB inference
  4. Convert to ball_traj_2D.csv
  5. Reconstruct 3D
  6. Compare against GT JSON
  7. Append a row to a summary CSV

Usage:
    python scripts/run_witb_batch.py --max-id 5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WITB_DIR = Path("/media/skr/storage/ten_bad/notes/screenshots/where_is_the_ball/clips_mp4")
WASB_SCRIPT = Path("/media/skr/storage/ten_bad/wasb_ball_detect.py")
WASB_MODEL = Path("/media/skr/storage/ten_bad/wasb_tennis_best.pth.tar")
PYTHON = "/media/skr/storage/ten_bad/.venv/bin/python"
PROJ = Path(__file__).resolve().parents[1]


def find_clip(idx: int) -> tuple[Path, Path]:
    pat = re.compile(rf"^id{idx:02d}_.*\.mp4$")
    for f in WITB_DIR.iterdir():
        if pat.match(f.name):
            return f, f.with_suffix(".json")
    raise FileNotFoundError(f"no clip with id {idx:02d}")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(cmd)}):\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return res.stdout


def gt_to_our_frame(gt_their: np.ndarray) -> np.ndarray:
    return np.column_stack([gt_their[:, 0], gt_their[:, 2], gt_their[:, 1]])


def evaluate(rally_dir: Path, gt_json: Path) -> dict:
    with open(gt_json) as fh:
        d = json.load(fh)
    gt = gt_to_our_frame(np.array(d["gt"], dtype=np.float64))
    pred_unref = gt_to_our_frame(np.array(d["pred_unrefined"], dtype=np.float64))

    ours_csv = rally_dir / "ball_traj_3D.csv"
    if not ours_csv.exists() or ours_csv.stat().st_size < 30:
        return {
            "id": d["id"], "n_gt": len(gt), "n_ours": 0, "n_match": 0,
            "med_3d": float("nan"), "mean_3d": float("nan"),
            "p90_3d": float("nan"), "med_z": float("nan"),
            "n_bounces": 0, "paper_med_3d": _paper_err(gt, pred_unref),
        }
    ours = pd.read_csv(ours_csv).sort_values("idx").reset_index(drop=True)

    matched = []
    for _, r in ours.iterrows():
        f = int(r["idx"])
        if 0 <= f < len(gt):
            matched.append((f, np.array([r["x"], r["y"], r["z"]]), gt[f]))
    if not matched:
        return {
            "id": d["id"], "n_gt": len(gt), "n_ours": len(ours), "n_match": 0,
            "med_3d": float("nan"), "mean_3d": float("nan"),
            "p90_3d": float("nan"), "med_z": float("nan"),
            "n_bounces": 0, "paper_med_3d": _paper_err(gt, pred_unref),
        }
    o = np.array([m[1] for m in matched])
    g = np.array([m[2] for m in matched])
    err = np.linalg.norm(o - g, axis=1)
    err_z = np.abs(o[:, 2] - g[:, 2])

    bounces_csv = rally_dir / "bounces.csv"
    n_b = len(pd.read_csv(bounces_csv)) if bounces_csv.exists() else 0

    return {
        "id": d["id"], "n_gt": len(gt), "n_ours": len(ours), "n_match": len(matched),
        "med_3d": float(np.median(err)),
        "mean_3d": float(np.mean(err)),
        "p90_3d": float(np.quantile(err, 0.9)),
        "med_z": float(np.median(err_z)),
        "n_bounces": n_b,
        "paper_med_3d": _paper_err(gt, pred_unref),
    }


def _paper_err(gt: np.ndarray, pred: np.ndarray) -> float:
    """Median 3D error between paper's `pred_unrefined` and `gt` — for context."""
    err = np.linalg.norm(pred - gt, axis=1)
    return float(np.median(err))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-id", type=int, default=5,
                   help="Run ids 0..max-id (real-GT clips are 0-5)")
    p.add_argument("--reproj-threshold", type=float, default=20.0)
    p.add_argument("--window-seconds", type=float, default=0.4)
    p.add_argument("--summary-csv", type=Path,
                   default=PROJ / "data" / "witb_summary.csv")
    p.add_argument("--no-chain-arcs", action="store_true",
                   help="Pass --no-chain-arcs to reconstruct.py")
    p.add_argument("--min-vz-down", type=float, default=1.0)
    p.add_argument("--chain-reproj-gate", type=float, default=100.0)
    p.add_argument("--enable-spin", action="store_true")
    args = p.parse_args()

    rows = []
    for idx in range(args.max_id + 1):
        try:
            mp4, gt_json = find_clip(idx)
        except FileNotFoundError as e:
            print(f"[batch] skip {idx}: {e}")
            continue
        print(f"\n========== clip {idx:02d}  {mp4.name} ==========")
        rally_dir = PROJ / "data" / f"witb_id{idx:02d}"
        rally_dir.mkdir(parents=True, exist_ok=True)
        rally_mp4 = rally_dir / "rally.mp4"
        if not rally_mp4.exists():
            rally_mp4.symlink_to(mp4)

        try:
            run([PYTHON, str(PROJ / "scripts" / "calibrate.py"), str(rally_dir)])
            run([PYTHON, str(WASB_SCRIPT),
                 "--video", str(rally_mp4),
                 "--model", str(WASB_MODEL),
                 "--csv", str(rally_dir / "wasb_full.csv"),
                 "--output", str(rally_dir / "wasb_overlay.mp4")])
            run([PYTHON, str(PROJ / "scripts" / "make_ball_csv.py"),
                 "--src", str(rally_dir / "wasb_full.csv"),
                 "--dst", str(rally_dir / "ball_traj_2D.csv")])
            recon_cmd = [PYTHON, str(PROJ / "scripts" / "reconstruct.py"),
                         str(rally_dir),
                         "--fps", "25",
                         "--reproj-threshold", str(args.reproj_threshold),
                         "--window-seconds", str(args.window_seconds),
                         "--min-vz-down", str(args.min_vz_down),
                         "--chain-reproj-gate", str(args.chain_reproj_gate)]
            if args.no_chain_arcs:
                recon_cmd.append("--no-chain-arcs")
            if args.enable_spin:
                recon_cmd.append("--enable-spin")
            run(recon_cmd)
        except RuntimeError as e:
            print(f"[batch] FAILED clip {idx}: {e}")
            # Remove stale outputs so evaluate() doesn't read leftover CSVs
            for stale in ("ball_traj_3D.csv", "bounces.csv"):
                p = rally_dir / stale
                if p.exists():
                    p.unlink()

        row = evaluate(rally_dir, gt_json)
        rows.append(row)
        print(f"[batch] id{idx:02d}  bounces={row['n_bounces']}  "
              f"matched={row['n_match']}/{row['n_gt']}  "
              f"med_3d={row['med_3d']:.2f}m  med_z={row['med_z']:.2f}m  "
              f"(paper unref med_3d={row['paper_med_3d']:.2f}m)")

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[batch] wrote summary -> {args.summary_csv}")

    print("\n=== Aggregate ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    valid = df.dropna(subset=["med_3d"])
    if len(valid) > 0:
        print(f"  ours median(med_3d) over {len(valid)} clips: "
              f"{valid['med_3d'].median():.2f} m")
        print(f"  paper unrefined median(med_3d) over same clips: "
              f"{valid['paper_med_3d'].median():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
