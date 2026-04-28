#!/usr/bin/env python3
"""Bucket the TrackNet landing error by clip length.

Reads the per-clip JSON already produced by eval_tracknet_landing.py
(no recomputation, no changes to existing files) and reports:

    * bounce-weighted mean landing error per length bucket
    * number of clips + number of GT bounces per bucket

If landing error grows with clip length, that's evidence that short
training sequences (our 53-234 frames) can't support the long
real-TrackNet clips (up to 870 frames).

Usage:
    /media/skr/storage/ten_bad/.venv/bin/python bucket_landing_by_length.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np


DEFAULT_EDGES = (0, 100, 200, 400, 600, 10_000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="inference_output/tracknet_eval_5k_v2_landing/"
                "tracknet_eval_landing_results.json",
    )
    p.add_argument(
        "--edges", type=str, default=",".join(str(e) for e in DEFAULT_EDGES),
        help="Comma-separated bucket edges on num_frames.",
    )
    args = p.parse_args()

    edges = [int(x) for x in args.edges.split(",")]
    with open(args.results) as f:
        d = json.load(f)

    clips = d["clips"]
    print(f"Source: {args.results}")
    print(f"Model epoch: {d.get('model_epoch')}  "
          f"evaluated clips: {d.get('evaluated_clips')}  "
          f"bounces: {d.get('num_bounces')}")
    print(f"Overall landing xz mean: {d.get('landing_xz_mean_m'):.3f} m  "
          f"median: {d.get('landing_xz_median_m'):.3f} m\n")

    print("Bucket by clip length (num_frames):")
    print(f"  {'range':>14} | {'#clips':>6} | {'#bounces':>8} | "
          f"{'mean_xz_m':>10} | {'median_xz_m':>11} | {'mean_reproj_px':>14}")
    print("  " + "-" * 78)

    # Collect per-bucket stats (bounces weighted, clips simple).
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bucket = [c for c in clips if lo <= c["num_frames"] < hi]
        n_clips = len(in_bucket)
        if n_clips == 0:
            print(f"  [{lo:>5},{hi:>6}) | {0:>6} | {0:>8} | "
                  f"{'-':>10} | {'-':>11} | {'-':>14}")
            continue

        # Expand per-bounce values: every clip contributes num_bounces copies
        # of its mean (so the bucket mean weights bounces, not clips).
        landings = []
        for c in in_bucket:
            nb = c.get("num_bounces", 0)
            m = c.get("landing_xz_mean")
            if nb > 0 and m is not None:
                landings.extend([m] * nb)

        n_bounces = len(landings)
        mean_xz = float(np.mean(landings)) if landings else float("nan")
        med_xz = float(np.median(landings)) if landings else float("nan")
        mean_reproj = float(np.mean([c["reproj_mean"] for c in in_bucket]))

        print(f"  [{lo:>5},{hi:>6}) | {n_clips:>6} | {n_bounces:>8} | "
              f"{mean_xz:>10.3f} | {med_xz:>11.3f} | {mean_reproj:>14.2f}")

    # Also: Spearman-style direction check — correlation between length
    # and per-clip landing error (only clips with bounces).
    with_bounce = [c for c in clips
                   if c.get("num_bounces", 0) > 0
                   and c.get("landing_xz_mean") is not None]
    if len(with_bounce) >= 3:
        lens = np.array([c["num_frames"] for c in with_bounce], dtype=float)
        errs = np.array([c["landing_xz_mean"] for c in with_bounce], dtype=float)
        # Rank correlation without SciPy.
        r_len = lens.argsort().argsort()
        r_err = errs.argsort().argsort()
        n = len(lens)
        rho = 1.0 - 6.0 * np.sum((r_len - r_err) ** 2) / (n * (n * n - 1))
        # Pearson for reference.
        prho = float(np.corrcoef(lens, errs)[0, 1])
        print(f"\n  Spearman rho(length, landing_err) = {rho:+.3f}  "
              f"(n={n})")
        print(f"  Pearson  r  (length, landing_err) = {prho:+.3f}")

    # Per-game summary (already in JSON) — show for context.
    pg = d.get("per_game", {})
    if pg:
        print("\nPer-game mean landing error (for context):")
        for g in sorted(pg):
            v = pg[g]
            print(f"  {g:>7}: n={v['n']:>3}  mean={v['mean']:.3f} m  "
                  f"median={v['median']:.3f} m")


if __name__ == "__main__":
    main()
