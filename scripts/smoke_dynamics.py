"""M2 smoke test: integrate a stylised tennis groundstroke through tt3d's
dynamics with tennis constants and pre-bounce spin held at zero.

Compares two integrations:
  * Real physics  : gravity + drag + (Magnus, but spin starts at 0)
  * Vacuum check  : gravity only (KD, KM zeroed at runtime)

Prints the trajectory at t = -0.4, 0 (bounce), +0.4 s and the difference at
t = +0.4 s between real and vacuum, which isolates how much horizontal travel
drag is taking out of a typical tennis arc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tennis_3d import constants as C
from tennis_3d.rally import casadi_dae as dae


def integrate(p_bounce, v0_pre, w0_pre, t_grid):
    """Wrapper around tt3d's `rebuild`. Returns ndarray (N, 3)."""
    traj = dae.rebuild(t_grid, p_bounce, v0_pre, w0_pre)
    return np.array(traj.full() if hasattr(traj, "full") else traj)


def with_zero_aero(fn):
    """Run `fn` with KD=KM=0 in the dynamics module."""
    old_kd, old_km = dae.KD, dae.KM
    dae.KD = 0.0
    dae.KM = 0.0
    try:
        return fn()
    finally:
        dae.KD = old_kd
        dae.KM = old_km


def main() -> int:
    # Stylised groundstroke: ball falls into baseline area, no spin pre-bounce.
    # Bounce point inside the near baseline corner of the receiver's court.
    p_bounce = np.array([0.0, -9.0, 0.0])              # X across, Y along court, Z up
    v0_pre = np.array([0.0, -22.0, -8.0])              # m/s, downward + toward near baseline
    w0_pre = np.array([0.0, 0.0, 0.0])                 # spin OFF pre-bounce

    fps = 50
    t_grid = np.arange(-0.4, 0.4 + 1e-9, 1.0 / fps)
    i0 = int(np.argmin(np.abs(t_grid - 0.0)))
    iL = 0
    iR = len(t_grid) - 1

    print(f"[smoke] tennis constants: R={C.R} M={C.M} KD={C.KD:.3e} "
          f"KM={C.KM:.3e} COR={C.COR} MU={C.MU}")
    print(f"[smoke] p_bounce={p_bounce}  v0_pre={v0_pre}  w0_pre={w0_pre}")

    traj_real = integrate(p_bounce, v0_pre, w0_pre, t_grid)
    traj_vac  = with_zero_aero(lambda: integrate(p_bounce, v0_pre, w0_pre, t_grid))

    def row(label, arr, idx):
        x, y, z = arr[idx]
        print(f"  {label:>14s}  t={t_grid[idx]:+.2f}s  "
              f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}")

    print("\n[smoke] real physics (gravity + drag + Magnus[from friction-spin]):")
    row("pre-bounce", traj_real, iL)
    row("bounce", traj_real, i0)
    row("post-bounce", traj_real, iR)

    print("\n[smoke] vacuum (gravity only):")
    row("pre-bounce", traj_vac, iL)
    row("bounce", traj_vac, i0)
    row("post-bounce", traj_vac, iR)

    diff = traj_real[iR] - traj_vac[iR]
    horiz_diff = float(np.linalg.norm(diff[:2]))
    print(f"\n[smoke] horizontal drag-induced displacement at t=+0.4s: "
          f"{horiz_diff:.3f} m  (full diff vec = {diff})")

    # Sanity: the ball should bounce upward (+z) post-bounce and continue away
    # from the bounce point along Y for a normal arc.
    z_after_short = traj_real[i0 + 2, 2]
    if not (z_after_short > 0):
        print(f"[smoke] FAIL: z just after bounce should be > 0, got {z_after_short:.3f}")
        return 1
    if not (np.linalg.norm(traj_real[iR, :2] - p_bounce[:2]) > 1.0):
        print("[smoke] FAIL: post-bounce horizontal travel < 1 m, dynamics off")
        return 1
    print("\n[smoke] M2 smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
