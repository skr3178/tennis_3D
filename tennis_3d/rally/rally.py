"""3D rally reconstruction adapted for tennis.

Differences from tt3d/rally/rally.py:

  * No serve special case (`solve_serve` not used in v1).
  * No 2D classify_q (u-direction reversal is unreliable in tennis). Instead,
    *every* interior segment break is treated as a candidate bounce. We solve
    the trajectory around each candidate, then accept those whose median
    reprojection error stays below `reproj_threshold`. False breaks (mid-arc
    splits, racket hits) typically fit much worse than real court bounces.
  * Spin is held off pre-bounce (`lock_spin=True`); friction at the bounce
    still generates post-bounce angular velocity inside the bounce_model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from tennis_3d.rally.casadi_reconstruction import solve_trajectory
from tennis_3d.rally.casadi_dae import rebuild
from tennis_3d.rally.geometry import get_transform, intersect_ray_plane


@dataclass
class ArcFit:
    q_idx: int                    # index into q_sol
    q_frame: int                  # video frame number of the candidate bounce
    bounce_pt_world: np.ndarray   # (3,) court-frame, z≈0 for true bounces
    v_pre: np.ndarray             # (3,) pre-bounce velocity in world frame
    w_pre: np.ndarray             # (3,) pre-bounce spin (locked at 0)
    err: float                    # reprojection objective from solver
    median_px: float              # median per-frame reprojection error in pixels
    t_window: np.ndarray          # absolute timestamps for this arc window
    traj_window_world: np.ndarray # (n, 3) world-frame ball positions per t


def project(points_3d, rvec, tvec, K):
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
    P = K @ np.hstack([R, np.asarray(tvec).reshape(3, 1)])
    homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    proj = (P @ homog.T).T
    return proj[:, :2] / proj[:, 2:3]


def reconstruct_arcs(
    t: np.ndarray,
    traj_2d: np.ndarray,
    q_sol: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    fps: float,
    reproj_threshold: float = 8.0,
    min_pts: int = 6,
    verbose: bool = False,
    bounce_y_max: float = 11.885,    # singles baseline
    bounce_x_max: float = 5.485,     # doubles sideline
    min_vz_down: float = 1.0,        # |v_z| at bounce must exceed this
    window_seconds: float = 0.4,     # half-window for fit (trim from breakpoints)
    chain_arcs: bool = True,         # extend output beyond fit window
    chain_reproj_gate: float = 100.0, # px gate when chaining (drop drift)
    lock_spin: bool = True,          # if False, allow Magnus during pre-bounce fit
) -> List[ArcFit]:
    """For each interior break q in q_sol, attempt to reconstruct it as a
    court bounce. Returns one ArcFit per accepted candidate.

    `t` is timestamp per visible frame; `traj_2d` is (n, ≥2) with columns
    [u, v, ...].
    """
    rvec = np.asarray(rvec).reshape(3)
    tvec = np.asarray(tvec).reshape(3)
    T_table = get_transform(rvec, tvec)
    T_table_inv = np.linalg.inv(T_table)

    fits: List[ArcFit] = []
    if len(q_sol) < 3:
        return fits

    # Iterate interior breaks: q_sol[1] .. q_sol[-2]
    for i in range(1, len(q_sol) - 1):
        q = int(q_sol[i])
        a = int(q_sol[i - 1])
        b = int(q_sol[i + 1])
        if (b - a) < min_pts:
            continue

        u_q = float(traj_2d[q, 0])
        v_q = float(traj_2d[q, 1])
        try:
            bp_cam = intersect_ray_plane(rvec, tvec, (u_q, v_q), K)
        except ValueError:
            continue
        bp_world = T_table_inv[:3, :3] @ bp_cam + T_table_inv[:3, 3]

        if verbose:
            print(f"  candidate q={i} frame={int(traj_2d[q,2]) if traj_2d.shape[1]>2 else q}  "
                  f"u,v=({u_q:.0f},{v_q:.0f})  "
                  f"bp_world=({bp_world[0]:+.2f},{bp_world[1]:+.2f},{bp_world[2]:+.3f})  "
                  f"window=[{a},{b}] n={b-a+1}")

        # Geometric sanity: bounce inside the singles court (with ~0.5 m
        # margin on the sidelines for wide balls and along the baseline).
        if abs(bp_world[0]) > bounce_x_max + 1.5 or abs(bp_world[1]) > bounce_y_max + 0.5:
            if verbose:
                print(f"    REJECT: bounce point outside court bbox")
            continue

        # Build window excluding the bounce sample itself: rebuild_diff splits
        # strictly into t<0 and t>0, so t==0 must not appear in the inputs.
        # Also trim to ±window_seconds so the physics fit isn't asked to
        # extrapolate across an entire racket-to-racket arc — the spin-locked
        # model can't curve enough to match real topspin/slice trajectories
        # far from the bounce.
        t_q = t[q]
        idx_before = np.arange(a, q)
        idx_after = np.arange(q + 1, b + 1)
        idx_win = np.concatenate([idx_before, idx_after])
        keep = np.abs(t[idx_win] - t_q) <= window_seconds
        idx_win = idx_win[keep]
        if len(idx_win) < min_pts:
            if verbose:
                print(f"    REJECT: <{min_pts} pts within ±{window_seconds}s window")
            continue
        t_win = t[idx_win] - t_q
        pts_win = traj_2d[idx_win, :2]

        # Init: assume the ball was approximately moving toward the bounce
        # point along Y, with slight downward v_z. Sign of v_y guessed from
        # pre-break ball motion in image space (descending v means going
        # toward camera = +Y in our court frame, depending on camera).
        # Robust fallback: try both Y-direction inits and keep the better fit.
        best = None
        for vy_guess in (-15.0, +15.0):
            init = np.array([0.0, vy_guess, -3.0, 0.0, 0.0, 0.0])
            try:
                v_sol, w_sol, err = solve_trajectory(
                    bp_world,
                    pts_win,
                    t_win,
                    K,
                    rvec,
                    tvec,
                    init_params=init,
                    lock_spin=lock_spin,
                    verbose=False,
                )
            except Exception as e:
                if verbose:
                    print(f"    solver exception (vy_guess={vy_guess}): "
                          f"{type(e).__name__}: {e}")
                continue
            v_arr = np.array(v_sol).flatten()
            w_arr = np.array(w_sol).flatten()
            err_f = float(np.array(err).flatten()[0])
            # Per-pixel median for accept/reject
            traj_world = np.array(rebuild(t_win, bp_world, v_arr, w_arr))
            if traj_world.ndim == 1:
                traj_world = traj_world.reshape(-1, 3)
            proj_2d = project(traj_world, rvec, tvec, K)
            res = np.linalg.norm(proj_2d - pts_win, axis=1)
            med_px = float(np.median(res))
            if best is None or med_px < best.median_px:
                best = ArcFit(
                    q_idx=i,
                    q_frame=int(traj_2d[q, 2]) if traj_2d.shape[1] > 2 else q,
                    bounce_pt_world=bp_world,
                    v_pre=v_arr,
                    w_pre=w_arr,
                    err=err_f,
                    median_px=med_px,
                    t_window=t[idx_win],
                    traj_window_world=traj_world,
                )
        if best is not None:
            # Physics gate: a real bounce arrives with substantial downward
            # velocity. Mid-arc apex artifacts and racket-hit candidates
            # produce v_z close to zero or wrong-sign.
            vz_ok = best.v_pre[2] <= -min_vz_down
            err_ok = best.median_px <= reproj_threshold
            verdict = "ACCEPT" if (vz_ok and err_ok) else "REJECT"
            reason = []
            if not err_ok:
                reason.append(f"med_px>{reproj_threshold:.1f}")
            if not vz_ok:
                reason.append(f"|v_z|<{min_vz_down}")
            if verbose:
                tag = verdict + (f" ({','.join(reason)})" if reason else "")
                print(f"    best fit: med_px={best.median_px:.2f}  "
                      f"v_pre=({best.v_pre[0]:+.1f},{best.v_pre[1]:+.1f},{best.v_pre[2]:+.1f})  "
                      f"{tag}")
            if vz_ok and err_ok:
                fits.append(best)

    if not chain_arcs or not fits:
        return fits

    # Extend each accepted arc out to the half-time between its bounce and
    # the previous/next accepted bounces (or to the first/last visible frame
    # at the ends). Per-frame reproj gate drops frames where the spin-locked
    # physics has drifted too far from the WASB observation.
    extended: List[ArcFit] = []
    for k, fit in enumerate(fits):
        q_self = int(q_sol[fit.q_idx])
        t_q = t[q_self]
        if k > 0:
            t_prev = t[int(q_sol[fits[k - 1].q_idx])]
            lo_t = 0.5 * (t_prev + t_q)
        else:
            lo_t = t[0]
        if k < len(fits) - 1:
            t_next = t[int(q_sol[fits[k + 1].q_idx])]
            hi_t = 0.5 * (t_q + t_next)
        else:
            hi_t = t[-1]

        mask = (t >= lo_t) & (t <= hi_t)
        idx_ext = np.where(mask)[0]
        idx_ext = idx_ext[idx_ext != q_self]
        if len(idx_ext) < 2:
            extended.append(fit)
            continue

        t_rel = t[idx_ext] - t_q
        pts_ext = traj_2d[idx_ext, :2]
        try:
            traj_ext = np.array(rebuild(t_rel, fit.bounce_pt_world,
                                        fit.v_pre, fit.w_pre))
        except Exception as e:
            if verbose:
                print(f"  chain k={k}: rebuild failed ({type(e).__name__}: {e})")
            extended.append(fit)
            continue
        if traj_ext.ndim == 1:
            traj_ext = traj_ext.reshape(-1, 3)

        proj_ext = project(traj_ext, rvec, tvec, K)
        res = np.linalg.norm(proj_ext - pts_ext, axis=1)
        keep = res <= chain_reproj_gate
        if int(keep.sum()) < 2:
            extended.append(fit)
            continue

        if verbose:
            print(f"  chain k={k} q_frame={fit.q_frame}: extended "
                  f"{len(idx_ext)} -> kept {int(keep.sum())} "
                  f"(gate={chain_reproj_gate:.0f}px, "
                  f"max_res={res.max():.1f}px)")

        extended.append(ArcFit(
            q_idx=fit.q_idx,
            q_frame=fit.q_frame,
            bounce_pt_world=fit.bounce_pt_world,
            v_pre=fit.v_pre,
            w_pre=fit.w_pre,
            err=fit.err,
            median_px=fit.median_px,
            t_window=t[idx_ext][keep],
            traj_window_world=traj_ext[keep],
        ))
    return extended


def stitch_3d_csv(
    fits: List[ArcFit], fps: float, out_csv: str
) -> Optional[np.ndarray]:
    """Concatenate per-arc 3D positions into a single (frame, x, y, z) CSV.
    Overlapping arc windows fuse via mean.
    """
    if not fits:
        return None
    rows = []
    for fit in fits:
        # Include the bounce frame itself (excluded from t_window because
        # rebuild_diff splits at t=0).
        bt = fit.q_frame / fps
        bp = fit.bounce_pt_world
        rows.append((float(bt), float(bp[0]), float(bp[1]), float(bp[2])))
        for tk, pk in zip(fit.t_window, fit.traj_window_world):
            rows.append((float(tk), float(pk[0]), float(pk[1]), float(pk[2])))
    if not rows:
        return None
    rows = np.array(rows)
    # Bucket by frame index
    frame_idx = np.round(rows[:, 0] * fps).astype(np.int64)
    uniq = np.unique(frame_idx)
    out = np.zeros((len(uniq), 4), dtype=np.float64)
    for i, f in enumerate(uniq):
        sel = rows[frame_idx == f]
        out[i, 0] = f
        out[i, 1:] = sel[:, 1:].mean(axis=0)
    import csv

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "x", "y", "z"])
        for row in out:
            w.writerow([int(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return out
