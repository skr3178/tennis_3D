#!/usr/bin/env python3
"""Convert OUR Unity-generated dataset (frames.csv + camera.json + meta.json
per episode) into the same per-sequence .npz format paper_to_npz.py emits,
applying the unity.md acceptance checklist.

Filter priorities (per unity.md):
  1. plane-point explosion       (enable_plane_point)
  2. endpoint / no partial flight (always: ground_thresh + partial_flight_*)
  3. serve removal               (enable_serve_filter)
  4. minimum post-trim length    (min_len)
  5. volley / no internal bounce (enable_volley_filter)
  6. stroke count sanity         (enable_stroke_filter)        -- recommended
  7. max-height range            (enable_height_filter)        -- recommended
  8. court coverage              (enable_court_filter)         -- recommended
  9. 2D framing / image edges    (enable_framing_filter)       -- recommended
 10. broadcast camera subset     (enable_camera_filter)        -- nice-to-have
 12. post-trim endpoint sanity   (enable_post_trim_check)      -- nice-to-have
"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
from dataclasses import dataclass, field, asdict
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    # Priority 2: endpoint trim. Bounces are detected as local minima of y;
    # ground_thresh is the maximum y allowed at a bounce nadir (loose because
    # 50fps sampling rarely captures the true contact frame — typical observed
    # nadir lies in [0.05, 0.20] m even for clean ground bounces).
    ground_thresh: float = 0.20
    # "True-bounce" confirmation: within ±3 frames of trim start/end, require
    # at least N frames with y < ground_thresh * mult — guards against trimming
    # at a single-frame low-y outlier in mid-flight.
    bounce_confirm_window: int = 3
    bounce_confirm_min_frames: int = 2
    bounce_confirm_y_mult: float = 1.5

    # Priority 4
    min_len: int = 50                    # was 20

    # Priority 1
    enable_plane_point: bool = True
    plane_p_max: float = 1e3
    plane_eps_denom: float = 1e-3

    # Priority 3
    enable_serve_filter: bool = True
    serve_pre_hit_y_max: float = 0.15

    # Priority 5: reject pure-volley exchanges (no internal bounce between
    # the trim endpoints). Disabled by default because the trim already
    # forces both endpoints to be bounces, so single-arc rallies (legit per
    # unity.md) would otherwise be over-filtered.
    enable_volley_filter: bool = False
    internal_contact_min: int = 1

    # Recommended
    enable_stroke_filter: bool = True
    stroke_min: int = 2
    stroke_max: int = 8

    enable_height_filter: bool = True
    height_min: float = 0.6
    height_max: float = 5.0

    enable_court_filter: bool = True
    court_x_max: float = 8.5
    court_z_max: float = 30.0
    bbox_y_min: float = -0.05

    enable_framing_filter: bool = True
    framing_edge_px: int = 10
    framing_max_edge_frac: float = 0.20

    # Nice-to-have (off by default)
    enable_camera_filter: bool = False
    fx_min: float = 1500.0
    fx_max: float = 2500.0
    cam_y_min: float = 5.0
    cam_y_max: float = 11.0
    cam_z_min: float = -40.0
    cam_z_max: float = -22.0
    pitch_min_deg: float = 8.0
    pitch_max_deg: float = 22.0

    enable_post_trim_check: bool = False
    post_trim_y_max: float = 0.05
    post_trim_max_jump: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_ground_contacts(y: np.ndarray, vy: np.ndarray,
                         thresh: float = 0.20) -> np.ndarray:
    """Bounce frames: local minima of y with y < thresh.

    A frame i is a ground contact iff y[i] is a local minimum (y[i] <= y[i-1]
    and y[i] <= y[i+1]) AND y[i] < thresh. This is robust to 50fps sampling
    where the actual contact nadir is rarely captured at y ≈ 0; the local-min
    frame is at most one sampling tick off the true bounce. `vy` is unused but
    kept for backwards compatibility with callers.
    """
    if len(y) < 3:
        return np.array([], dtype=int)
    is_min = (y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:]) & (y[1:-1] < thresh)
    return (np.where(is_min)[0] + 1).astype(int)


def compute_plane_points(uv: np.ndarray, cam: dict) -> tuple[float, float]:
    """Return (max |P_x or P_z|, min |ray_y|) for ground-plane back-projection.

    Plane-point P is the (x, _, z) world point where the camera ray through
    pixel (u,v) hits the y=0 plane. Degenerate when ray has tiny y-component
    (small denom) or the projection blows up far from court (large |P|).
    Uses OpenGL camera convention: cam looks down -Z, image v increases down.
    """
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    E = np.asarray(cam["worldToCamera"], dtype=np.float64)
    R = E[:3, :3]

    if "position" in cam:
        cam_pos = np.asarray(cam["position"], dtype=np.float64)
    else:
        cam_pos = -R.T @ E[:3, 3]

    u, v = uv[:, 0].astype(np.float64), uv[:, 1].astype(np.float64)
    xc = (u - cx) / fx
    yc = -(v - cy) / fy
    zc = -np.ones_like(u)
    ray_cam = np.stack([xc, yc, zc], axis=1)
    ray_world = ray_cam @ R              # R^T @ ray_cam, in row form

    denom = ray_world[:, 1]
    denom_abs_min = float(np.abs(denom).min()) if len(denom) else 0.0
    if denom_abs_min < 1e-9:
        return float("inf"), denom_abs_min

    t_param = -cam_pos[1] / denom
    P = cam_pos[None, :] + t_param[:, None] * ray_world
    pp_max = float(max(np.abs(P[:, 0]).max(), np.abs(P[:, 2]).max()))
    return pp_max, denom_abs_min


def camera_pitch_deg(cam: dict) -> float:
    """Camera downward pitch (degrees), positive = looking down."""
    E = np.asarray(cam["worldToCamera"], dtype=np.float64)
    fwd = -E[2, :3]                       # camera looks down -Z (OpenGL)
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    return float(np.degrees(np.arcsin(-fwd[1])))


# ---------------------------------------------------------------------------
# Per-episode filter
# ---------------------------------------------------------------------------

def filter_episode(ep_dir: pathlib.Path, cfg: FilterConfig
                   ) -> tuple[bool, str, dict | None, int]:
    """Return (keep, reason, data_or_None, post_trim_len)."""
    f = np.genfromtxt(ep_dir / "frames.csv", delimiter=",", names=True)
    cam = json.loads((ep_dir / "camera.json").read_text())
    meta_path = ep_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    xyz_full = np.stack([f["x"], f["y"], f["z"]], axis=1).astype(np.float32)
    uv_full  = np.stack([f["u"], f["v"]], axis=1).astype(np.float32)
    eot_full = f["eot"].astype(np.uint8) if "eot" in f.dtype.names \
                                         else np.zeros(len(xyz_full), np.uint8)
    vy_full  = (f["vy"].astype(np.float32) if "vy" in f.dtype.names
                else np.zeros(len(xyz_full), np.float32))

    # ---- Pre-trim filters (whole-episode meta) ----
    if cfg.enable_stroke_filter:
        ns = int(meta.get("numStrokes", 0))
        if not (cfg.stroke_min <= ns <= cfg.stroke_max):
            return False, f"stroke_count={ns}", None, 0

    if cfg.enable_height_filter:
        mh = float(meta.get("maxHeight", 0.0))
        if not (cfg.height_min <= mh <= cfg.height_max):
            return False, f"maxHeight={mh:.2f}", None, 0

    if cfg.enable_court_filter:
        bmin = meta.get("bboxMin", [0.0, 0.0, 0.0])
        bmax = meta.get("bboxMax", [0.0, 0.0, 0.0])
        if max(abs(bmin[0]), abs(bmax[0])) > cfg.court_x_max:
            return False, "bbox_x_oob", None, 0
        if max(abs(bmin[2]), abs(bmax[2])) > cfg.court_z_max:
            return False, "bbox_z_oob", None, 0
        if bmin[1] < cfg.bbox_y_min:
            return False, f"bbox_y_min={bmin[1]:.3f}", None, 0

    if cfg.enable_camera_filter:
        if not (cfg.fx_min <= cam["fx"] <= cfg.fx_max):
            return False, f"fx={cam['fx']:.0f}", None, 0
        cam_pos = cam.get("position", [0.0, 0.0, 0.0])
        if not (cfg.cam_y_min <= cam_pos[1] <= cfg.cam_y_max):
            return False, f"cam_y={cam_pos[1]:.2f}", None, 0
        if not (cfg.cam_z_min <= cam_pos[2] <= cfg.cam_z_max):
            return False, f"cam_z={cam_pos[2]:.2f}", None, 0
        pitch = camera_pitch_deg(cam)
        if not (cfg.pitch_min_deg <= pitch <= cfg.pitch_max_deg):
            return False, f"pitch={pitch:.1f}deg", None, 0

    # ---- Serve filter (uses raw episode + hitFrames) ----
    if cfg.enable_serve_filter:
        hits = meta.get("hitFrames", [])
        if hits:
            first_hit = int(hits[0])
            contacts_full = find_ground_contacts(
                xyz_full[:, 1], vy_full, thresh=cfg.ground_thresh)
            first_contact = int(contacts_full[0]) if len(contacts_full) else None
            pre_hit_min_y = float(xyz_full[:max(first_hit, 1), 1].min())
            if (first_contact is None or first_hit < first_contact) \
                    and pre_hit_min_y > cfg.serve_pre_hit_y_max:
                return False, f"serve(pre_hit_min_y={pre_hit_min_y:.2f})", None, 0

    # ---- Trim to first/last ground contacts ----
    contacts = find_ground_contacts(xyz_full[:, 1], vy_full,
                                    thresh=cfg.ground_thresh)
    if len(contacts) < 2:
        return False, "no_ground_contacts", None, 0
    start, end = int(contacts[0]), int(contacts[-1])

    if end - start + 1 < cfg.min_len:
        return False, f"too_short={end - start + 1}", None, end - start + 1

    xyz = xyz_full[start:end + 1]
    uv  = uv_full[start:end + 1]
    eot = eot_full[start:end + 1]
    vy  = vy_full[start:end + 1]
    L = len(xyz)

    # ---- Post-trim filters ----
    # True-bounce confirmation at start and end of trim
    bw = cfg.bounce_confirm_window
    y_lim = cfg.ground_thresh * cfg.bounce_confirm_y_mult
    start_lo = max(0, start - bw)
    start_hi = min(len(xyz_full), start + bw + 1)
    near_start = int((xyz_full[start_lo:start_hi, 1] < y_lim).sum())
    if near_start < cfg.bounce_confirm_min_frames:
        return False, f"isolated_low_y_start={near_start}", None, L
    end_lo = max(0, end - bw)
    end_hi = min(len(xyz_full), end + bw + 1)
    near_end = int((xyz_full[end_lo:end_hi, 1] < y_lim).sum())
    if near_end < cfg.bounce_confirm_min_frames:
        return False, f"isolated_low_y_end={near_end}", None, L

    if cfg.enable_volley_filter and len(contacts) >= 2:
        # internal bounces = local-min contacts strictly between start and end
        n_internal = int(((contacts > start) & (contacts < end)).sum())
        if n_internal < cfg.internal_contact_min:
            return False, "no_internal_bounce", None, L

    if cfg.enable_framing_filter:
        W = int(cam.get("width", 1280))
        H = int(cam.get("height", 720))
        e = cfg.framing_edge_px
        u_, v_ = uv[:, 0], uv[:, 1]
        edge_frac = float(((u_ < e) | (u_ > W - e) |
                           (v_ < e) | (v_ > H - e)).mean())
        if edge_frac > cfg.framing_max_edge_frac:
            return False, f"edge_frac={edge_frac:.2f}", None, L

    if cfg.enable_plane_point:
        pp_max, denom_min = compute_plane_points(uv, cam)
        if denom_min < cfg.plane_eps_denom:
            return False, f"plane_denom={denom_min:.4f}", None, L
        if pp_max > cfg.plane_p_max:
            return False, f"plane_p_max={pp_max:.0f}", None, L

    if cfg.enable_post_trim_check:
        if xyz[0, 1] > cfg.post_trim_y_max or xyz[-1, 1] > cfg.post_trim_y_max:
            return False, "post_trim_endpoint_high", None, L
        jumps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if jumps.max(initial=0.0) > cfg.post_trim_max_jump:
            return False, f"xyz_jump={jumps.max():.2f}", None, L

    intr = np.array([cam["fx"], cam["cx"], cam["cy"]], dtype=np.float32)
    E = np.asarray(cam["worldToCamera"], dtype=np.float32)
    return True, "ok", {
        "uv": uv, "xyz": xyz, "eot": eot,
        "intrinsics": intr, "extrinsic": E,
    }, L


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, type=pathlib.Path)
    p.add_argument("--out_dir",   required=True, type=pathlib.Path)
    p.add_argument("--filter_report", type=pathlib.Path, default=None,
                   help="CSV path to log per-episode keep/reason/length")
    p.add_argument("--ground_thresh", type=float, default=0.20)
    p.add_argument("--min_len", type=int, default=50)
    # Per-filter toggles
    p.add_argument("--no_plane_point",  action="store_true")
    p.add_argument("--no_serve_filter", action="store_true")
    p.add_argument("--enable_volley_filter", action="store_true")
    p.add_argument("--no_stroke_filter", action="store_true")
    p.add_argument("--no_height_filter", action="store_true")
    p.add_argument("--no_court_filter", action="store_true")
    p.add_argument("--no_framing_filter", action="store_true")
    p.add_argument("--enable_camera_filter", action="store_true")
    p.add_argument("--enable_post_trim_check", action="store_true")
    p.add_argument("--legacy", action="store_true",
                   help="Disable all new filters; reproduce old behavior "
                        "(min_len=20, ground_thresh=0.05).")
    args = p.parse_args()

    if args.legacy:
        cfg = FilterConfig(
            ground_thresh=0.05, min_len=20,
            enable_plane_point=False, enable_serve_filter=False,
            enable_volley_filter=False, enable_stroke_filter=False,
            enable_height_filter=False, enable_court_filter=False,
            enable_framing_filter=False)
    else:
        cfg = FilterConfig(
            ground_thresh=args.ground_thresh,
            min_len=args.min_len,
            enable_plane_point=not args.no_plane_point,
            enable_serve_filter=not args.no_serve_filter,
            enable_volley_filter=args.enable_volley_filter,
            enable_stroke_filter=not args.no_stroke_filter,
            enable_height_filter=not args.no_height_filter,
            enable_court_filter=not args.no_court_filter,
            enable_framing_filter=not args.no_framing_filter,
            enable_camera_filter=args.enable_camera_filter,
            enable_post_trim_check=args.enable_post_trim_check,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(d for d in args.split_dir.iterdir()
                 if d.is_dir() and d.name.startswith("ep_"))

    written = 0
    skipped = 0
    reason_counts: dict[str, int] = {}
    lengths: list[int] = []
    report_rows: list[dict] = []

    for ep in eps:
        keep, reason, data, post_len = filter_episode(ep, cfg)
        report_rows.append({
            "episode": ep.name,
            "kept": int(keep),
            "reason": reason,
            "post_trim_len": post_len,
        })
        if not keep:
            key = reason.split("=")[0].split("(")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
            skipped += 1
            continue
        np.savez(args.out_dir / f"seq_{written:05d}.npz", **data)
        lengths.append(len(data["xyz"]))
        written += 1

    print(f"wrote {written} sequences, skipped {skipped} -> {args.out_dir}")
    if reason_counts:
        print("  reject reasons:")
        for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:30s} {v}")
    if lengths:
        print(f"  length: min={min(lengths)} max={max(lengths)} "
              f"mean={np.mean(lengths):.0f} median={np.median(lengths):.0f}")
        first_ys = [np.load(args.out_dir / f"seq_{i:05d}.npz")["xyz"][0, 1]
                    for i in range(written)]
        print(f"  first_y range: [{min(first_ys):.4f}, {max(first_ys):.4f}]")

    if args.filter_report:
        args.filter_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.filter_report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["episode", "kept",
                                               "reason", "post_trim_len"])
            w.writeheader()
            w.writerows(report_rows)
        print(f"wrote filter report -> {args.filter_report}")


if __name__ == "__main__":
    main()
