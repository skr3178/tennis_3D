# tennis_3d — Plan

Goal: per-frame 3D ball trajectory for a fixed-camera tennis clip, by repurposing
TT3D's reconstruction core. Output format mirrors tt3d's `ball_traj_3D.csv`
(`idx, x, y, z` in meters, court-centered).

## Approach: minimal, milestone by milestone

Solve and **validate one missing piece before starting the next**. Do not write
later milestones until earlier ones are working.

| # | Missing piece | Status from current map |
|---|---|---|
| 1 | Camera pose (keypoint → PnP) | ⚠️ Have TennisCourtDetector keypoints; need PnP step |
| 2 | Tennis ball/court physics constants | ❌ Need values |
| 3 | Bounce segmenter retuning | ❌ TT cadence ≠ tennis |
| 4 | 3D output + reprojection sanity check | ❌ No GT, only visual check |

## Inputs available

- Video (fixed camera): `/media/skr/storage/ten_bad/S_Original_HL_clip_cropped.mp4` (1280×720 @ 50 fps)
- 2D ball track: `/media/skr/storage/ten_bad/wasb_game1_clip1_50fps.csv`
  (columns: `frame,x,y,visible,interpolated,score`)
- Court keypoint detector: `/media/skr/storage/ten_bad/TennisCourtDetector/`
  with `model_best.pt`, `tracknet.py`, `postprocess.py`, `homography.py`,
  `court_reference.py` (14 keypoints in image space)
- Reconstruction core to reuse (verbatim):
  `/media/skr/storage/ten_bad/tt3d/tt3d/rally/{casadi_dae.py, casadi_reconstruction.py, geometry.py}`
  and `traj_seg/segmenter.py` (`basic_segmenter` + `get_accurate_bouncing_pose`).

---

## Milestone 1 — Camera pose (do first, in isolation)

**Deliverable:** `data/game1_clip1/camera.yaml` in tt3d's format
(`rvec, tvec, f, h, w`).

**Steps:**
1. Run TennisCourtDetector on a single frame (or first N frames, average) of the
   clip. Save the 14 detected 2D keypoints.
2. Define the 14 corresponding **3D court keypoints in meters** (z=0 plane,
   origin at center of net), matching the index order used in
   `TennisCourtDetector/court_reference.py:key_points`.
3. Use `cv2.solvePnP` with `SOLVEPNP_SQPNP` to estimate `rvec, tvec` given
   2D↔3D pairs. For focal length: try (a) `cv2.calibrateCamera` from a few
   frames or (b) iterate over a small set of plausible `f` values and pick the
   one with lowest reprojection error.
4. Write `camera.yaml` using `tt3d/calibration/utils.py:write_camera_info`.

**Test (must pass before moving to M2):**
- Project all 14 court 3D points back through `(rvec, tvec, K)` and overlay on
  the source frame. Median reprojection error < 5 px.
- Save overlay as `data/game1_clip1/calib_check.png` for visual confirmation.

**Files added in M1 only:**
- `tennis_3d/tennis_3d/calibration/utils.py` (copy of tt3d's)
- `tennis_3d/tennis_3d/calibration/court_keypoints_3d.py` (14 court points)
- `tennis_3d/tennis_3d/calibration/court_pnp.py` (PnP + yaml writer)
- `tennis_3d/scripts/calibrate.py` (CLI wrapper)

---

## Milestone 2 — Tennis physics constants

Only start after M1 passes.

**Deliverable:** `tennis_3d/tennis_3d/constants.py` replacing tt3d's TT values.

Initial values (hard court, no surface tuning yet):
- `R = 0.0335`  (tennis ball radius, m)
- `M = 0.058`   (mass, kg)
- `KD ≈ 5.5e-4` (drag, derived from 0.5·ρ·Cd·A with Cd≈0.55)
- `KM ≈ 3.0e-5` (Magnus)
- `COR = 0.75`  (hard court)
- `MU  = 0.6`   (hard court friction)
- `G = 9.81`

**Test:** import `casadi_dae.py` in a smoke script, integrate a dummy serve
(v0 ≈ (0, -50, -2) m/s, no spin, p_bounce on baseline), confirm it lands within
the court roughly when expected.

---

## Milestone 3 — Ball CSV adapter + segmenter retuning

**Deliverables:**
- `tennis_3d/tennis_3d/ball_io/wasb_to_traj.py`: WASB CSV → tt3d `Frame,Visibility,X,Y` shape (drop blur features; run `use_blur=False`).
- Tuned `L` parameter for `basic_segmenter` on tennis (TT used L=200; expect smaller for tennis given longer arcs / sparser bounces).

**Test:**
- Run segmenter on the WASB CSV; visualize segment boundaries on `(t, u)` and
  `(t, v)` plots. Boundaries should land on visible bounce events in the video
  (eyeball check, ±2 frames).

---

## Milestone 4 — 3D reconstruction + reprojection check

**Deliverables:**
- `tennis_3d/tennis_3d/rally/rally.py` (adapted from tt3d): every interior
  segment break treated as a court bounce (`q_table = q_sol[1:-1]`). No
  TT-style serve special case in v1.
- Output `data/game1_clip1/ball_traj_3D.csv`.
- `tennis_3d/scripts/render_overlay.py`: project the 3D trajectory back onto
  `rally.mp4` to validate visually (no 3D GT available).

**Test:**
- Reprojected ball position tracks the WASB 2D detections within ~10 px median
  error along the visible portion of the rally.

---

## Out of v1 scope

- Multi-rally auto segmentation (one rally at a time, manual frame range).
- Player 3D pose (skip MotionBERT/RTMPose path entirely).
- Pan/zoom cameras (clip is fixed-camera).
- Surface auto-detection (default to hard court constants).
- Quantitative metric vs 3D ground truth (no GT exists).

## Folder layout (target)

```
tennis_3d/
├── PLAN.md
├── data/
│   └── game1_clip1/
│       ├── rally.mp4 -> ../../../S_Original_HL_clip_cropped.mp4
│       ├── camera.yaml          # M1
│       ├── calib_check.png      # M1
│       ├── ball_traj_2D.csv     # M3
│       └── ball_traj_3D.csv     # M4
├── tennis_3d/
│   ├── __init__.py
│   ├── constants.py             # M2
│   ├── calibration/             # M1
│   ├── ball_io/                 # M3
│   ├── rally/                   # M4 (copies of tt3d core + adapted rally.py)
│   └── traj_seg/                # M3 (copy of tt3d segmenter)
└── scripts/
    ├── calibrate.py             # M1
    ├── make_ball_csv.py         # M3
    ├── reconstruct.py           # M4
    └── render_overlay.py        # M4
```

## Reuse summary

**Verbatim from tt3d:** `rally/casadi_dae.py`, `rally/casadi_reconstruction.py`,
`rally/geometry.py`, `calibration/utils.py`, `traj_seg/segmenter.py` (subset),
`traj_seg/utils.py` (adapted to WASB columns).

**Reused without copy:** `TennisCourtDetector/` (model + court_reference +
homography), WASB CSV outputs, the source clip, main project `.venv`.

**New (small):** constants, court_pnp.py, court_keypoints_3d.py,
wasb_to_traj.py, adapted `rally.py`, render_overlay.py.

## Next action

Start Milestone 1: scaffold the calibration submodule and write
`court_pnp.py` + the 3D court keypoints, then validate with the
reprojection-overlay test. **No other milestones touched until M1 passes.**
