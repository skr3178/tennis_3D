# 3D Player Tracking from Monocular Tennis Video

Two parallel pipelines extract 3D player skeletons from a single broadcast-angle video and render them on a tennis court in a Three.js viewer.

**Input video:** `S_Original_HL_clip_cropped.mp4` (767 frames, 50 fps, 1280x720)

---

## Pipeline 1: MediaPipe Pose Landmarker

**Model:** `pose_landmarker_heavy.task` (30 MB, stored in `models/`)

**Steps:**
1. YOLO + PlayerTracker (`tennis-tracking/`) detects and tracks near/far player bboxes per frame
2. CourtDetector computes camera-to-court homography
3. Per player per frame: crop bbox (pad=1.0), run MediaPipe Pose Landmarker (VIDEO mode, bidirectional pass for far player)
4. Map 33 MediaPipe landmarks → COCO-17 joints
5. Anchor feet to court position via homography projection
6. Savitzky-Golay temporal smoothing (window=11, poly=2)
7. Ball positions from `wasb_ball_positions.csv` projected to court via same homography

**Commands:**
```bash
source /media/skr/storage/ten_bad/.venv/bin/activate
python mediapipe_pose_pipeline.py
```

**Outputs:**
- `tennis_match_3d_mediapipe.json` (~1 MB) — 767 frames, COCO-17 skeleton, near 99%, far 100%, ball 97%
- `mediapipe_output/output_near.mp4`, `output_far.mp4` — side-by-side debug videos (input + 3D skeleton)

---

## Pipeline 2: PromptHMR (SMPL-X)

**Repo:** `PromptHMR/` (cloned from `github.com/yufu-wang/PromptHMR`)

**Models (in `PromptHMR/data/`):**
- SMPL-X body models (symlinked from `/home/skr/Downloads/video2robot/body_models/smplx`)
- SMPL neutral/male/female (symlinked from extracted basicmodel .pkl files)
- ViTPose-H (from HuggingFace `JunkyByte/easy_ViTPose`)
- SAM2 hiera tiny (from Facebook CDN)
- Detectron2 keypoint_rcnn (from model zoo)

**Steps:**
1. Detectron2 person detection → SAM2 segmentation → multi-person tracking
2. DROID-SLAM camera motion estimation
3. ViTPose-H 2D keypoint estimation
4. PromptHMR SMPL-X regression (world-frame body parameters)
5. Post-optimization (temporal smoothing + ground contact)
6. Converter: SMPL-X forward pass → 127 joints → pick COCO-17 via index map
7. Track selection: filter to court bounds (|Z| ≤ half-court + 3 m), pick longest track per half
8. Anchor skeleton to court via bbox foot → homography → court meters
9. Ball from `wasb_ball_positions.csv` projected to court

**Commands:**
```bash
# Step 1: Run PromptHMR detection + reconstruction (~39 min on RTX 3060)
cd /media/skr/storage/ten_bad/PromptHMR
source .venv-phmr/bin/activate
python demo_video.py --video ../S_Original_HL_clip_cropped.mp4 --out_folder results

# Step 2: Convert results.pkl → viewer JSON
source /media/skr/storage/ten_bad/.venv/bin/activate
cd /media/skr/storage/ten_bad
python prompthmr_to_viewer.py
```

**Outputs:**
- `PromptHMR/results/S_Original_HL_clip_cropped/results.pkl` (24 MB) — SMPL-X params, bboxes, tracks
- `tennis_match_3d_prompthmr.json` (~1 MB) — 767 frames, COCO-17 skeleton, near 100%, far 100%, ball 97%

---

## Three.js Viewer

Each pipeline has its own viewer (all load the same JSON schema, different data files):

| Viewer | Directory | JSON source | URL path |
|--------|-----------|-------------|----------|
| VideoPose3D (original) | `viewer_3d/` | `tennis_match_3d.json` | `/viewer_3d/` |
| MediaPipe | `viewer_3d_mediapipe/` | `tennis_match_3d_mediapipe.json` | `/viewer_3d_mediapipe/` |
| PromptHMR | `viewer_3d_prompthmr/` | `tennis_match_3d_prompthmr.json` | `/viewer_3d_prompthmr/` |

**Serving:**
```bash
cd /media/skr/storage/ten_bad
python -m http.server 8000 --bind 0.0.0.0
# Open http://10.24.186.110:8000/viewer_3d_prompthmr/
```

**Viewer features:**
- Two COCO-17 skeletons (near + far player) with colored bones
- Ball trajectory (green sphere)
- Tennis court with lines, net, and surface
- Play/pause, frame scrubber, speed control
- Camera at (18, 15, 22) framing both baselines
- Multi-frame catch-up loop for correct 50 fps playback

---

## JSON Schema

```json
{
  "fps": 50.0,
  "total_frames": 767,
  "court": {"width": 10.97, "length": 23.77, "net_height": 1.07},
  "skeleton_connections": [[0,1], [0,2], ...],
  "keypoint_names": ["nose", "left_eye", ..., "right_ankle"],
  "source": "prompthmr",
  "frames": [
    {
      "frame": 0,
      "players": [
        {
          "id": 1,
          "role": "far",
          "keypoints_3d": [[x, y, z], ...],
          "confidence": [0.95, ...]
        }
      ],
      "ball": [x, y, z]
    }
  ]
}
```

## Key Scripts

| File | Purpose |
|------|---------|
| `mediapipe_pose_pipeline.py` | MediaPipe full pipeline (detect → pose → court anchor → JSON) |
| `mediapipe_skeleton_video.py` | Side-by-side debug video (input frame + 3D skeleton) |
| `prompthmr_to_viewer.py` | Convert PromptHMR results.pkl → viewer JSON |
| `prompthmr_detection_video.py` | Render PromptHMR bbox tracks as overlay video |
