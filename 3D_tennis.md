# Plan: 3D Tennis Court Viewer with Stick-Figure Players

## Context
We have a working 2D schematic court renderer (`schematic_output.mp4`) showing player skeletons and ball trajectory projected onto a perspective court view. The user wants a **3D interactive viewer** (like the badminton reference `Bad.mp4`) where you can rotate the camera 360 degrees around the court, viewing the stick-figure players and ball in 3D space. The viewer should be web-based using Three.js.

## What We Have
- **17 COCO keypoints** per player per frame (camera-space 2D + confidence)
- **Court homography** mapping camera → reference court (ground plane)
- **Ball positions** in camera-space 2D (`wasb_ball_positions.csv`)
- **Court geometry** from `court_reference.py` (reference court: 1117x2408 pixels, sidelines, baselines, net, service lines)
- **`SKELETON_CONNECTIONS`** defining how to draw stick figures (16 bone connections)
- **`compute_upright_skeleton`** already lifts 2D keypoints to upright positions using perspective scaling

## Approach: Two-Part Pipeline

### Part 1: Python — Export frame-by-frame 3D data to JSON

New script: `export_3d_data.py`

**For each frame, export:**
```json
{
  "fps": 50,
  "court": { "width": 10.97, "length": 23.77, "net_height": 1.07 },
  "frames": [
    {
      "frame": 0,
      "players": [
        {
          "id": 1,
          "role": "far",
          "keypoints_3d": [[x,y,z], ...],
          "confidence": [0.9, ...]
        },
        { "id": 2, "role": "near", ... }
      ],
      "ball": [x, y, z]
    }
  ]
}
```

**3D coordinate system:**
- X = lateral (across court), origin at court center
- Y = height (up from ground), 0 = ground plane  
- Z = depth (along court), origin at net

**Lifting 2D → 3D:**
1. Project foot position through homography → gives (ref_x, ref_y) on the reference court
2. Map reference court coords to meters: `x_meters = (ref_x - 832.5) / 1093 * 10.97`, `z_meters = (ref_y - 1748) / 2408 * 23.77`
3. For each keypoint above the foot: `dx` and `dy` offsets from foot in camera pixels, scaled by `pixel_scale = court_width_meters / court_width_camera_pixels` at that depth
4. `x_3d = foot_x_meters + dx * pixel_scale` (lateral)
5. `y_3d = dy * pixel_scale` (height above ground — always positive)
6. `z_3d = foot_z_meters` (all keypoints share foot's depth — monocular assumption)

**Ball height:** The ball is not on the ground plane. Since we only have 2D, approximate: project ball x,y to court plane for lateral/depth position, set height to a fixed estimate (e.g. 1.0m) or interpolate between player heights. Simple first pass: project to ground plane (y=0).

**Reuse from existing code:**
- `generate_schematic_video.py` — two-pass detection pipeline, `prepare_keypoints()`, smoothing
- `schematic_renderer.py` — `precompute_camera_court_geometry()`, `transform_foot_to_schematic()`, court width functions
- `tennis-tracking/court_detector.py` — `CourtDetector` for homography
- `wasb_ball_positions.csv` — ball positions

### Part 2: Three.js — Interactive 3D viewer

New file: `viewer_3d/index.html` (self-contained, no build step)

**Components:**
1. **Court mesh**: Green/blue ground plane with white court lines, net as semi-transparent vertical plane
2. **Stick figures**: Cylinders/lines for bones, spheres for joints, using SKELETON_CONNECTIONS
3. **Ball**: Small yellow sphere
4. **Camera**: OrbitControls for 360-degree rotation, zoom, pan
5. **Animation**: Frame-by-frame playback with play/pause, scrub bar

**Libraries (CDN, no npm needed):**
- Three.js (r160+)
- OrbitControls

**Features:**
- Load JSON data exported from Part 1
- Animate at original FPS (50fps)
- Color-coded players (green = near, gold = far)
- Play/pause button, frame slider
- Auto-orbit option

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `export_3d_data.py` | **Create** | Python script: runs detection pipeline, exports JSON |
| `viewer_3d/index.html` | **Create** | Three.js viewer (self-contained HTML) |
| `tennis_match_3d.json` | **Generated** | Output from export script |

No existing files modified — this is additive.

## Implementation Steps

1. Create `export_3d_data.py`:
   - Reuse court detection + pose detection from `generate_schematic_video.py`
   - Add 2D→3D lifting using reference court → meters conversion
   - Smooth positions (reuse `smooth_positions`)
   - Load ball CSV, project to 3D court coordinates
   - Write JSON

2. Create `viewer_3d/index.html`:
   - Three.js scene with court ground plane + lines + net
   - Load JSON, create stick-figure meshes
   - Animate per frame with OrbitControls
   - Simple UI: play/pause, frame counter, speed control

3. Run and verify:
   ```bash
   # Export data
   python3 export_3d_data.py
   
   # Serve viewer
   python3 -m http.server 8000
   # Open http://localhost:8000/viewer_3d/index.html
   ```

## Verification
1. Run `python3 export_3d_data.py` — should produce `tennis_match_3d.json`
2. Open viewer in browser — should see court with two stick figures and ball
3. Drag to rotate 360 degrees — players should be upright from all angles
4. Play animation — movement should be smooth and match `schematic_output.mp4` timing
5. Player positions should correspond to court positions in original video
