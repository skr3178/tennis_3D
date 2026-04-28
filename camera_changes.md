Here's what needs to change in Unity, based on the current code:

## Current Camera Setup (Completely Static)

In `GameSceneBuilder.cs` (lines 33-41), the camera is hardcoded:
- **Position:** `(0, 14 + 1.295, -30)` → `(0, 15.3, -30)`
- **Look-at:** `(0, 1.295, 0)` (center court)
- **FOV:** 28°

This is set once at scene build time (lines 94-96) and never changes between episodes.

## What Needs to Change

### File 1: `GameSceneBuilder.cs`
Make camera params settable instead of hardcoded — add a public method like `SetCamera(Vector3 pos, Vector3 lookAt, float fov)` that the episode runner can call before each episode.

### File 2: `GameEpisodeRunner.cs`
Add camera randomization ranges as configurable fields:
- **Height:** 5–15m (currently fixed at 14m)
- **Distance behind baseline:** 20–40m (currently fixed at 30m)
- **Lateral offset:** -3 to +3m (currently fixed at 0)
- **FOV:** 20–35° (currently fixed at 28°)

At the start of each episode, sample random camera params and call the new `SetCamera()` method.

### File 3: `TrajectoryRecorder.cs`
Already handles this correctly — it snapshots the camera's current position/intrinsics at recording start (lines 79-95), so as long as the camera moves before recording begins, the saved `camera.json` will reflect the new position.

### File 4: CLI args in `GameDatasetRunner.cs`
Add optional CLI flags for camera range overrides (e.g., `-camHeightMin 5 -camHeightMax 15`) so we can control diversity from the shell script without recompiling.

---

## What Does NOT Need to Change

- **Ball physics / bot behavior** — already randomized via seeds
- **TrajectoryRecorder** — already captures whatever camera state exists
- **Resolution** — keep at 1280×720 to match real video
- **The existing 1000 episodes** — still valid, just from one camera angle

The total code change is ~50 lines across 2-3 files. Want me to write the modifications?