
• Yes. Based on the paper assumptions, the demo assets in notes/screenshots/where_is_the_ball/data/tracknet_tid_desc.json,
and the sample visuals in notes/screenshots/federer_djokovic/image.png and notes/screenshots/federer_djokovic/Screenshot
2026-04-21 at 11.48.27 AM.png, these are the dataset/clipping fixes I would use.

Clip Rules

- Clip each sequence so it starts and ends with the ball near the ground or in a dead-ball state. This matches the
paper’s stated assumption in pdf/Where_Is_The_Ball_2025.pdf.
- Keep multiple internal hits and bounces inside one sequence. Do not split every projectile; the EoT network is supposed
to handle internal boundaries.
- Do not use arbitrary mid-flight crops. Your worst failures come from clips that start or end in the air.
- Prefer rally subsequences, not entire raw clips. The demo TrackNet selections are manually trimmed inside the source
clips; the chosen starts are often several frames after the clip begins in notes/screenshots/where_is_the_ball/data/
tracknet_tid_desc.json.

Exclude or Trim Aggressively

- Exclude the serve toss and any sequence that starts with the ball in the player’s hand. That violates the model’s
start-on-ground assumption.
- In practice, do not start a sample at the serve. Start after the first meaningful ground-bounded rally phase if you
want paper-consistent input.
- Exclude or downweight volleys / net exchanges where the ball changes direction near the net without a clean ground-
contact boundary. The paper itself shows this as a failure mode in pdf/Where_Is_The_Ball_2025.pdf.
- Exclude clips with replay cuts, zoom changes, camera pans, or non-standard broadcast angles.
- Exclude clips where the ball disappears for long stretches or is heavily occluded by players / scorebug / crowd.

Prefer These Tennis Patterns

- Baseline-to-baseline rallies from the standard high broadcast camera.
- Rallies where the ball clearly bounces on court and then continues.
- Clips where the whole rally stays inside one stable camera shot.
- Sequences with visible court lines and good camera calibration support.
- Exchanges whose start and end can be identified from bounce/dead-ball states, not guessed.

Camera / View Constraints

- Use only the fixed high, downward-facing broadcast view assumption the paper relies on.
- Avoid side cameras, low-angle cameras, extreme zooms, and highlight edits.
- Keep one calibration per stable shot; don’t mix frames from different zoom levels into one sample.

Length / Coverage

- Avoid very short snippets. The paper’s real TrackNet sequences span a range, but the useful ones are still complete
- Reject samples where the 2D track contains large jumps, identity swaps, or tracker hallucinations.
- Reject samples where the ball path is dominated by missing detections and interpolation.

Short Version

- Do not feed serves, tosses, or random 2-second crops.
- Feed rally subsequences that are ground -> multi-hit/multi-bounce rally fragment -> ground.
- Prefer clean high broadcast rallies, and drop volleys / odd net exchanges / edited shots.


• The full clip’s corrected camera looks almost exactly like a median Unity broadcast camera. I’m computing percentiles now
so I can tell you whether the training orientation is actually the wrong part of the setup.
• Ran python - <<'PY'
│ import json, glob, numpy as np, math
│ cams=sorted(glob.glob('/media/skr/storage/ten_bad/TennisDataset_game_5000/*/ep_*/camera.json'))
│ … +29 lines
└ fx real 1900.0 percentile 45.276220976781424
cam_y real 7.853956067028884 percentile 50.18014411529224
abs_cam_z real 32.37725819873043 percentile 50.0400320256205
pitch real 14.845825671443157 percentile 56.94555644515612

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Check Result

The Unity training camera orientation looks consistent with the corrected camera for S_Original_HL_clip_cropped.mp4.
Based on this check, the training dataset is not obviously wrong because of camera orientation.

Unity Dataset Orientation

- Unity cameras use the convention recorded in TennisDataset_game_5000/test/ep_000000/camera.json:
unity_worldToCamera_opengl_view (cam looks down -Z, +Y up).
- Representative Unity camera poses are:
- above ground: Y ≈ 3.0 to 13.0 m, median 7.84 m
- behind baseline: Z ≈ -45 to -20 m, median -32.37 m
- downward pitch: ≈ 3.9° to 32.5°, median 13.6°
- focal length fx ≈ 1001 to 3000, median 1993

So the Unity dataset is basically: high broadcast camera, centered or near-centered, behind the near baseline, looking
slightly downward toward the far court.

Real Clip Camera Orientation
Using LSTM_Model/inference_output/calibrated_camera_fixed.json, the corrected camera for the full clip is:

- camera position: x ≈ 0.03, y ≈ 7.85, z ≈ -32.38
- forward direction: [-0.0005, -0.2562, 0.9666]
- downward pitch: 14.85°
- focal length: fx = 1900

This is almost exactly in the middle of the Unity training distribution:

- fx: 45th percentile
- camera height: 50th percentile
- camera depth: 50th percentile
- pitch: 57th percentile

Important Caveat
The old file LSTM_Model/inference_output/calibrated_camera.json is still the wrong below-ground version:

- y ≈ -7.87
- same rough depth/focal length, but wrong reflection sign

If you used that file, the output would be wrong. If you used LSTM_Model/inference_output/calibrated_camera_fixed.json,
the camera orientation is fine.

What I noticed in the real clip
The first extracted frame of S_Original_HL_clip_cropped.mp4 is a serve sequence, not a ground-bounded rally fragment.
That is much more likely to be the reason the checkpoint performs badly than camera orientation.

Bottom line

- Unity dataset orientation: looks correct
- Real clip camera orientation: matches Unity very closely
- Likely remaining problem: sequence content / clipping assumptions, not camera angle



## Filters

Yes. On top of the current trim in ours_to_npz.py:15, these are the filters I would add to paper_npz_rev1/ours_game_5000_trimmed.

Must Add

- Boundary quality filter
Require both start and end frames to be near ground and not rising fast.
The current code already checks y < thresh and vy <= 1.0 in ours_to_npz.py:15, but I would make it stricter:
start/end y < 0.03–0.05 m, |vy| small or downward, and optionally low local height over a 2-3 frame window.
- No partial-flight filter
Reject samples where the trimmed start is clearly after launch or the trimmed end is clearly before dead-ball.
In practice:
if the first 5-10 frames immediately rise into a tall arc with no nearby ground contact before them, or the last 5-10 frames are
still mid-flight, reject.
- Serve removal filter
Exclude serve-style openings.
Use hitFrames from meta.json plus early trajectory shape:
if the first meaningful motion is a serve toss / strike before the first ground-bounded rally phase, reject or trim later.
For your current paper-consistent setup, sequences should start closer to a rally bounce/dead-ball boundary, not from ball-in-
hand or toss.
- Plane-point stability filter
Reject sequences whose plane-point parameterization explodes.
You already found this once. Add a hard filter on max |P| or on very small ray-plane denominators after parameterization.
This is one of the highest-value filters.
- Minimum usable length after trimming
Keep only sequences that still have enough context after all filtering.
Your current min_len=20 in ours_to_npz.py:27 is too permissive for tennis rallies.
I would keep something closer to the paper’s real TrackNet lower range, and practically prefer >= 50 frames for this dataset.

Strongly Recommended

- Trajectory-type filter
Exclude unusual cases the paper struggles with:
volley-only exchanges,
immediate net rebounds,
very short near-net redirections,
anything that does not have clean ground-bounded endpoints.
- Stroke-count sanity filter
Use numStrokes in meta.json.
Reject trivial clips with too little structure and extreme clips with too many unstable events.
A reasonable training subset is usually moderate-complexity rallies, not the weirdest ones.
- Height-range filter
Reject trajectories with unrealistic or extreme max height for your target deployment.
meta.json already has maxHeight.
This helps remove odd lobs / artifacts if they are rare and hurting generalization.
- Court-coverage filter
Reject samples where the ball spends too much time outside the normal playable region or near degenerate court geometry.
Use bboxMin/bboxMax from meta.json or x,z directly.
- 2D visibility / framing filter
Reject sequences where projected u,v are too close to the image boundary for too many frames.
Those usually cause noisier real-world tracking mismatch later.

Nice to Have

- Camera-subset filter
Since camera orientation is not the main problem for your tested real clip, this is optional, but you can narrow training to
cameras closer to standard broadcast:
moderate focal length,
medium height,
medium downward pitch,
centered behind-baseline view.
This reduces unnecessary view diversity if real deployment is narrow.
- Event-consistency filter
Use eot and hitFrames together.
Reject sequences where event flags are inconsistent with the trimmed boundaries or where trimming cuts too close to an event.
- Post-trim endpoint check
After trimming, verify:
first/last y near zero,
first/last eot not in a contradictory position,
no immediate pathological jump in xyz or uv.

What the Current Export Already Does

- Finds low-height contacts using y and vy
- Trims from first to last such contact
- Drops short outputs

That is useful, but it is only the first pass.

Practical Priority Order

1. Plane-point explosion filter
2. Stronger endpoint / no-partial-flight filter
3. Serve removal
4. Raise minimum post-trim length
5. Remove volleys / weird near-net exchanges
6. Optional camera-subset narrowing

If you want, I can next convert this into a precise per-episode acceptance checklist using only the fields you already have in
frames.csv and meta.json.

## already in place


• Yes, if you train from the raw Unity tennis episodes, they should be prepared first.

If you train from paper_npz_rev1/ours_game_5000_trimmed, that prep has already been applied at least partially. If you
train directly from TennisDataset_game_5000, then yes, trim/filter it before training.

What to do

- Trim each sample so it starts near ground and ends near ground.
- Keep internal hits and bounces inside the same sample.
- Remove serve toss / ball-in-hand / pre-rally waiting frames.
- Remove samples that still start or end mid-flight after trimming.
- Remove samples that become too short after trimming.
- Remove pathological samples with extreme plane-points or near-horizontal rays.
- Use the Unity 3D ground truth (y, vy, event flags if available) to do this, not image heuristics.

What not to do

- Do not keep full raw game clips as training samples.
you probably want.

What is already applied in ours_to_npz.py:15:
- It drops sequences that become too short in ours_to_npz.py:42.

So the exported set does roughly enforce:

- start near ground
- end near ground
- minimum usable length

That is why I said it is prepared partially.

What it does not explicitly do:

- no explicit serve filter
- no explicit volley / weird net exchange filter
- no semantic check that the sequence starts at the “right” rally moment, only the first low-y frame
- no semantic check that it ends at the “right” dead-ball moment, only the last low-y frame
- no explicit filtering of bad plane-point sequences
- no explicit filtering of unusual trajectory types that the paper struggles with

So “partially” means:

- Yes: endpoint trimming is there.
- No: full paper-style dataset curation is not obviously there.

In practice, paper_npz_rev1/ours_game_5000_trimmed is better than raw Unity episodes, but it is not the same as saying “the
dataset is fully curated and guaranteed paper-consistent.”

If you want, I can next give you a precise list of extra filters to add on top of ours_game_5000_trimmed.


› yes show the list of filter that need to be added to ours_game_5000_trimmed.


• I’m checking what raw episode fields are available in the Unity export so the filter list is based on signals you actually have,
not hypothetical labels.