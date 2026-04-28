claude --resume 9cf1b4df-d5ad-4aa4-8ccc-53716f6f46bf

claude --resume dbcc3b4c-d5f2-42bb-82eb-5c5d1659b924


Root Cause: Exploding Plane-Point Parameterization

The raw xyz and camera intrinsics are fine — no NaN, no crazy values. The problem is in
the P (plane-point) parameterization computed at training time. When the ball is near the
camera's height, the ray becomes nearly horizontal, and the ground-plane intersection
explodes:

- Median max|P|: 23.7
- 99th percentile: 514
- Worst case: 117,204 (seq_00319)

14 sequences have max|P| > 1,000 — these cause the loss spikes. A single frame with
P=117,000 produces L_3D in the billions.

The correlation between max|P| and 1/min|d_world_y| is r=0.925 — confirming it's the
near-horizontal ray issue.

Fix Options

1. Quick: Remove the 14 worst sequences (0.6% of data), retrain
2. Better: Clamp P values to a reasonable range (e.g., |P| < 50) in the dataset loader
3. Best: Reject/interpolate frames where |d_world_y| < 1.0 during parameterization




## fix
Yes, fix 1 is easier. Here's the issue and the fix:                                        
                                                    
The problem: PnP calibrates using court points that ALL have Y=0 (flat on the ground). This
creates a Y-reflection ambiguity — the camera could be above or below the court plane and
project identically onto those points. OpenCV's PnP is choosing the "below ground" solution
(Y=-7.85), but the model expects "above ground" (Y=+7.85 like Unity).                     
                                                                                        
The fix: After the OpenGL flip in eval_tracknet.py, negate column 1 of the rotation matrix 
in E. This:                                                                                
- Preserves 2D projection for all Y=0 court points (unchanged)                             
- Flips the camera position to Y>0 (above ground)                                          
- Makes ray-ground intersections match what the model saw during training
                                                                                        
It's a one-line fix in calibrate_camera_from_image:                                        
E[:3, 1] *= -1  # fix Y-reflection ambiguity                                               
                                                                                        
Right after line 155 where E is constructed. Want me to apply it and test?                 

claude --resume "parallelize-unity-data-collection"             




• Debug Steps
    inference_output/2s_calibrated_camera_fixed.json as stale/buggy and do not feed it to inference.
2. Keep the camera fix separate from the trajectory issue. The camera convention/sign bug is fixed in the corrected run; the
    remaining problem is mostly sequence preparation, not calibration.
3. Prepare inference clips so each input rally is ground-bounded:
    first frame near ground contact, last frame near ground contact, with any number of internal hits/bounces allowed. This matches
    the paper assumption and your training conversion in ours_to_npz.py:5.
4. Do not use arbitrary short crops that start or end mid-flight. Your current 2-second result is failing mainly because both
    detected rallies were untrimmed in LSTM_Model/infer_video_segmented.py:157 and LSTM_Model/inference_output/2s_fixed_camera/
    rally_segmented_detail.json.
5. Replace the current bounce trimming heuristic mentally when debugging: the hardcoded pixel thresholds in LSTM_Model/
    infer_video_segmented.py:97 are not reliable across viewpoints. When reviewing a clip, check rally boundaries against court
    geometry / visible bounce events, not just image y thresholds.
6. Before trusting an inference run, verify three sanity checks:
    corrected camera file,
    rally starts/ends near ground,
    no obvious partial-arc crop.
7. Compare the segment against training distribution:
    your checkpoint was trained on ours_game_5000_trimmed in training_5k_v2.log:1, where sequences were trimmed to ground-contact
    boundaries. If inference input does not look like that, poor output is expected.
8. After rerunning on a proper rally segment, inspect:
    endpoint heights should be near zero,
    EoT peaks should occur mainly at internal direction changes,
    trajectory should stay physically plausible on court.
9. If results are still bad after using correct camera + proper rally boundaries, then the next suspect is real-vs-synthetic domain
    gap, especially rare cases like volleys or unusual bounces, not the core LSTM architecture.

Bottom line
The debugging order is: correct camera file -> correct rally boundaries -> paper-consistent input sequence -> then judge model
quality.
