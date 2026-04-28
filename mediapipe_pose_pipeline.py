"""
Alternative pose pipeline using MediaPipe Pose Landmarker.

Runs in PARALLEL to the existing VideoPose3D pipeline — does not modify or
replace any existing files. Outputs `tennis_match_3d_mediapipe.json` for the
companion `viewer_3d_mediapipe/` viewer so the two pipelines can be compared.

Pipeline:
  1. Reuse `PoseDetector` + `PlayerTracker` (read-only import) to get per-frame
     near/far player bboxes and to track them across frames.
  2. Reuse `CourtDetector` (read-only import) for the camera→court homography
     so player feet can be projected to court meters.
  3. For each player per frame, crop the bbox (with padding, plus upscaling for
     the small far player) and run MediaPipe Pose Landmarker on the crop.
  4. Take MediaPipe's `pose_world_landmarks` (33 joints, meters, hip-rooted),
     map 33 → COCO 17, scale to PLAYER_HEIGHT_M, anchor feet on the ground at
     the homography-projected court foot position.
  5. Reuse `wasb_ball_positions.csv` for ball trajectory (same as exporter).
  6. Write JSON in the same schema as `tennis_match_3d.json`.

Coordinate system (matches existing viewer):
  X = lateral (across court), +X = right
  Y = height (up), 0 = ground
  Z = depth, 0 = net, +Z = near side, -Z = far side
"""
import sys
import os
import json
import csv
import cv2
import numpy as np
from scipy.signal import savgol_filter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker  # noqa: E402

VIDEO = 'S_Original_HL_clip_cropped.mp4'
BALL_CSV = 'wasb_ball_positions.csv'
OUTPUT_JSON = 'tennis_match_3d_mediapipe.json'
MP_MODEL = 'models/pose_landmarker_heavy.task'

SMOOTH_WINDOW = 11
SMOOTH_POLY = 2

COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
NET_HEIGHT = 1.07

REF_LEFT, REF_RIGHT = 286, 1379
REF_TOP, REF_BOTTOM = 561, 2935
REF_NET_Y = 1748
REF_CX = (REF_LEFT + REF_RIGHT) / 2
REF_WIDTH = REF_RIGHT - REF_LEFT
REF_HEIGHT = REF_BOTTOM - REF_TOP

PLAYER_HEIGHT_M = 1.80

# COCO skeleton bones, plus two extra "neck" bones from nose (0) to each
# shoulder (5, 6) so the head doesn't visually float free in the 3D viewer.
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),                     # neck (nose → shoulders)
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# MediaPipe (33) → COCO (17). MediaPipe indices, in COCO order.
MP_TO_COCO = [
    0,   # nose
    2,   # left_eye  (MP center-eye)
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

# Per-player crop padding (fraction of bbox).
# Wide pad (1.0) gives MediaPipe more body context; combined with VIDEO
# running mode this lifts far-player coverage from ~40% to ~97%. Upscaling
# pre-MP actually hurts because MP downsizes to 256² internally — pre-upscaled
# bicubic just softens the source.
NEAR_PAD = 1.0
FAR_PAD = 1.0
FAR_UPSCALE = 1


def smooth_skeletons(skel_list, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Smooth a list of (17,3) skeletons or None across consecutive runs.

    MediaPipe per-frame predictions jitter — especially in z (depth) since
    monocular depth is under-constrained. Apply Savitzky-Golay along time on
    each joint x/y/z independently, but only within unbroken runs of
    detections so we don't smear across real gaps.
    """
    n = len(skel_list)
    if n == 0:
        return skel_list
    runs = []
    i = 0
    while i < n:
        if skel_list[i] is not None:
            j = i
            while j < n and skel_list[j] is not None:
                j += 1
            runs.append((i, j))  # half-open
            i = j
        else:
            i += 1
    for s, e in runs:
        L = e - s
        if L < window:
            continue
        stack = np.stack(skel_list[s:e], axis=0)  # (L, 17, 3)
        smoothed = savgol_filter(stack, window, poly, axis=0)
        for k in range(L):
            skel_list[s + k] = smoothed[k]
    return skel_list


def smooth_positions(positions, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    n = len(positions)
    xs = np.array([p[0] if p is not None else np.nan for p in positions])
    ys = np.array([p[1] if p is not None else np.nan for p in positions])
    valid = ~np.isnan(xs)
    if np.sum(valid) < window:
        return positions
    indices = np.arange(n)
    if np.any(~valid):
        xs[~valid] = np.interp(indices[~valid], indices[valid], xs[valid])
        ys[~valid] = np.interp(indices[~valid], indices[valid], ys[valid])
    if len(xs) >= window:
        xs_s = savgol_filter(xs, window, poly)
        ys_s = savgol_filter(ys, window, poly)
    else:
        xs_s, ys_s = xs, ys
    return [(float(xs_s[i]), float(ys_s[i])) if positions[i] is not None else None
            for i in range(n)]


def ref_to_meters(ref_x, ref_y):
    x_m = (ref_x - REF_CX) / REF_WIDTH * COURT_WIDTH
    z_m = (ref_y - REF_NET_Y) / REF_HEIGHT * COURT_LENGTH
    return x_m, z_m


def get_foot_cam(det):
    """Foot pixel position from a YOLO detection (avg ankles, fallback bbox bottom)."""
    kps = det['keypoints']
    ankles = [kps[j] for j in [15, 16] if kps[j, 2] > 0.3]
    if ankles:
        return np.mean([a[:2] for a in ankles], axis=0)
    return np.array([(det['bbox'][0] + det['bbox'][2]) / 2, det['bbox'][3]])


def crop_with_padding(frame, bbox, pad_frac):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1p = int(max(0, x1 - bw * pad_frac))
    y1p = int(max(0, y1 - bh * pad_frac))
    x2p = int(min(w, x2 + bw * pad_frac))
    y2p = int(min(h, y2 + bh * pad_frac))
    if x2p <= x1p or y2p <= y1p:
        return None
    return frame[y1p:y2p, x1p:x2p]


def run_mediapipe(landmarker, crop_bgr, ts_ms, upscale=1):
    """Run MediaPipe Pose Landmarker (VIDEO mode) on a BGR crop.

    `ts_ms` must be monotonically increasing per landmarker. The caller MUST
    use a separate landmarker per role so temporal state isn't crossed.
    """
    if upscale != 1:
        ch, cw = crop_bgr.shape[:2]
        crop_bgr = cv2.resize(crop_bgr, (cw * upscale, ch * upscale),
                              interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    if not result.pose_world_landmarks:
        return None, None
    world = result.pose_world_landmarks[0]
    visibility = np.array([lm.visibility for lm in world], dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)
    return coords, visibility


def transform_skeleton(world_33, foot_x_m, foot_z_m):
    """
    MediaPipe world landmarks (33,3) in meters, hip-rooted, Y-down →
    COCO-ordered (17,3) world meters, Y-up, anchored at (foot_x_m, *, foot_z_m).
    """
    # 1. Y-flip so up is positive
    s = world_33.copy()
    s[:, 1] = -s[:, 1]

    # 2. Scale to PLAYER_HEIGHT_M based on observed head→ankle span
    head_y = s[0, 1]                      # nose
    feet_y = (s[27, 1] + s[28, 1]) / 2    # ankles
    span = head_y - feet_y
    if span < 0.1:
        span = 1.0
    # Real human nose-to-ankle ≈ 0.93 × total height
    scale = (PLAYER_HEIGHT_M * 0.93) / span
    s = s * scale

    # 3. Drop feet to ground (Y=0). Anchor at the LOWEST foot point across
    #    ankles + heels + foot_index so neither foot ever sinks below the
    #    ground plane (one-foot-up frames previously left one ankle at -Y).
    foot_indices = [27, 28, 29, 30, 31, 32]  # ankles, heels, foot_index
    foot_y = float(min(s[k, 1] for k in foot_indices))
    s[:, 1] -= foot_y

    # 4. Translate XZ to court foot position (depth z gets the local variation)
    s[:, 0] += foot_x_m
    s[:, 2] = foot_z_m + s[:, 2]

    # 5. Map 33 → COCO 17
    coco = s[MP_TO_COCO]
    return coco


def main():
    # Court detector lives inside tennis-tracking/ — temporarily chdir to import.
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    from court_detector import CourtDetector
    os.chdir(ORIG_DIR)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Input: {w}x{h}, {fps}fps')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    print(f'Read {n_frames} frames')

    # Court homography
    print('Detecting court...')
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    cd = CourtDetector()
    cd.detect(frames[0])
    os.chdir(ORIG_DIR)
    cwm = cd.court_warp_matrix[0]
    cam_to_ref_h = cv2.invert(cwm)[1]

    # Player bboxes
    print('Tracking players (YOLO)...')
    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    near_bbox = [None] * n_frames
    far_bbox = [None] * n_frames
    near_foot = [None] * n_frames
    far_foot = [None] * n_frames
    for i in range(n_frames):
        players = tracker.update(frames[i], conf=0.1)
        for role, bbox_store, foot_store in [
            ('near', near_bbox, near_foot),
            ('far', far_bbox, far_foot),
        ]:
            d = players.get(role)
            if d is not None:
                bbox_store[i] = d['bbox'].tolist()
                foot_store[i] = get_foot_cam(d).tolist()
        if i % 100 == 0:
            print(f'  Frame {i}/{n_frames}')

    # Smooth foot pixels then project to court
    print('Smoothing + projecting feet to court...')
    near_foot = smooth_positions(near_foot)
    far_foot = smooth_positions(far_foot)

    near_court = [None] * n_frames
    far_court = [None] * n_frames
    for i in range(n_frames):
        for cam_store, court_store in [
            (near_foot, near_court),
            (far_foot, far_court),
        ]:
            if cam_store[i] is not None:
                fc = cam_store[i]
                p = np.array([[[fc[0], fc[1]]]], dtype=np.float32)
                tp = cv2.perspectiveTransform(p, cam_to_ref_h)
                court_store[i] = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))
    near_court = smooth_positions(near_court)
    far_court = smooth_positions(far_court)

    # MediaPipe Pose Landmarker (VIDEO mode — one landmarker per role so the
    # temporal tracker doesn't get confused interleaving near/far inputs).
    print('Initialising MediaPipe Pose Landmarker (VIDEO mode, per-role)...')
    def make_landmarker():
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return mp_vision.PoseLandmarker.create_from_options(opts)
    landmarker_near = make_landmarker()
    landmarker_far = make_landmarker()

    print('Running MediaPipe per player crop...')
    near_skel = [None] * n_frames     # (17,3) in court meters
    near_conf = [None] * n_frames     # (17,)
    far_skel = [None] * n_frames
    far_conf = [None] * n_frames

    for i in range(n_frames):
        ts_ms = int(i * 1000 / fps)
        for role, bbox_store, court_store, skel_store, conf_store, pad, upscale, lm in [
            ('near', near_bbox, near_court, near_skel, near_conf,
             NEAR_PAD, 1, landmarker_near),
            ('far', far_bbox, far_court, far_skel, far_conf,
             FAR_PAD, FAR_UPSCALE, landmarker_far),
        ]:
            if bbox_store[i] is None or court_store[i] is None:
                # Still feed a frame to keep timestamps monotonic — pass a
                # blank crop so the tracker sees "no person" rather than a
                # timestamp gap that would invalidate state.
                blank = np.zeros((64, 64, 3), dtype=np.uint8)
                run_mediapipe(lm, blank, ts_ms, upscale=1)
                continue
            crop = crop_with_padding(frames[i], bbox_store[i], pad)
            if crop is None or crop.size == 0:
                blank = np.zeros((64, 64, 3), dtype=np.uint8)
                run_mediapipe(lm, blank, ts_ms, upscale=1)
                continue
            world_33, vis_33 = run_mediapipe(lm, crop, ts_ms, upscale=upscale)
            if world_33 is None:
                continue
            foot_x, foot_z = court_store[i]
            coco_kps = transform_skeleton(world_33, foot_x, foot_z)
            coco_conf = vis_33[MP_TO_COCO]
            skel_store[i] = coco_kps
            conf_store[i] = coco_conf
        if i % 100 == 0:
            print(f'  Frame {i}/{n_frames}')

    landmarker_near.close()
    landmarker_far.close()

    # Backward pass for the far player. The forward VIDEO-mode tracker
    # occasionally loses lock; running a second pass in the reverse temporal
    # direction (with its own landmarker so timestamps stay monotonic) gives
    # a second chance to acquire on each frame. We then merge by per-frame
    # mean visibility — best-of-two.
    fwd_far_count = sum(1 for s in far_skel if s is not None)
    print(f'Forward-only far hits: {fwd_far_count}/{n_frames}')
    print('Backward MediaPipe pass for far player...')
    landmarker_far_bwd = make_landmarker()
    recovered = 0
    overridden = 0
    for k in range(n_frames):
        i = n_frames - 1 - k
        ts_ms = int(k * 1000 / fps)  # monotonic for the reverse stream
        if far_bbox[i] is None or far_court[i] is None:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            run_mediapipe(landmarker_far_bwd, blank, ts_ms, upscale=1)
            continue
        crop = crop_with_padding(frames[i], far_bbox[i], FAR_PAD)
        if crop is None or crop.size == 0:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            run_mediapipe(landmarker_far_bwd, blank, ts_ms, upscale=1)
            continue
        world_33, vis_33 = run_mediapipe(landmarker_far_bwd, crop, ts_ms,
                                          upscale=FAR_UPSCALE)
        if world_33 is None:
            continue
        foot_x, foot_z = far_court[i]
        coco_kps = transform_skeleton(world_33, foot_x, foot_z)
        coco_conf = vis_33[MP_TO_COCO]
        if far_skel[i] is None:
            far_skel[i] = coco_kps
            far_conf[i] = coco_conf
            recovered += 1
        elif float(np.mean(coco_conf)) > float(np.mean(far_conf[i])):
            far_skel[i] = coco_kps
            far_conf[i] = coco_conf
            overridden += 1
        if k % 100 == 0:
            print(f'  bwd frame {k}/{n_frames}')
    landmarker_far_bwd.close()
    final_far = sum(1 for s in far_skel if s is not None)
    print(f'  newly recovered: {recovered}')
    print(f'  overrode fwd:    {overridden}')
    print(f'  far total:       {final_far}/{n_frames}')

    # Temporal smoothing of joints to reduce per-frame jitter.
    print('Smoothing skeleton joints (Savitzky-Golay)...')
    smooth_skeletons(near_skel)
    smooth_skeletons(far_skel)

    # Assemble per-frame data
    print('Assembling JSON...')
    frames_data = []
    for i in range(n_frames):
        players = []
        for role, skel, conf, pid in [
            ('far', far_skel[i], far_conf[i], 1),
            ('near', near_skel[i], near_conf[i], 2),
        ]:
            if skel is None:
                continue
            kps_list = [[round(float(skel[j, 0]), 4),
                         round(float(skel[j, 1]), 4),
                         round(float(skel[j, 2]), 4)] for j in range(17)]
            conf_list = [round(float(c), 3) for c in conf]
            players.append({
                'id': pid,
                'role': role,
                'keypoints_3d': kps_list,
                'confidence': conf_list,
            })
        frames_data.append({'frame': i, 'players': players, 'ball': None})

    # Ball — same approach as the existing exporter (ground projection).
    print('Projecting ball...')
    ball_cam = [None] * n_frames
    if os.path.exists(BALL_CSV):
        with open(BALL_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fi = int(row['frame'])
                if fi < n_frames and int(row['visible']):
                    ball_cam[fi] = (float(row['x']), float(row['y']))

        ball_3d = [None] * n_frames
        for i in range(n_frames):
            if ball_cam[i] is not None:
                p = np.array([[[ball_cam[i][0], ball_cam[i][1]]]], dtype=np.float32)
                tp = cv2.perspectiveTransform(p, cam_to_ref_h)
                bx, bz = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))
                ball_3d[i] = (bx, 0.5, bz)

        ball_xz = [(b[0], b[2]) if b is not None else None for b in ball_3d]
        ball_xz = smooth_positions(ball_xz, window=5, poly=2)
        for i in range(n_frames):
            if ball_xz[i] is not None:
                frames_data[i]['ball'] = [
                    round(ball_xz[i][0], 4), 0.5, round(ball_xz[i][1], 4)
                ]
    else:
        print(f'  {BALL_CSV} not found — ball field left null.')

    output = {
        'fps': float(fps),
        'total_frames': n_frames,
        'court': {'width': COURT_WIDTH, 'length': COURT_LENGTH, 'net_height': NET_HEIGHT},
        'skeleton_connections': COCO_SKELETON_CONNECTIONS,
        'keypoint_names': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        ],
        'source': 'mediapipe_pose_landmarker_heavy',
        'frames': frames_data,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f)

    size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    print(f'\nWrote {OUTPUT_JSON} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
