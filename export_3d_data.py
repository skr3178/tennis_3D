"""
Export 3D skeleton + ball data for the Three.js viewer.

Pipeline:
  1. Pose detection: track players in 2D, project feet to court meters via homography
  2. Load VideoPose3D per-player 3D outputs (root-relative H36M skeleton)
  3. Combine: anchor local skeleton at homography court position, scale to real height
  4. Convert H36M 17-joint format → COCO 17-joint format (for viewer compatibility)
  5. Ball: project to ground plane via homography (approximate, ignores height)
  6. Write JSON

Coordinate system (meters, real-world tennis court):
  X = lateral (across court), 0 = court center, positive = right
  Y = height (up from ground), 0 = ground
  Z = depth (along court), 0 = net, positive = near side, negative = far side
"""
import sys
import os
import json
import csv
import cv2
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker

VIDEO = 'S_Original_HL_clip_cropped.mp4'
BALL_CSV = 'wasb_ball_positions.csv'
OUTPUT_JSON = 'tennis_match_3d.json'
VP3D_NEAR = 'videopose3d_output/output_near.npy'
VP3D_FAR = 'videopose3d_output/output_far.npy'

SMOOTH_WINDOW = 7
SMOOTH_POLY = 2

# Real tennis court dimensions (meters)
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
NET_HEIGHT = 1.07

# Reference court coordinate system
REF_LEFT, REF_RIGHT = 286, 1379
REF_TOP, REF_BOTTOM = 561, 2935
REF_NET_Y = 1748
REF_CX = (REF_LEFT + REF_RIGHT) / 2
REF_WIDTH = REF_RIGHT - REF_LEFT
REF_HEIGHT = REF_BOTTOM - REF_TOP

# Typical player height (meters)
PLAYER_HEIGHT_M = 1.80

# COCO skeleton connections (for the viewer's skeleton bones)
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),                   # head
    (5, 6),                                            # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),                   # arms
    (5, 11), (6, 12), (11, 12),                        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
]


def smooth_positions(positions, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Smooth a list of (x, y) or None positions."""
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
    """Reference court coords → court meters."""
    x_m = (ref_x - REF_CX) / REF_WIDTH * COURT_WIDTH
    z_m = (ref_y - REF_NET_Y) / REF_HEIGHT * COURT_LENGTH
    return x_m, z_m


def h36m_to_coco(h36m_kps):
    """Convert VideoPose3D H36M 17-joint layout to COCO 17-joint layout.

    H36M layout (index -> joint):
      0=hip_root, 1=r_hip, 2=r_knee, 3=r_foot, 4=l_hip, 5=l_knee, 6=l_foot,
      7=spine, 8=thorax, 9=neck/nose, 10=head,
      11=l_shoulder, 12=l_elbow, 13=l_wrist,
      14=r_shoulder, 15=r_elbow, 16=r_wrist

    COCO layout (index -> joint):
      0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear,
      5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow, 9=l_wrist, 10=r_wrist,
      11=l_hip, 12=r_hip, 13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle
    """
    h = h36m_kps
    coco = np.zeros((17, 3), dtype=np.float32)
    # Face: approximate eyes/ears from neck and head positions
    nose = h[9]
    head = h[10]
    head_vec = head - nose  # from neck to top-of-head
    coco[0] = nose                              # nose
    coco[1] = nose + 0.4 * head_vec             # left eye (slightly up)
    coco[2] = nose + 0.4 * head_vec             # right eye
    coco[3] = head                              # left ear
    coco[4] = head                              # right ear
    # Upper body
    coco[5] = h[11]   # left shoulder
    coco[6] = h[14]   # right shoulder
    coco[7] = h[12]   # left elbow
    coco[8] = h[15]   # right elbow
    coco[9] = h[13]   # left wrist
    coco[10] = h[16]  # right wrist
    # Lower body
    coco[11] = h[4]   # left hip
    coco[12] = h[1]   # right hip
    coco[13] = h[5]   # left knee
    coco[14] = h[2]   # right knee
    coco[15] = h[6]   # left ankle
    coco[16] = h[3]   # right ankle
    return coco


def transform_vp3d_skeleton(h36m_skel, foot_x_m, foot_z_m, facing_far=False):
    """
    Transform VideoPose3D output (root-relative H36M, Y-down) to world space.

    Args:
      h36m_skel: (17, 3) array in VP3D coords (meters, Y-down, root at origin)
      foot_x_m, foot_z_m: player's foot position on court in meters
      facing_far: True if player is on far side of net (might need X flip)

    Returns: (17, 3) COCO-ordered 3D keypoints in world meters (Y-up)
    """
    # 1. Flip Y axis (VP3D: Y-down → viewer: Y-up)
    skel = h36m_skel.copy()
    skel[:, 1] = -skel[:, 1]

    # 2. Scale to real human height
    # H36M model output has body span roughly 1m (head-to-feet)
    # Real player is PLAYER_HEIGHT_M meters
    # Compute actual body span from head (10) to feet (3, 6) in VP3D output
    head_y = skel[10, 1]
    feet_y = (skel[3, 1] + skel[6, 1]) / 2
    body_span = head_y - feet_y  # after Y-flip, head_y > feet_y (head higher)
    if body_span < 0.1:
        body_span = 0.5  # fallback
    scale = PLAYER_HEIGHT_M / body_span
    skel = skel * scale

    # 3. Anchor: move feet to ground (Y=0), place at homography court position
    # After scaling, find average foot Y position; subtract it so feet are at Y=0
    foot_y = (skel[3, 1] + skel[6, 1]) / 2
    skel[:, 1] -= foot_y  # feet now at Y=0, head up

    # 4. Translate X, Z to court position
    skel[:, 0] += foot_x_m
    skel[:, 2] = foot_z_m + skel[:, 2] * 0.1  # tiny Z variation between joints

    # 5. Convert to COCO layout
    return h36m_to_coco(skel)


def prepare_keypoints(det_data, role):
    """Same helper: synthesize far-player lower body if missing."""
    kps = det_data['keypoints'].copy()
    bbox = det_data['bbox']
    foot_cam = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
    if role == 'far':
        bbox_cx = (bbox[0] + bbox[2]) / 2
        bbox_bottom = bbox[3]
        bbox_h = bbox[3] - bbox[1]
        synth = {
            11: (bbox_cx - bbox_h * 0.06, bbox[1] + bbox_h * 0.55),
            12: (bbox_cx + bbox_h * 0.06, bbox[1] + bbox_h * 0.55),
            13: (bbox_cx - bbox_h * 0.05, bbox[1] + bbox_h * 0.75),
            14: (bbox_cx + bbox_h * 0.05, bbox[1] + bbox_h * 0.75),
            15: (bbox_cx - bbox_h * 0.05, bbox_bottom),
            16: (bbox_cx + bbox_h * 0.05, bbox_bottom),
        }
        for idx, (sx, sy) in synth.items():
            if kps[idx, 2] < 0.3:
                kps[idx] = [sx, sy, 0.35]
    return kps, foot_cam


def main():
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

    # Court detection
    print('Detecting court...')
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    cd = CourtDetector()
    cd.detect(frames[0])
    os.chdir(ORIG_DIR)
    cwm = cd.court_warp_matrix[0]
    cam_to_ref_h = cv2.invert(cwm)[1]
    print('  Court detected')

    # Pose detection
    print('Detecting players...')
    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    near_foot = [None] * n_frames
    far_foot = [None] * n_frames
    for i in range(n_frames):
        players = tracker.update(frames[i], conf=0.1)
        for role, store in [('near', near_foot), ('far', far_foot)]:
            det_data = players.get(role)
            if det_data is not None:
                _, foot_cam = prepare_keypoints(det_data, role)
                store[i] = foot_cam.tolist()
        if i % 100 == 0:
            print(f'  Frame {i}/{n_frames}')

    # Smooth foot positions in camera space
    print('Smoothing...')
    near_foot = smooth_positions(near_foot)
    far_foot = smooth_positions(far_foot)

    # Project feet to reference court → meters
    near_court = [None] * n_frames
    far_court = [None] * n_frames
    for i in range(n_frames):
        for role, cam_store, court_store in [
            ('near', near_foot, near_court),
            ('far', far_foot, far_court),
        ]:
            if cam_store[i] is not None:
                fc = cam_store[i]
                p = np.array([[[fc[0], fc[1]]]], dtype=np.float32)
                tp = cv2.perspectiveTransform(p, cam_to_ref_h)
                x_m, z_m = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))
                court_store[i] = (x_m, z_m)

    # Smooth court positions (x, z in meters)
    near_court = smooth_positions(near_court)
    far_court = smooth_positions(far_court)

    # Load VideoPose3D outputs
    print('Loading VideoPose3D skeletons...')
    vp3d_near = np.load(VP3D_NEAR)  # (767, 17, 3) H36M, root-relative
    vp3d_far = np.load(VP3D_FAR)
    assert vp3d_near.shape[0] == n_frames, f'VP3D near has {vp3d_near.shape[0]} frames, expected {n_frames}'
    assert vp3d_far.shape[0] == n_frames

    # Combine: anchor VP3D skeleton at homography court position
    print('Combining VP3D + homography...')
    frames_data = []
    for i in range(n_frames):
        players = []
        for role, vp3d, court_pos, pid in [
            ('far', vp3d_far, far_court, 1),
            ('near', vp3d_near, near_court, 2),
        ]:
            if court_pos[i] is None:
                continue
            foot_x, foot_z = court_pos[i]
            coco_kps = transform_vp3d_skeleton(vp3d[i], foot_x, foot_z,
                                                facing_far=(role == 'far'))
            kps_list = [[round(float(coco_kps[j, 0]), 4),
                         round(float(coco_kps[j, 1]), 4),
                         round(float(coco_kps[j, 2]), 4)] for j in range(17)]
            players.append({
                'id': pid,
                'role': role,
                'keypoints_3d': kps_list,
                'confidence': [0.9] * 17,  # VP3D doesn't give per-joint conf
            })
        frames_data.append({'frame': i, 'players': players, 'ball': None})

    # Ball (simple ground projection, same as before)
    print('Processing ball...')
    ball_cam = [None] * n_frames
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
            ball_3d[i] = (bx, 0.5, bz)  # placeholder height 0.5m

    # Smooth ball (x, z only; keep y constant for now)
    ball_xz = [(b[0], b[2]) if b is not None else None for b in ball_3d]
    ball_xz = smooth_positions(ball_xz, window=5, poly=2)
    for i in range(n_frames):
        if ball_xz[i] is not None:
            frames_data[i]['ball'] = [
                round(ball_xz[i][0], 4), 0.5, round(ball_xz[i][1], 4)
            ]

    # Write JSON
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
        'frames': frames_data,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f)

    size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    print(f'\nExported {OUTPUT_JSON} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
