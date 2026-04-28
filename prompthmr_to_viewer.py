"""
Convert PromptHMR results.pkl → viewer JSON in the same schema as the
MediaPipe / VideoPose3D pipelines.

This is a *parallel* pipeline — it does not modify any existing file or replace
the VideoPose3D outputs. It loads the SMPL-X parameters PromptHMR produces,
runs the SMPL-X layer to get per-frame 3D joints, picks 17 in COCO order, and
anchors each track onto the tennis court via the same camera→court homography
the existing `export_3d_data.py` uses.

Inputs:
  - PromptHMR/results/<basename>/results.pkl  (joblib dump)
  - The original video (for court detection & frame indexing)
  - wasb_ball_positions.csv (ball trajectory, projected to court)

Output:
  tennis_match_3d_prompthmr.json  (consumed by viewer_3d_prompthmr/)
"""
import sys
import os
import json
import csv
import argparse
import numpy as np
import joblib
import torch
import cv2

PROMPTHMR_DIR = '/media/skr/storage/ten_bad/PromptHMR'
sys.path.insert(0, PROMPTHMR_DIR)
sys.path.insert(0, '/media/skr/storage/ten_bad/tennis-tracking')

DEFAULT_VIDEO = '/media/skr/storage/ten_bad/S_Original_HL_clip_cropped.mp4'
DEFAULT_RESULTS_PKL = os.path.join(PROMPTHMR_DIR, 'results', 'S_Original_HL_clip_cropped', 'results.pkl')
DEFAULT_BALL_CSV = '/media/skr/storage/ten_bad/wasb_ball_positions.csv'
DEFAULT_OUTPUT_JSON = '/media/skr/storage/ten_bad/tennis_match_3d_prompthmr.json'
SMPLX_PATH = os.path.join(PROMPTHMR_DIR, 'data/body_models/smplx')

# Court calibration constants (same as MediaPipe pipeline).
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
NET_HEIGHT = 1.07
REF_LEFT, REF_RIGHT = 286, 1379
REF_TOP, REF_BOTTOM = 561, 2935
REF_NET_Y = 1748
REF_CX = (REF_LEFT + REF_RIGHT) / 2
REF_WIDTH = REF_RIGHT - REF_LEFT
REF_HEIGHT = REF_BOTTOM - REF_TOP

# SMPL-X joint indices → COCO 17 order.
SMPLX_TO_COCO = [
    55,  # nose
    57,  # left_eye
    56,  # right_eye
    59,  # left_ear
    58,  # right_ear
    16,  # left_shoulder
    17,  # right_shoulder
    18,  # left_elbow
    19,  # right_elbow
    20,  # left_wrist
    21,  # right_wrist
    1,   # left_hip
    2,   # right_hip
    4,   # left_knee
    5,   # right_knee
    7,   # left_ankle
    8,   # right_ankle
]

COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def ref_to_meters(ref_x, ref_y):
    x_m = (ref_x - REF_CX) / REF_WIDTH * COURT_WIDTH
    z_m = (ref_y - REF_NET_Y) / REF_HEIGHT * COURT_LENGTH
    return x_m, z_m


def detect_court(first_frame):
    """Re-use the existing court detector for the homography (read-only import)."""
    orig = os.getcwd()
    os.chdir('/media/skr/storage/ten_bad/tennis-tracking')
    from court_detector import CourtDetector
    cd = CourtDetector()
    cd.detect(first_frame)
    os.chdir(orig)
    cwm = cd.court_warp_matrix[0]
    cam_to_ref_h = cv2.invert(cwm)[1]
    return cam_to_ref_h


def smplx_world_joints(track, smplx_layer, device='cuda'):
    """Run the SMPL-X layer on a track's world-frame parameters → (N, 127, 3).

    PromptHMR stores `smplx_world['pose']` as 165-d axis-angle (55 joints × 3).
    The SMPLX wrapper expects rotation matrices for global_orient/body_pose
    (mirrors what demo_video.py does internally before forward).
    """
    from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
    pose = torch.from_numpy(track['smplx_world']['pose']).to(device).float()
    trans = torch.from_numpy(track['smplx_world']['trans']).to(device).float()
    shape = torch.from_numpy(track['smplx_world']['shape']).to(device).float()
    rotmat = axis_angle_to_matrix(pose.reshape(-1, 55, 3))   # (N, 55, 3, 3)
    out = smplx_layer(
        global_orient=rotmat[:, :1],
        body_pose=rotmat[:, 1:22],
        betas=shape,
        transl=trans,
    )
    return out.joints.detach().cpu().numpy()  # (N, 127, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=DEFAULT_VIDEO,
                    help='Source video used by demo_video.py')
    ap.add_argument('--results', default=DEFAULT_RESULTS_PKL,
                    help='PromptHMR results.pkl')
    ap.add_argument('--ball-csv', default=DEFAULT_BALL_CSV,
                    help='WASB ball positions CSV (optional, projected if present)')
    ap.add_argument('--out', default=DEFAULT_OUTPUT_JSON,
                    help='Output JSON path')
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f'results.pkl not found at {args.results} — run demo_video.py first.')

    print(f'Loading {args.results}...')
    results = joblib.load(args.results)
    print('Top-level keys:', sorted(results.keys()))
    print('# tracks:', len(results.get('people', {})))

    # Frame metadata from the source video.
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f'Failed to read first frame from {args.video}')
    print(f'Video: {n_frames} frames @ {fps} fps')

    # Court homography for projecting feet → court meters.
    print('Detecting court...')
    cam_to_ref_h = detect_court(first_frame)

    # SMPL-X layer (use PromptHMR's wrapper to match how the model was trained).
    # The wrapper expects data/body_models/* relative to cwd, so chdir for init.
    print('Loading SMPL-X layer...')
    orig_cwd = os.getcwd()
    os.chdir(PROMPTHMR_DIR)
    from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
    smplx_layer = SMPLX_Layer(SMPLX_PATH).cuda()
    os.chdir(orig_cwd)

    # Per-track: SMPL-X joints in PromptHMR's world frame, then anchor to court.
    print('Computing per-track world joints...')
    tracks_data = {}
    for tid, track in results['people'].items():
        joints = smplx_world_joints(track, smplx_layer)
        coco_joints = joints[:, SMPLX_TO_COCO]   # (N, 17, 3)
        # PromptHMR world: Y is up (gravity-aligned). Confirm by checking sign convention.
        frames = np.asarray(track['frames'])
        tracks_data[tid] = {
            'frames': frames,                    # (N,) frame indices
            'coco_world': coco_joints,           # (N, 17, 3) PromptHMR world
        }
        print(f'  track {tid}: {len(frames)} frames, '
              f'world Y range [{coco_joints[:,:,1].min():.2f}, {coco_joints[:,:,1].max():.2f}]')

    # Decide which track is near, which is far, by mean court-Z (after homography)
    # of foot positions across the track lifetime.
    # We use the camera-frame foot pixel from PromptHMR's per-frame bbox.
    print('Classifying tracks as near/far via court-Z...')
    track_court_z = {}
    for tid, track in results['people'].items():
        # Use camera-frame ankles (smplx_cam) projected via bbox bottom.
        bboxes = track.get('bboxes')   # (N, 4) xyxy in camera pixels
        if bboxes is None:
            print(f'  track {tid}: no bboxes — skipping classification')
            continue
        bboxes = np.asarray(bboxes)
        foot_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        foot_y = bboxes[:, 3]
        zs = []
        for fx, fy in zip(foot_x, foot_y):
            p = np.array([[[float(fx), float(fy)]]], dtype=np.float32)
            tp = cv2.perspectiveTransform(p, cam_to_ref_h)
            _, z_m = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))
            zs.append(z_m)
        track_court_z[tid] = float(np.mean(zs))
        print(f'  track {tid}: mean court Z = {track_court_z[tid]:+.2f} m')

    # Player selection: tracks beyond the court (|Z| > court_length/2 + 2 m
    # margin) are spectators/umpire — exclude them. From the rest, pick
    # the longest track on each side of the net (Z>0 = near, Z<0 = far).
    half_l = COURT_LENGTH / 2          # 11.88 m
    margin = 3.0                       # accept up to 3 m past baseline
    candidates = {tid: z for tid, z in track_court_z.items()
                  if abs(z) <= half_l + margin}
    near_cands = [(tid, z) for tid, z in candidates.items() if z > 0]
    far_cands  = [(tid, z) for tid, z in candidates.items() if z < 0]
    if not near_cands or not far_cands:
        sys.exit(f'Could not find players on both sides. candidates={candidates}')
    # Within each half, pick track with most frames (longest = real player).
    def _pick(cands):
        return max(cands, key=lambda kv: len(results['people'][kv[0]]['frames']))[0]
    near_tid = _pick(near_cands)
    far_tid = _pick(far_cands)
    print(f'  → near track: {near_tid} (z={track_court_z[near_tid]:+.2f}, '
          f'{len(results["people"][near_tid]["frames"])} frames)')
    print(f'  → far track:  {far_tid} (z={track_court_z[far_tid]:+.2f}, '
          f'{len(results["people"][far_tid]["frames"])} frames)')

    # For each frame, anchor each player's PromptHMR-world skeleton onto the
    # court using YOLO bbox + homography. We keep the *relative* skeleton from
    # PromptHMR (its joint configuration) and only transplant the foot XZ
    # position from the homography projection.
    print('Anchoring skeletons to court coords...')
    near_skel = [None] * n_frames
    near_conf = [None] * n_frames
    far_skel = [None] * n_frames
    far_conf = [None] * n_frames

    for role, tid, skel_store, conf_store in [
        ('near', near_tid, near_skel, near_conf),
        ('far', far_tid, far_skel, far_conf),
    ]:
        td = tracks_data[tid]
        bboxes = np.asarray(results['people'][tid]['bboxes'])
        for k, fi in enumerate(td['frames']):
            if fi < 0 or fi >= n_frames:
                continue
            joints = td['coco_world'][k].copy()  # (17,3) in PromptHMR world

            # Anchor: lowest body Y → 0
            min_y = float(joints[:, 1].min())
            joints[:, 1] -= min_y

            # Foot court position from this frame's bbox
            fx = float((bboxes[k, 0] + bboxes[k, 2]) / 2)
            fy = float(bboxes[k, 3])
            p = np.array([[[fx, fy]]], dtype=np.float32)
            tp = cv2.perspectiveTransform(p, cam_to_ref_h)
            foot_x_m, foot_z_m = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))

            # Pelvis is approximately the mean of left/right hip in COCO indices 11,12
            pelvis_xz = (joints[11, [0, 2]] + joints[12, [0, 2]]) / 2
            joints[:, 0] += foot_x_m - pelvis_xz[0]
            joints[:, 2] += foot_z_m - pelvis_xz[1]

            skel_store[int(fi)] = joints
            conf_store[int(fi)] = np.ones(17, dtype=np.float32) * 0.95

    # Build per-frame entries.
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
            players.append({
                'id': pid,
                'role': role,
                'keypoints_3d': kps_list,
                'confidence': [round(float(c), 3) for c in conf],
            })
        frames_data.append({'frame': i, 'players': players, 'ball': None})

    # Ball — same projection as the existing exporter.
    print('Projecting ball...')
    if os.path.exists(args.ball_csv):
        ball_cam = [None] * n_frames
        with open(args.ball_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fi = int(row['frame'])
                if fi < n_frames and int(row['visible']):
                    ball_cam[fi] = (float(row['x']), float(row['y']))
        for i in range(n_frames):
            if ball_cam[i] is not None:
                p = np.array([[[ball_cam[i][0], ball_cam[i][1]]]], dtype=np.float32)
                tp = cv2.perspectiveTransform(p, cam_to_ref_h)
                bx, bz = ref_to_meters(float(tp[0, 0, 0]), float(tp[0, 0, 1]))
                frames_data[i]['ball'] = [round(bx, 4), 0.5, round(bz, 4)]

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
        'source': 'prompthmr',
        'frames': frames_data,
    }

    with open(args.out, 'w') as f:
        json.dump(output, f)
    sz = os.path.getsize(args.out) / 1024 / 1024
    print(f'\nWrote {args.out} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
