"""
Render a side-by-side comparison MP4 for the MediaPipe Pose Landmarker, in
the same style as `videopose3d_output/output_<role>.mp4`:

    ┌─────────────────────┐  ┌─────────────────────┐
    │       Input         │  │   Reconstruction    │
    │  (frame + skeleton) │  │   (matplotlib 3D)   │
    └─────────────────────┘  └─────────────────────┘

Standalone — does not modify any existing file. Reuses `PoseDetector` /
`PlayerTracker` (read-only import) so the player crops match the rest of the
project, then runs MediaPipe on each crop.

Usage:
    python mediapipe_skeleton_video.py near
    python mediapipe_skeleton_video.py far
    python mediapipe_skeleton_video.py both    # writes both
"""
import sys
import os
import io
import subprocess
import cv2
import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SMOOTH_WINDOW = 11
SMOOTH_POLY = 2

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker  # noqa: E402

VIDEO = 'S_Original_HL_clip_cropped.mp4'
MP_MODEL = 'models/pose_landmarker_heavy.task'
OUTPUT_DIR = 'mediapipe_output'

# VIDEO mode + wide pad (1.0) + no pre-upscale gives best coverage.
# See sweep — far coverage 40% → 97% with this combo.
NEAR_PAD = 1.0
FAR_PAD = 1.0
FAR_UPSCALE = 1

# MediaPipe POSE_CONNECTIONS as (a, b) index pairs. Mirror set provided by
# mp.solutions.pose for the 33-landmark model.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Right-side joints: render in red. Everything else: black.
RIGHT_INDICES = {2, 5, 7, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def crop_with_padding(frame, bbox, pad_frac):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1p = int(max(0, x1 - bw * pad_frac))
    y1p = int(max(0, y1 - bh * pad_frac))
    x2p = int(min(w, x2 + bw * pad_frac))
    y2p = int(min(h, y2 + bh * pad_frac))
    if x2p <= x1p or y2p <= y1p:
        return None, None
    return frame[y1p:y2p, x1p:x2p], (x1p, y1p, x2p, y2p)


def run_mediapipe(landmarker, crop_bgr, ts_ms, upscale=1):
    """VIDEO-mode call. ts_ms must be monotonic per landmarker.
    Returns (image_landmarks_normalised(33,3), world_landmarks(33,3), visibility(33,))."""
    if upscale != 1:
        ch, cw = crop_bgr.shape[:2]
        crop_bgr = cv2.resize(crop_bgr, (cw * upscale, ch * upscale),
                              interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    if not result.pose_world_landmarks:
        return None, None, None
    world = result.pose_world_landmarks[0]
    img = result.pose_landmarks[0]
    world_arr = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)
    img_arr = np.array([[lm.x, lm.y, lm.z] for lm in img], dtype=np.float32)
    vis = np.array([lm.visibility for lm in world], dtype=np.float32)
    return img_arr, world_arr, vis


def draw_2d_skeleton(frame, img_landmarks, crop_box, vis):
    """Draw the 2D skeleton overlay, mapping landmark normalised coords back
    to the original frame via the crop box."""
    x1p, y1p, x2p, y2p = crop_box
    cw_, ch_ = x2p - x1p, y2p - y1p
    pts = []
    for i, lm in enumerate(img_landmarks):
        px = int(x1p + lm[0] * cw_)
        py = int(y1p + lm[1] * ch_)
        pts.append((px, py))
    for a, b in POSE_CONNECTIONS:
        if vis[a] < 0.3 or vis[b] < 0.3:
            continue
        color = (0, 0, 255) if (a in RIGHT_INDICES or b in RIGHT_INDICES) else (255, 255, 255)
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if vis[i] < 0.3:
            continue
        color = (0, 0, 255) if i in RIGHT_INDICES else (255, 255, 255)
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
    return frame


def render_3d_panel(world_lms, vis, panel_size_px, axis_lim=0.9):
    """Render the matplotlib 3D reconstruction at a fixed pixel size, return BGR."""
    w_in = panel_size_px[0] / 100.0
    h_in = panel_size_px[1] / 100.0
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')

    if world_lms is not None:
        # MediaPipe world: x right, y down, z forward → flip y so up is +.
        xs = world_lms[:, 0]
        ys = -world_lms[:, 1]
        zs = world_lms[:, 2]

        for a, b in POSE_CONNECTIONS:
            if vis is not None and (vis[a] < 0.3 or vis[b] < 0.3):
                continue
            color = 'red' if (a in RIGHT_INDICES or b in RIGHT_INDICES) else 'black'
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]],
                    color=color, linewidth=2)
    else:
        ax.text(0, 0, 0, 'no detection', color='gray', ha='center')

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=-70)
    ax.set_title('Reconstruction', fontsize=14, pad=8)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())  # (h, w, 4)
    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr


def add_input_label(frame):
    """Title strip above the input panel."""
    h, w = frame.shape[:2]
    bar = np.full((40, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, 'Input', (w // 2 - 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def directional_pass(frames, bboxes, fps, pad, upscale, make_landmarker, reverse=False):
    """Run a single MP pass over the frame sequence (forward or backward).

    Returns three lists of length n: img_lms[i], world_lms[i], vis[i]
    (each None when no detection / no bbox). Caller side handles merging.
    A fresh landmarker is created so timestamps stay monotonic per stream.
    """
    n = len(frames)
    img_out = [None] * n
    world_out = [None] * n
    vis_out = [None] * n
    crop_box_out = [None] * n  # needed later for drawing 2D landmarks back on frame

    lm = make_landmarker()
    indices = range(n - 1, -1, -1) if reverse else range(n)
    for k, i in enumerate(indices):
        ts_ms = int(k * 1000 / fps)  # monotonic for this stream
        bbox = bboxes[i]
        if bbox is None:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            run_mediapipe(lm, blank, ts_ms, upscale=1)
            continue
        crop, crop_box = crop_with_padding(frames[i], bbox, pad)
        if crop is None or crop.size == 0:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            run_mediapipe(lm, blank, ts_ms, upscale=1)
            continue
        img_arr, world_arr, v = run_mediapipe(lm, crop, ts_ms, upscale=upscale)
        if img_arr is not None:
            img_out[i] = img_arr
            world_out[i] = world_arr
            vis_out[i] = v
            crop_box_out[i] = crop_box
        if k % 100 == 0:
            tag = 'bwd' if reverse else 'fwd'
            print(f'  {tag} {k}/{n}')
    lm.close()
    return img_out, world_out, vis_out, crop_box_out


def merge_directional(img_f, world_f, vis_f, cb_f, img_b, world_b, vis_b, cb_b):
    """Per-frame: pick the direction with higher mean visibility."""
    n = len(img_f)
    img = [None] * n; world = [None] * n; vis = [None] * n; cb = [None] * n
    recovered = 0; overrode = 0
    for i in range(n):
        f_ok = vis_f[i] is not None
        b_ok = vis_b[i] is not None
        if f_ok and b_ok:
            if float(np.mean(vis_b[i])) > float(np.mean(vis_f[i])):
                img[i], world[i], vis[i], cb[i] = img_b[i], world_b[i], vis_b[i], cb_b[i]
                overrode += 1
            else:
                img[i], world[i], vis[i], cb[i] = img_f[i], world_f[i], vis_f[i], cb_f[i]
        elif f_ok:
            img[i], world[i], vis[i], cb[i] = img_f[i], world_f[i], vis_f[i], cb_f[i]
        elif b_ok:
            img[i], world[i], vis[i], cb[i] = img_b[i], world_b[i], vis_b[i], cb_b[i]
            recovered += 1
    return img, world, vis, cb, recovered, overrode


def smooth_landmark_seq(seq, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Savitzky-Golay smoothing per joint XYZ across consecutive runs of detections."""
    n = len(seq)
    if n == 0:
        return seq
    i = 0
    while i < n:
        if seq[i] is not None:
            j = i
            while j < n and seq[j] is not None:
                j += 1
            L = j - i
            if L >= window:
                stack = np.stack(seq[i:j], axis=0)  # (L, 33, 3)
                smoothed = savgol_filter(stack, window, poly, axis=0)
                for k in range(L):
                    seq[i + k] = smoothed[k]
            i = j
        else:
            i += 1
    return seq


def render_role(role, frames, bboxes, fps, out_path, make_landmarker):
    """Render side-by-side video for one role (near/far) with bidirectional MP + smoothing."""
    n = len(frames)
    h_in, w_in = frames[0].shape[:2]
    pad = NEAR_PAD if role == 'near' else FAR_PAD
    upscale = 1 if role == 'near' else FAR_UPSCALE

    # Forward + backward MP, then merge per-frame by mean visibility.
    print(f'  [{role}] forward pass...')
    img_f, world_f, vis_f, cb_f = directional_pass(
        frames, bboxes, fps, pad, upscale, make_landmarker, reverse=False)
    print(f'  [{role}] backward pass...')
    img_b, world_b, vis_b, cb_b = directional_pass(
        frames, bboxes, fps, pad, upscale, make_landmarker, reverse=True)
    img_lms, world_lms, vis_lms, crop_boxes, rec, ovr = merge_directional(
        img_f, world_f, vis_f, cb_f, img_b, world_b, vis_b, cb_b)
    final = sum(1 for v in vis_lms if v is not None)
    print(f'  [{role}] merged: {final}/{n}  (recovered {rec}, overrode {ovr})')

    # Temporal smoothing on both image and world coords.
    smooth_landmark_seq(img_lms)
    smooth_landmark_seq(world_lms)

    panel_w = w_in
    panel_h = h_in + 40
    out_w = w_in * 2
    out_h = h_in + 40

    ff_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    for i in range(n):
        frame = frames[i].copy()
        if img_lms[i] is not None and crop_boxes[i] is not None:
            frame = draw_2d_skeleton(frame, img_lms[i], crop_boxes[i], vis_lms[i])

        left_panel = add_input_label(frame)
        right_panel = render_3d_panel(world_lms[i], vis_lms[i], (panel_w, panel_h))
        if right_panel.shape[:2] != (panel_h, panel_w):
            right_panel = cv2.resize(right_panel, (panel_w, panel_h))

        combined = np.hstack([left_panel, right_panel])
        if combined.shape[:2] != (out_h, out_w):
            combined = cv2.resize(combined, (out_w, out_h))
        proc.stdin.write(combined.tobytes())

        if i % 100 == 0:
            print(f'  [{role}] render {i}/{n}')

    proc.stdin.close()
    proc.wait()
    print(f'  [{role}] wrote {out_path}')


def main():
    role_arg = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if role_arg not in ('near', 'far', 'both'):
        print('Usage: python mediapipe_skeleton_video.py [near|far|both]')
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    print(f'Read {n} frames at {fps} fps')

    # Track players to get bboxes per frame (same as the JSON pipeline).
    print('Tracking players (YOLO)...')
    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    near_bboxes = [None] * n
    far_bboxes = [None] * n
    for i, frame in enumerate(frames):
        players = tracker.update(frame, conf=0.1)
        if players.get('near') is not None:
            near_bboxes[i] = players['near']['bbox'].tolist()
        if players.get('far') is not None:
            far_bboxes[i] = players['far']['bbox'].tolist()
        if i % 100 == 0:
            print(f'  yolo {i}/{n}')

    # VIDEO mode needs monotonic timestamps and a fresh landmarker per
    # independent stream — one per role.
    print('Initialising MediaPipe Pose Landmarker (VIDEO mode, per-role)...')
    def make_landmarker():
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    roles = ['near', 'far'] if role_arg == 'both' else [role_arg]
    for role in roles:
        bboxes = near_bboxes if role == 'near' else far_bboxes
        out_path = os.path.join(OUTPUT_DIR, f'output_{role}.mp4')
        print(f'Rendering {role} → {out_path}')
        render_role(role, frames, bboxes, fps, out_path, make_landmarker)

    print('Done.')


if __name__ == '__main__':
    main()
