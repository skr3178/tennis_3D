"""
Compare three near-player 2D detection pipelines feeding VideoPose3D:
  1. YOLO (full-frame)
  2. Detectron2 (full-frame, no crop)
  3. Detectron2 (per-frame crop around player) — current best

For each pipeline:
  - Write Detectron-format 2D npz
  - Build VP3D custom dataset
  - Run VP3D inference → .npy
  - Post-scale to 1.7m body span
  - Collect metrics: detection rate, wrist speed, body span stability

All three outputs go to videopose3d_output/ with distinct suffixes:
  output_near_yolo.npy
  output_near_det2_full.npy
  output_near_det2_crop.npy
"""
import os, sys, subprocess, warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np

ORIG_DIR = os.getcwd()
sys.path.insert(0, 'tennis-tracking')

VIDEO = 'S_Original_HL_clip_cropped.mp4'
V3D_DIR = 'VideoPose3D'
CKPT = 'pretrained_h36m_detectron_coco.bin'
OUT_DIR = 'videopose3d_output'
TARGET_SPAN = 1.70  # real body span


def save_detectron_npz(kps_per_frame, bb_per_frame, w, h, out_path):
    n = len(kps_per_frame)
    boxes, keypoints = [], []
    for i in range(n):
        kps = kps_per_frame[i]
        bb = bb_per_frame[i]
        if kps is None or bb is None:
            boxes.append([[], np.zeros((0, 5), dtype=np.float32)])
            keypoints.append([[], np.zeros((0, 4, 17), dtype=np.float32)])
            continue
        bbox_arr = np.array([[bb[0], bb[1], bb[2], bb[3], 1.0]], dtype=np.float32)
        kps_arr = np.zeros((1, 4, 17), dtype=np.float32)
        kps_arr[0, 0, :] = kps[:, 0]
        kps_arr[0, 1, :] = kps[:, 1]
        kps_arr[0, 2, :] = 0.0
        kps_arr[0, 3, :] = kps[:, 2]
        boxes.append([[], bbox_arr])
        keypoints.append([[], kps_arr])
    metadata = {'w': w, 'h': h}
    boxes_arr = np.array(boxes, dtype=object)
    kps_arr = np.array(keypoints, dtype=object)
    segs_arr = np.array([None] * n, dtype=object)
    np.savez_compressed(out_path, boxes=boxes_arr, segments=segs_arr,
                        keypoints=kps_arr, metadata=metadata)


def run_vp3d_on_npz(variant_name, input_w, input_h):
    """Build VP3D custom dataset & run inference. Writes <variant_name>.npy."""
    det_dir = os.path.join(ORIG_DIR, OUT_DIR, f'detect_{variant_name}')
    data_dir = os.path.join(ORIG_DIR, V3D_DIR, 'data')
    print(f'  Building VP3D custom dataset...')
    subprocess.run(['python', 'prepare_data_2d_custom.py',
                    '-i', det_dir, '-o', f'compare_{variant_name}'],
                   cwd=data_dir, check=True, capture_output=True)

    vp3d_root = os.path.join(ORIG_DIR, V3D_DIR)
    if not os.path.exists(os.path.join(vp3d_root, VIDEO)):
        subprocess.run(['cp', VIDEO, vp3d_root], check=True)

    out_npy = os.path.join(ORIG_DIR, OUT_DIR, f'output_near_{variant_name}.npy')
    out_mp4 = os.path.join(ORIG_DIR, OUT_DIR, f'_tmp_{variant_name}.mp4')
    print(f'  Running VP3D inference...')
    subprocess.run(['python', 'run.py',
                    '-d', 'custom', '-k', f'compare_{variant_name}',
                    '-arc', '3,3,3,3,3', '-c', 'checkpoint',
                    '--evaluate', CKPT, '--render',
                    '--viz-subject', VIDEO, '--viz-action', 'custom',
                    '--viz-camera', '0', '--viz-video', VIDEO,
                    '--viz-output', out_mp4, '--viz-export', out_npy,
                    '--viz-size', '6', '--viz-downsample', '1'],
                   cwd=vp3d_root, check=True, capture_output=True)
    # Cleanup temp render
    if os.path.exists(out_mp4):
        os.remove(out_mp4)
    return out_npy


# ─── Variant 1: YOLO full-frame ───

def run_yolo_fullframe():
    from pose_detector import PoseDetector, PlayerTracker
    print('\n=== Variant 1: YOLO full-frame ===')
    cap = cv2.VideoCapture(VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    near_kps, near_bb = [None] * n, [None] * n
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        players = tracker.update(frame, conf=0.1)
        d = players.get('near')
        if d is not None:
            near_kps[i] = d['keypoints']
            near_bb[i] = d['bbox']
        if i % 100 == 0:
            n_ok = sum(1 for k in near_kps[:i+1] if k is not None)
            print(f'  Frame {i}/{n}  detected={n_ok}')
        i += 1
    cap.release()

    det_dir = os.path.join(OUT_DIR, 'detect_yolo')
    os.makedirs(det_dir, exist_ok=True)
    save_detectron_npz(near_kps, near_bb, w, h,
                       os.path.join(det_dir, f'{VIDEO}.npz'))
    return run_vp3d_on_npz('yolo', w, h), sum(1 for k in near_kps if k is not None), n


# ─── Variant 2: Detectron2 full-frame ───

def run_det2_fullframe():
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    print('\n=== Variant 2: Detectron2 full-frame ===')
    cfg = get_cfg()
    cfg_file = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    near_kps, near_bb = [None] * n, [None] * n
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = predictor(frame)['instances'].to('cpu')
        if out.has('pred_boxes') and len(out.pred_boxes) > 0:
            boxes = out.pred_boxes.tensor.numpy()
            bb_h = boxes[:, 3] - boxes[:, 1]
            bb_cy = (boxes[:, 1] + boxes[:, 3]) / 2
            # near: tallest in lower 55% of frame
            mask = bb_cy > h * 0.45
            if mask.any():
                idx = np.where(mask)[0][np.argmax(bb_h[mask])]
                near_bb[i] = boxes[idx, :4]
                near_kps[i] = out.pred_keypoints.numpy()[idx]
        if i % 100 == 0:
            n_ok = sum(1 for k in near_kps[:i+1] if k is not None)
            print(f'  Frame {i}/{n}  detected={n_ok}')
        i += 1
    cap.release()

    det_dir = os.path.join(OUT_DIR, 'detect_det2_full')
    os.makedirs(det_dir, exist_ok=True)
    save_detectron_npz(near_kps, near_bb, w, h,
                       os.path.join(det_dir, f'{VIDEO}.npz'))
    return run_vp3d_on_npz('det2_full', w, h), sum(1 for k in near_kps if k is not None), n


def compute_metrics(npy_path, label):
    """Post-scale and compute stats."""
    arr = np.load(npy_path)
    # Body span in VP3D coords (Y-down)
    head_y = -arr[:, 10, 1]
    feet_y = -(arr[:, 3, 1] + arr[:, 6, 1]) / 2
    span_per_frame = head_y - feet_y
    median_span = np.median(span_per_frame)
    scale = TARGET_SPAN / median_span
    # Apply uniform scale around hip root
    root = arr[:, 0:1, :]
    scaled = root + (arr - root) * scale
    np.save(npy_path, scaled)  # overwrite with scaled

    # Recompute after scaling
    arr2 = scaled
    head_y2 = -arr2[:, 10, 1]
    feet_y2 = -(arr2[:, 3, 1] + arr2[:, 6, 1]) / 2
    span2 = head_y2 - feet_y2
    # Wrist speed
    wrist = arr2[:, 13, :]
    deltas = np.linalg.norm(np.diff(wrist, axis=0), axis=1) * 50  # m/s
    # Jitter = p99 / median (high = jittery spikes)
    jitter = deltas.max() / max(np.median(deltas), 0.01)

    return {
        'label': label,
        'body_span_mean': span2.mean(),
        'body_span_std': span2.std(),
        'pre_scale_span': median_span,
        'scale_factor': scale,
        'wrist_mean': deltas.mean(),
        'wrist_median': np.median(deltas),
        'wrist_p95': np.percentile(deltas, 95),
        'wrist_max': deltas.max(),
        'wrist_jitter': jitter,
    }


def main():
    results = []

    # 1. YOLO
    npy_path, detected, total = run_yolo_fullframe()
    m = compute_metrics(npy_path, 'YOLO full-frame')
    m['detection_rate'] = f'{detected}/{total} ({100*detected/total:.1f}%)'
    results.append(m)

    # 2. Detectron2 full-frame
    npy_path, detected, total = run_det2_fullframe()
    m = compute_metrics(npy_path, 'Detectron2 full-frame')
    m['detection_rate'] = f'{detected}/{total} ({100*detected/total:.1f}%)'
    results.append(m)

    # 3. Detectron2 cropped (already exists from previous run)
    npy_path = os.path.join(OUT_DIR, 'output_near_det2_crop.npy')
    m = compute_metrics(npy_path, 'Detectron2 + crop')
    # Detection rate from saved npz
    d = np.load(os.path.join(OUT_DIR, 'detect_near', f'{VIDEO}.npz'),
                allow_pickle=True)
    boxes = d['boxes']
    detected = sum(1 for i in range(len(boxes)) if len(boxes[i][1]) > 0)
    total = len(boxes)
    m['detection_rate'] = f'{detected}/{total} ({100*detected/total:.1f}%)'
    results.append(m)

    # ─── Print comparison table ───
    print('\n' + '=' * 100)
    print(f'{"Metric":<35}{"YOLO":<22}{"Detectron2 full":<22}{"Detectron2 + crop":<22}')
    print('=' * 100)
    rows = [
        ('Detection rate', 'detection_rate'),
        ('Pre-scale body span (m)', 'pre_scale_span'),
        ('Post-scale factor applied', 'scale_factor'),
        ('Body span after scale (mean)', 'body_span_mean'),
        ('Body span after scale (std)', 'body_span_std'),
        ('Wrist speed median (m/s)', 'wrist_median'),
        ('Wrist speed mean (m/s)', 'wrist_mean'),
        ('Wrist speed p95 (m/s)', 'wrist_p95'),
        ('Wrist speed max (m/s)', 'wrist_max'),
        ('Jitter ratio (max/median)', 'wrist_jitter'),
    ]
    for label, key in rows:
        vals = [r[key] for r in results]
        formatted = []
        for v in vals:
            if isinstance(v, str):
                formatted.append(f'{v:<22}')
            else:
                formatted.append(f'{v:<22.3f}')
        print(f'{label:<35}' + ''.join(formatted))
    print('=' * 100)
    print()
    print('Interpretation:')
    print('  - Detection rate: % of frames where a near-player was found')
    print('  - Pre-scale body span: model output scale before post-processing (ideal ~1.7m)')
    print('  - Scale factor: how much we had to stretch (closer to 1.0 = better match to reality)')
    print('  - Wrist speed: 2-5 m/s typical tennis rally, 15-25 m/s peak swings')
    print('  - Jitter ratio: lower = smoother. Spikes >> median indicates keypoint jitter.')


if __name__ == '__main__':
    main()
