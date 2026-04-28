"""
Run VideoPose3D on the tennis video for both near and far players using a
two-pass crop-and-detect pipeline.

Pipeline:
  Pass 1 (full-frame tracking):
    Run Detectron2 MaskRCNN on each full frame, classify detections
    into near/far player, get per-frame bbox for each.

  Pass 2 (per-player crop detection):
    For each player per frame: make a square padded crop around the bbox,
    resize to a fixed size (CROP_SIZE), re-run Detectron2 on that crop
    so the player fills the frame (like the ski dancer demo).
    Keep only the highest-confidence detection in the crop (which should
    be our target player since they dominate the crop).

  Save each player's .npz with CROP_SIZE as the metadata frame dimensions,
  so VP3D's normalization treats the crop as if it were the full video.

  Then run VP3D inference per player → 3D .npy + rendered .mp4.

Why this matches VP3D's training distribution:
  VP3D was trained on broadcast clips where the subject fills most of the
  frame (H36M: studio shots of a single person). With our tennis broadcast
  where players are small, Detectron2 on the full frame + keypoint rescaling
  was an approximation. Crop-then-detect lets Detectron2 itself produce
  keypoints in crop-space where the player truly fills the frame.
"""
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np

ORIG_DIR = os.getcwd()

VIDEO = 'S_Original_HL_clip_cropped.mp4'
V3D_DIR = 'VideoPose3D'
CKPT = 'pretrained_h36m_detectron_coco.bin'
OUT_DIR = 'videopose3d_output'
os.makedirs(OUT_DIR, exist_ok=True)

# Crop size fed to VP3D (square so aspect matches H36M well enough)
CROP_SIZE = 512
# Padding around each player's bbox (factor: 1.0 = tight, 1.3 = 30% extra margin)
CROP_PAD = 1.30


# ─── Detectron2 ───

def load_detectron_predictor():
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg_file = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    cfg.MODEL.DEVICE = 'cuda'
    return DefaultPredictor(cfg)


def classify_detections(boxes, keypoints, frame_h):
    """Pick near + far player from all Detectron2 detections in one frame.

    Rules:
      near: tallest bbox with y_center in lower 55% of frame
      far: tallest bbox ≥ 40 px in y_center band 10-35% of frame height
    Returns (near_bb, near_kps, far_bb, far_kps), any can be None.
    """
    if len(boxes) == 0:
        return None, None, None, None

    bb_h = boxes[:, 3] - boxes[:, 1]
    bb_cy = (boxes[:, 1] + boxes[:, 3]) / 2

    near_idx, far_idx = None, None
    near_mask = bb_cy > frame_h * 0.45
    if near_mask.any():
        near_idx = np.where(near_mask)[0][np.argmax(bb_h[near_mask])]

    far_mask = (bb_cy >= frame_h * 0.08) & (bb_cy <= frame_h * 0.38) & (bb_h >= 40)
    if far_mask.any():
        far_idx = np.where(far_mask)[0][np.argmax(bb_h[far_mask])]

    nbb = boxes[near_idx, :4] if near_idx is not None else None
    nkp = keypoints[near_idx] if near_idx is not None else None
    fbb = boxes[far_idx, :4] if far_idx is not None else None
    fkp = keypoints[far_idx] if far_idx is not None else None
    return nbb, nkp, fbb, fkp


def make_square_crop(bbox, frame_w, frame_h, pad=CROP_PAD):
    """Return (x0, y0, size) of a square padded crop around bbox.

    Extends the smaller bbox dimension to match the larger, then multiplies
    by pad, and clamps to frame bounds.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    size = max(bbox_w, bbox_h) * pad
    # Can't exceed frame
    size = min(size, frame_w, frame_h)
    x0 = cx - size / 2
    y0 = cy - size / 2
    # Clamp to frame
    x0 = max(0, min(frame_w - size, x0))
    y0 = max(0, min(frame_h - size, y0))
    return x0, y0, size


def detect_in_crop(predictor, crop):
    """Run Detectron2 on a single crop, return (bbox, keypoints) of top person
    or (None, None) if no detection.
    """
    out = predictor(crop)['instances'].to('cpu')
    if not out.has('pred_boxes') or len(out.pred_boxes) == 0:
        return None, None
    boxes = out.pred_boxes.tensor.numpy()
    scores = out.scores.numpy()
    kps = out.pred_keypoints.numpy()  # (N, 17, 3)
    # Pick detection closest to crop center (most likely our target player)
    h, w = crop.shape[:2]
    cx_crop, cy_crop = w / 2, h / 2
    bb_cx = (boxes[:, 0] + boxes[:, 2]) / 2
    bb_cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bb_h = boxes[:, 3] - boxes[:, 1]
    # Score by: confidence × (1 / (1 + distance_to_center/w)) × (bbox_height / w)
    dist = np.sqrt((bb_cx - cx_crop)**2 + (bb_cy - cy_crop)**2) / w
    weight = scores * (1 / (1 + dist * 2)) * (bb_h / w)
    best = np.argmax(weight)
    return boxes[best], kps[best]


# ─── Pipeline ───

def pass1_full_frame_detect(predictor, video_path):
    """Pass 1: Detect near/far players on full frames. Return per-frame bboxes."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: {w}x{h} {fps}fps {total} frames')

    near_bb = [None] * total
    far_bb = [None] * total
    frames = []

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        out = predictor(frame)['instances'].to('cpu')
        if out.has('pred_boxes') and len(out.pred_boxes) > 0:
            boxes = out.pred_boxes.tensor.numpy()
            scores = out.scores.numpy()[:, None]
            boxes = np.concatenate([boxes, scores], axis=1)
            kps = out.pred_keypoints.numpy()
            nbb, _, fbb, _ = classify_detections(boxes, kps, h)
            near_bb[i] = nbb
            far_bb[i] = fbb
        if i % 50 == 0:
            n_near = sum(1 for b in near_bb[:i+1] if b is not None)
            n_far = sum(1 for b in far_bb[:i+1] if b is not None)
            print(f'  Pass 1: Frame {i}/{total}  near={n_near} far={n_far}')
        i += 1
    cap.release()

    # Smooth bboxes over time (simple temporal linear-interp fill for short gaps)
    near_bb = interp_bboxes(near_bb)
    far_bb = interp_bboxes(far_bb)

    return frames, near_bb, far_bb, w, h, fps


def interp_bboxes(bboxes):
    """Fill short None gaps by linear interpolation of adjacent bboxes."""
    n = len(bboxes)
    arr = np.full((n, 4), np.nan)
    for i, bb in enumerate(bboxes):
        if bb is not None:
            arr[i] = bb
    indices = np.arange(n)
    for j in range(4):
        valid = ~np.isnan(arr[:, j])
        if valid.sum() < 2:
            continue
        arr[:, j] = np.interp(indices, indices[valid], arr[valid, j])
    result = []
    # Only accept interpolated bboxes within reasonable range of nearest real ones
    has_any = np.any([b is not None for b in bboxes])
    if not has_any:
        return bboxes
    for i in range(n):
        if not np.any(np.isnan(arr[i])):
            result.append(arr[i].astype(np.float32))
        else:
            result.append(None)
    return result


def pass2_crop_detect(predictor, frames, bboxes, role, w, h):
    """Pass 2: For each frame, crop around player bbox + re-detect.
    Returns list of (keypoints_in_CROP_SIZE_coords or None) per frame.
    """
    n = len(frames)
    out_kps = [None] * n
    out_bb = [None] * n
    n_detected = 0
    for i in range(n):
        bb = bboxes[i]
        if bb is None:
            continue
        x0, y0, size = make_square_crop(bb, w, h)
        x1, y1 = x0 + size, y0 + size
        crop = frames[i][int(y0):int(y1), int(x0):int(x1)]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
        bbox_crop, kps_crop = detect_in_crop(predictor, crop_resized)
        if kps_crop is None:
            continue
        out_kps[i] = kps_crop
        out_bb[i] = bbox_crop
        n_detected += 1
        if i % 50 == 0:
            print(f'  Pass 2 ({role}): Frame {i}/{n}  detected={n_detected}')
    print(f'  Pass 2 ({role}): {n_detected}/{n} frames detected')
    return out_kps, out_bb


def save_detectron_npz(keypoints_per_frame, bboxes_per_frame, out_path):
    """Write Detectron-format .npz with crop-space keypoints."""
    n = len(keypoints_per_frame)
    boxes, keypoints = [], []
    for i in range(n):
        kps = keypoints_per_frame[i]
        bb = bboxes_per_frame[i]
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

    metadata = {'w': CROP_SIZE, 'h': CROP_SIZE}
    boxes_arr = np.array(boxes, dtype=object)
    kps_arr = np.array(keypoints, dtype=object)
    segs_arr = np.array([None] * n, dtype=object)
    np.savez_compressed(out_path, boxes=boxes_arr, segments=segs_arr,
                        keypoints=kps_arr, metadata=metadata)
    n_ok = sum(1 for k in keypoints_per_frame if k is not None)
    print(f'  wrote {out_path}  ({n_ok}/{n} frames)')


def main():
    predictor = load_detectron_predictor()

    # Pass 1
    print('Pass 1: Full-frame detection + classification')
    frames, near_bb, far_bb, w, h, fps = pass1_full_frame_detect(predictor, VIDEO)

    # Pass 2: Per-player crop detection
    print('\nPass 2: Crop-based re-detection')
    near_kps_crop, near_bb_crop = pass2_crop_detect(predictor, frames, near_bb, 'near', w, h)
    far_kps_crop, far_bb_crop = pass2_crop_detect(predictor, frames, far_bb, 'far', w, h)

    # Save npz per player
    print('\nWriting detection npz files...')
    for role, kps, bb in [('near', near_kps_crop, near_bb_crop),
                           ('far', far_kps_crop, far_bb_crop)]:
        det_dir = os.path.join(OUT_DIR, f'detect_{role}')
        os.makedirs(det_dir, exist_ok=True)
        save_detectron_npz(kps, bb, os.path.join(det_dir, f'{VIDEO}.npz'))

    # Build VP3D custom datasets
    for role in ['near', 'far']:
        det_dir = os.path.join(ORIG_DIR, OUT_DIR, f'detect_{role}')
        data_dir = os.path.join(ORIG_DIR, V3D_DIR, 'data')
        print(f'\n--- Building custom dataset for {role} ---')
        subprocess.run(['python', 'prepare_data_2d_custom.py',
                         '-i', det_dir, '-o', f'tennis_{role}'],
                        cwd=data_dir, check=True)

    # Run VP3D per player
    vp3d_root = os.path.join(ORIG_DIR, V3D_DIR)
    subprocess.run(['cp', VIDEO, vp3d_root], check=True)
    for role in ['near', 'far']:
        out_mp4 = os.path.join(ORIG_DIR, OUT_DIR, f'output_{role}.mp4')
        out_npy = os.path.join(ORIG_DIR, OUT_DIR, f'output_{role}.npy')
        print(f'\n--- Running VideoPose3D for {role} ---')
        subprocess.run(['python', 'run.py',
                         '-d', 'custom', '-k', f'tennis_{role}',
                         '-arc', '3,3,3,3,3', '-c', 'checkpoint',
                         '--evaluate', CKPT, '--render',
                         '--viz-subject', VIDEO, '--viz-action', 'custom',
                         '--viz-camera', '0', '--viz-video', VIDEO,
                         '--viz-output', out_mp4, '--viz-export', out_npy,
                         '--viz-size', '6'],
                        cwd=vp3d_root, check=True)

    # Merge (far | near)
    print('\n--- Merging ---')
    near_mp4 = os.path.join(OUT_DIR, 'output_near.mp4')
    far_mp4 = os.path.join(OUT_DIR, 'output_far.mp4')
    combined = os.path.join(OUT_DIR, 'combined.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                    '-i', far_mp4, '-i', near_mp4,
                    '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
                    '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', '-movflags', '+faststart', combined],
                    check=True)
    print(f'\nDone:')
    print(f'  {near_mp4}')
    print(f'  {far_mp4}')
    print(f'  {combined}')


if __name__ == '__main__':
    main()
