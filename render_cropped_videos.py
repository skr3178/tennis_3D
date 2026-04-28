"""
Render the per-frame 512x512 crops that were fed to Detectron2 in Pass 2
of run_videopose3d.py — one video per player.

Useful for visual sanity-check: "what does Detectron2 see when it tries to
detect the near/far player?"
"""
import os
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np

# Import same constants/helpers from run_videopose3d.py
from run_videopose3d import (
    VIDEO, OUT_DIR, CROP_SIZE, CROP_PAD,
    load_detectron_predictor, classify_detections,
    make_square_crop, interp_bboxes,
)


def main():
    print('Loading Detectron2...')
    predictor = load_detectron_predictor()

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: {w}x{h} {fps}fps {n} frames')

    # Pass 1: full-frame detect + classify
    print('Pass 1: full-frame detection...')
    frames = []
    near_bb = [None] * n
    far_bb = [None] * n
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
        if i % 100 == 0:
            print(f'  Frame {i}/{n}')
        i += 1
    cap.release()

    # Fill short gaps
    near_bb = interp_bboxes(near_bb)
    far_bb = interp_bboxes(far_bb)

    # Pass 2: generate crops and write videos
    for role, bboxes in [('near', near_bb), ('far', far_bb)]:
        out_path = os.path.join(OUT_DIR, f'cropped_{role}.mp4')
        print(f'\nWriting {out_path}...')

        # Use a temp raw mp4 then re-encode to H.264/avc1
        tmp_path = os.path.join(OUT_DIR, f'_tmp_cropped_{role}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (CROP_SIZE, CROP_SIZE))

        for i in range(n):
            bb = bboxes[i]
            if bb is None:
                # black frame
                canvas = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                cv2.putText(canvas, 'no detection', (30, CROP_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 200), 2)
            else:
                x0, y0, size = make_square_crop(bb, w, h)
                x1, y1 = x0 + size, y0 + size
                crop = frames[i][int(y0):int(y1), int(x0):int(x1)]
                if crop.size == 0:
                    canvas = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                else:
                    canvas = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                        interpolation=cv2.INTER_LINEAR)
            cv2.putText(canvas, f'{role} f{i}', (8, CROP_SIZE - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()

        # Re-encode to H.264/avc1 for browser playback
        print(f'  Re-encoding to H.264...')
        os.system(f'ffmpeg -y -loglevel error -i {tmp_path} '
                  f'-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p '
                  f'-movflags +faststart {out_path}')
        os.remove(tmp_path)
        print(f'  Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
