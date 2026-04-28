"""
Render PromptHMR's detected/tracked bboxes onto the input video as an overlay
MP4. Reads results.pkl from a finished demo_video.py run.

This shows what PromptHMR actually saw and tracked — distinct from the
YOLO+PlayerTracker view used by the MediaPipe / VideoPose3D pipelines.

Output is H.264 (avc1) via ffmpeg pipe (OpenCV's bundled ffmpeg lacks x264).
"""
import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
import joblib

PROMPTHMR_RESULTS_ROOT = '/media/skr/storage/ten_bad/PromptHMR/results'

# Per-track color palette (BGR).
COLORS = [
    (0, 255, 0),     # green
    (0, 165, 255),   # orange
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 0, 255),     # red
    (180, 105, 255), # pink
]


def load_results(video_basename):
    pkl_path = os.path.join(PROMPTHMR_RESULTS_ROOT, video_basename, 'results.pkl')
    if not os.path.exists(pkl_path):
        sys.exit(f'No results.pkl at {pkl_path}')
    print(f'Loading {pkl_path}...')
    return joblib.load(pkl_path)


def draw_overlay(frame, frame_idx, results):
    """Draw all tracks present at this frame as colored bboxes + track IDs."""
    h, w = frame.shape[:2]
    for color_idx, (tid, track) in enumerate(results['people'].items()):
        frames = np.asarray(track.get('frames', []))
        if len(frames) == 0:
            continue
        # Find row in this track for the current frame
        match = np.where(frames == frame_idx)[0]
        if len(match) == 0:
            continue
        row = int(match[0])
        bbox = np.asarray(track['bboxes'])[row]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        color = COLORS[color_idx % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f'track {tid}'
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'frame {frame_idx}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render(video_path, results, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: {n} frames @ {fps} fps, {w}x{h}')
    print(f'# tracks: {len(results.get("people", {}))}')

    ff_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_overlay(frame, i, results)
        proc.stdin.write(frame.tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f'Wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Path to input video used by demo_video.py')
    ap.add_argument('--out', required=True, help='Output MP4 path')
    args = ap.parse_args()

    basename = os.path.splitext(os.path.basename(args.video))[0]
    results = load_results(basename)
    render(args.video, results, args.out)


if __name__ == '__main__':
    main()
