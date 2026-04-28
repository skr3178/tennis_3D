"""
Render an animation from VideoPose3D post-scaled .npy output.
Left: original video (with tracked player bbox highlighted).
Right: 3D skeleton plot.
"""
import os
import sys
import subprocess
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

# H36M 17-joint skeleton (parent of each joint, -1 = root)
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# Which side is 'right' (for coloring)
H36M_RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def render_player(poses_3d, video_path, output_path, fps, player_role, bboxes=None):
    """Render a single-player animation (2D video | 3D skeleton)."""
    n_frames = len(poses_3d)

    # Read all frames from video
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    n_frames = min(n_frames, len(all_frames))
    all_frames = all_frames[:n_frames]
    poses = poses_3d[:n_frames]
    if bboxes is not None:
        bboxes = bboxes[:n_frames]

    size = 6
    fig = plt.figure(figsize=(size * 2, size))
    ax_in = fig.add_subplot(1, 2, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title(f'Input ({player_role})')

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.view_init(elev=15., azim=70)
    radius = 1.7
    ax_3d.set_xlim3d([-radius / 2, radius / 2])
    ax_3d.set_ylim3d([-radius / 2, radius / 2])
    ax_3d.set_zlim3d([0, radius])
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(f'Reconstruction ({player_role})')

    # Root trajectory (for panning the 3D axes with the hip)
    # VP3D outputs Y-down in their coord system; we flip for display
    # We need (X, Y) for the "ground plane" to pan
    trajectory = poses[:, 0, :2]  # hip XY

    image_handle = [None]
    bbox_rect = [None]
    lines_3d = []
    initialized = [False]

    def update(i):
        # Pan 3D axes to keep hip centered (optional — VP3D does this)
        ax_3d.set_xlim3d([-radius / 2 + trajectory[i, 0],
                           radius / 2 + trajectory[i, 0]])
        ax_3d.set_ylim3d([-radius / 2 + trajectory[i, 1],
                           radius / 2 + trajectory[i, 1]])

        if not initialized[0]:
            image_handle[0] = ax_in.imshow(all_frames[i], aspect='equal')
            if bboxes is not None and bboxes[i] is not None:
                bb = bboxes[i]
                from matplotlib.patches import Rectangle
                rect = Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                  linewidth=2, edgecolor='lime', facecolor='none')
                ax_in.add_patch(rect)
                bbox_rect[0] = rect

            pos = poses[i]
            for j, j_parent in enumerate(H36M_PARENTS):
                if j_parent == -1:
                    lines_3d.append(None)
                    continue
                col = 'red' if j in H36M_RIGHT_JOINTS else 'black'
                line = ax_3d.plot(
                    [pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,
                    linewidth=2,
                )[0]
                lines_3d.append(line)
            initialized[0] = True
        else:
            image_handle[0].set_data(all_frames[i])
            if bbox_rect[0] is not None and bboxes is not None and bboxes[i] is not None:
                bb = bboxes[i]
                bbox_rect[0].set_xy((bb[0], bb[1]))
                bbox_rect[0].set_width(bb[2]-bb[0])
                bbox_rect[0].set_height(bb[3]-bb[1])

            pos = poses[i]
            for j, j_parent in enumerate(H36M_PARENTS):
                if j_parent == -1:
                    continue
                ln = lines_3d[j]
                ln.set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                ln.set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                ln.set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

        if i % 50 == 0:
            print(f'  Frame {i}/{n_frames}', flush=True)

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=np.arange(0, n_frames),
                         interval=1000 / fps, repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata={}, bitrate=3000)
    print(f'  Saving {output_path}...')
    anim.save(output_path, writer=writer)
    plt.close()


def load_bboxes(npz_path):
    """Extract per-frame bboxes (in original video coords) from detection npz.
    Our saved npz has rescaled bboxes when rescale=True was used, so instead
    we need to re-detect to get original bboxes. But we don't need bboxes for
    the 3D rendering — just for drawing a box on the input video. Skip for now.
    """
    return None


def main():
    video = 'S_Original_HL_clip_cropped.mp4'
    fps = 50
    out_dir = 'videopose3d_output'

    for role in ['near', 'far']:
        npy_path = os.path.join(out_dir, f'output_{role}.npy')
        mp4_path = os.path.join(out_dir, f'output_{role}_scaled.mp4')
        print(f'\n--- Rendering {role} player ---')
        poses = np.load(npy_path)

        # Convert H36M camera space (Y-down, Z-forward) to display convention
        # We want: X-right, Y-forward, Z-up for the 3D plot (matplotlib default)
        # VP3D output: (X, Y, Z) where Y is down-ish, Z is depth-ish
        # Swap: display[x, z, y] (use -Y for height)
        poses_display = np.stack([
            poses[:, :, 0],       # X (lateral)
            poses[:, :, 2],       # Z → matplotlib Y (depth)
            -poses[:, :, 1],      # -Y → matplotlib Z (up)
        ], axis=2)

        # Also translate so feet are at Z=0 for consistent plotting
        feet_z = (poses_display[:, 3, 2] + poses_display[:, 6, 2]) / 2
        poses_display[:, :, 2] -= feet_z[:, None]

        render_player(poses_display, video, mp4_path, fps, role, bboxes=None)
        print(f'  Saved {mp4_path}')

    # Side-by-side
    near_mp4 = os.path.join(out_dir, 'output_near_scaled.mp4')
    far_mp4 = os.path.join(out_dir, 'output_far_scaled.mp4')
    combined = os.path.join(out_dir, 'combined_scaled.mp4')
    print('\n--- Merging ---')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                    '-i', far_mp4, '-i', near_mp4,
                    '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
                    '-map', '[v]',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                    '-movflags', '+faststart', combined], check=True)
    print(f'\nDone:')
    print(f'  {near_mp4}')
    print(f'  {far_mp4}')
    print(f'  {combined}')


if __name__ == '__main__':
    main()
