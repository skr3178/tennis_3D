"""
Render a side-by-side comparison:
  Left: original video
  Right: 3D scene with both near and far player skeletons
"""
import os
import subprocess
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

VIDEO = 'S_Original_HL_clip_cropped.mp4'
OUT = 'videopose3d_output/side_by_side.mp4'

# H36M 17-joint skeleton (parent of each joint, -1 = root)
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
H36M_RIGHT = {1, 2, 3, 14, 15, 16}


def h36m_to_display(poses):
    """Convert VP3D output (X-right, Y-down, Z-depth) → display (X, Y, Z-up).
    Also anchor feet at Z=0.
    """
    p = np.stack([
        poses[:, :, 0],      # X lateral
        poses[:, :, 2],      # Z forward (depth) → display Y
        -poses[:, :, 1],     # -Y → display Z (up)
    ], axis=2)
    # Lift feet to ground
    feet_z = (p[:, 3, 2] + p[:, 6, 2]) / 2
    p[:, :, 2] -= feet_z[:, None]
    return p


def main():
    # Load poses
    print('Loading poses...')
    poses_near = h36m_to_display(np.load('videopose3d_output/output_near.npy'))
    poses_far = h36m_to_display(np.load('videopose3d_output/output_far.npy'))

    # Place players on opposite sides of an imaginary court:
    # For display in this "side-by-side" view, translate far player to the
    # left of world origin and near player to the right, so they're visible together
    poses_far[:, :, 0] -= 1.5   # shift far left
    poses_near[:, :, 0] += 1.5  # shift near right

    # Read original video frames
    print('Reading video frames...')
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    n = min(len(frames), len(poses_near), len(poses_far))
    frames = frames[:n]
    poses_near = poses_near[:n]
    poses_far = poses_far[:n]

    # Set up figure: left = video, right = 3D skeleton plot
    size = 7
    fig = plt.figure(figsize=(size * 2.2, size))

    # Left — video
    ax_vid = fig.add_subplot(1, 2, 1)
    ax_vid.axis('off')
    ax_vid.set_title('Original video')

    # Right — 3D
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.view_init(elev=15., azim=-70)
    ax_3d.set_xlim3d(-4, 4)
    ax_3d.set_ylim3d(-2, 2)
    ax_3d.set_zlim3d(0, 2.0)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Reconstruction (far=blue/red, near=green/orange)')

    # Draw a simple ground grid for reference
    xx, yy = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(-2, 2, 5))
    ax_3d.plot_wireframe(xx, yy, np.zeros_like(xx),
                         color='lightgray', linewidth=0.5, alpha=0.4)

    image_h = [None]
    lines_near = []
    lines_far = []
    initialized = [False]

    def draw_skeleton_init(poses, colors_map):
        """Create skeleton lines for first frame."""
        lines = []
        pos = poses[0]
        for j, jp in enumerate(H36M_PARENTS):
            if jp == -1:
                lines.append(None)
                continue
            c = colors_map[1] if j in H36M_RIGHT else colors_map[0]
            ln = ax_3d.plot(
                [pos[j, 0], pos[jp, 0]],
                [pos[j, 1], pos[jp, 1]],
                [pos[j, 2], pos[jp, 2]],
                color=c, linewidth=2.2)[0]
            lines.append(ln)
        return lines

    def update_skeleton(lines, poses, i):
        pos = poses[i]
        for j, jp in enumerate(H36M_PARENTS):
            if jp == -1:
                continue
            ln = lines[j]
            ln.set_xdata(np.array([pos[j, 0], pos[jp, 0]]))
            ln.set_ydata(np.array([pos[j, 1], pos[jp, 1]]))
            ln.set_3d_properties(np.array([pos[j, 2], pos[jp, 2]]), zdir='z')

    def update(i):
        if not initialized[0]:
            image_h[0] = ax_vid.imshow(frames[i])
            lines_far.extend(draw_skeleton_init(poses_far,
                                                ('tab:blue', 'tab:red')))
            lines_near.extend(draw_skeleton_init(poses_near,
                                                 ('tab:green', 'tab:orange')))
            # Add labels
            ax_3d.text(poses_far[0, 0, 0], poses_far[0, 0, 1], 2.0,
                       'FAR', color='tab:blue', fontsize=10, weight='bold')
            ax_3d.text(poses_near[0, 0, 0], poses_near[0, 0, 1], 2.0,
                       'NEAR', color='tab:green', fontsize=10, weight='bold')
            initialized[0] = True
        else:
            image_h[0].set_data(frames[i])
            update_skeleton(lines_far, poses_far, i)
            update_skeleton(lines_near, poses_near, i)

        if i % 50 == 0:
            print(f'  Frame {i}/{n}', flush=True)

    fig.tight_layout()
    print('Rendering animation...')
    anim = FuncAnimation(fig, update, frames=np.arange(n),
                         interval=1000 / fps, repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata={}, bitrate=4000)
    anim.save(OUT, writer=writer)
    plt.close()
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
