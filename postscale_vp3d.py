"""
Post-scale VideoPose3D .npy outputs to hit a target body span.
Uniform scaling — preserves pose, just fixes overall scale.
"""
import numpy as np

TARGET_BODY_SPAN = 1.70  # meters (real human height approximation)


def rescale_vp3d_output(npy_path):
    """Load, scale, overwrite."""
    arr = np.load(npy_path)
    # H36M joints: 10 = head, 3 = r_foot, 6 = l_foot, Y-down convention
    # body span = head Y (negative) - foot Y (positive after flip)
    head_y = -arr[:, 10, 1]            # convert to Y-up for measurement
    feet_y = -(arr[:, 3, 1] + arr[:, 6, 1]) / 2
    span_per_frame = head_y - feet_y
    median_span = np.median(span_per_frame)
    scale = TARGET_BODY_SPAN / median_span
    print(f'  {npy_path}: median span={median_span:.3f}m → scale {scale:.3f}x '
          f'→ target {TARGET_BODY_SPAN:.2f}m')

    # Uniform scale around hip root (joint 0)
    root = arr[:, 0:1, :]  # (F, 1, 3)
    scaled = root + (arr - root) * scale
    np.save(npy_path, scaled)

    # Verify
    arr2 = np.load(npy_path)
    new_head = -arr2[:, 10, 1]
    new_feet = -(arr2[:, 3, 1] + arr2[:, 6, 1]) / 2
    new_span = (new_head - new_feet).mean()
    wrist_deltas = np.diff(arr2[:, 13, :], axis=0)
    ws = np.linalg.norm(wrist_deltas, axis=1)
    print(f'    after: body span mean={new_span:.3f}m, '
          f'wrist speed mean={ws.mean()*50:.2f} m/s, max={ws.max()*50:.2f} m/s')


if __name__ == '__main__':
    print('Post-scaling VP3D outputs...')
    rescale_vp3d_output('videopose3d_output/output_near.npy')
    rescale_vp3d_output('videopose3d_output/output_far.npy')
