"""Camera-info I/O. Format mirrors tt3d/tt3d/calibration/utils.py so the
reconstruction core can be reused unchanged."""
import numpy as np
import yaml


def get_K(f, h, w):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def write_camera_info(yaml_path, rvec, tvec, f, h, w):
    info = {
        "rvec": np.asarray(rvec).flatten().tolist(),
        "tvec": np.asarray(tvec).flatten().tolist(),
        "f": float(f),
        "h": int(h),
        "w": int(w),
    }
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(info, fh)


def read_camera_info(yaml_path):
    with open(yaml_path, "r") as fh:
        info = yaml.safe_load(fh)
    return (
        np.array(info["rvec"], dtype=np.float64),
        np.array(info["tvec"], dtype=np.float64),
        float(info["f"]),
        int(info["h"]),
        int(info["w"]),
    )
