"""3D world coordinates of the 14 court keypoints, in the same index order
TennisCourtDetector emits them.

Frame: X across the court (sidelines), Y along the court (baseline-to-baseline),
Z up. Origin at the center of the net on the court surface (z=0).

Keypoint index → court feature (mirrors TennisCourtDetector/court_reference.py
`key_points` ordering: baseline_top, baseline_bottom, left_inner_line,
right_inner_line, top_inner_line, bottom_inner_line, middle_line):

    0  doubles far-left baseline corner
    1  doubles far-right baseline corner
    2  doubles near-left baseline corner
    3  doubles near-right baseline corner
    4  singles far-left baseline corner
    5  singles near-left baseline corner
    6  singles far-right baseline corner
    7  singles near-right baseline corner
    8  far service line × left singles sideline (T-corner far-left)
    9  far service line × right singles sideline (T-corner far-right)
   10  near service line × left singles sideline (T-corner near-left)
   11  near service line × right singles sideline (T-corner near-right)
   12  far service line × center service line (far T)
   13  near service line × center service line (near T)
"""
import numpy as np

# Tennis court dimensions (ITF, meters)
DOUBLES_HALF_WIDTH = 5.485       # 10.97 / 2
SINGLES_HALF_WIDTH = 4.115       # 8.23 / 2
COURT_HALF_LENGTH = 11.885       # 23.77 / 2
SERVICE_LINE_FROM_NET = 6.40     # distance from net to each service line

COURT_KEYPOINTS_3D = np.array(
    [
        # 0,1: doubles far baseline (Y > 0)
        (-DOUBLES_HALF_WIDTH, +COURT_HALF_LENGTH, 0.0),
        (+DOUBLES_HALF_WIDTH, +COURT_HALF_LENGTH, 0.0),
        # 2,3: doubles near baseline (Y < 0)
        (-DOUBLES_HALF_WIDTH, -COURT_HALF_LENGTH, 0.0),
        (+DOUBLES_HALF_WIDTH, -COURT_HALF_LENGTH, 0.0),
        # 4,5: singles left sideline endpoints (far, near)
        (-SINGLES_HALF_WIDTH, +COURT_HALF_LENGTH, 0.0),
        (-SINGLES_HALF_WIDTH, -COURT_HALF_LENGTH, 0.0),
        # 6,7: singles right sideline endpoints (far, near)
        (+SINGLES_HALF_WIDTH, +COURT_HALF_LENGTH, 0.0),
        (+SINGLES_HALF_WIDTH, -COURT_HALF_LENGTH, 0.0),
        # 8,9: far service line endpoints (left, right)
        (-SINGLES_HALF_WIDTH, +SERVICE_LINE_FROM_NET, 0.0),
        (+SINGLES_HALF_WIDTH, +SERVICE_LINE_FROM_NET, 0.0),
        # 10,11: near service line endpoints (left, right)
        (-SINGLES_HALF_WIDTH, -SERVICE_LINE_FROM_NET, 0.0),
        (+SINGLES_HALF_WIDTH, -SERVICE_LINE_FROM_NET, 0.0),
        # 12,13: T-points (far, near)
        (0.0, +SERVICE_LINE_FROM_NET, 0.0),
        (0.0, -SERVICE_LINE_FROM_NET, 0.0),
    ],
    dtype=np.float64,
)
assert COURT_KEYPOINTS_3D.shape == (14, 3)
