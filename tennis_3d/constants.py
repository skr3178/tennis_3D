"""Tennis ball + court physics constants.

Layout mirrors tt3d/tt3d/rally/constants.py so the same dynamics module
(`casadi_dae.py`) can consume it after a single import-line swap.

Surface defaults: hard court. Override COR/MU at call sites for clay/grass.
"""
import numpy as np

# --- Universal ---
PI = np.pi
RHO = 1.225          # Air density [kg/m^3]
NU = 1.48e-5         # Air kinematic viscosity [m^2/s]
G = 9.81             # Gravity [m/s^2]

# --- Tennis ball ---
R = 0.0335           # Ball radius [m] (ITF: dia 6.54-6.86 cm)
M = 0.058            # Ball mass [kg] (ITF: 56.0-59.4 g)

# --- Aerodynamic coefficients (tt3d convention: F_drag = -KD*|v|*v) ---
# Drag: KD = 0.5 * RHO * Cd * A, with Cd ≈ 0.55 for tennis ball at typical Re,
# A = pi*R^2.
KD = 0.5 * RHO * 0.55 * PI * R ** 2     # ≈ 1.187e-3
# Magnus: tt3d uses linearised F_M = KM * (omega x v). Scaled from TT value
# (4.86e-6) by the ratio of (rho * pi * R^3) between tennis and TT, then
# rounded to a literature-consistent value (Cross, "Physics of Tennis").
KM = 3.0e-5

# --- Bounce model (hard court) ---
COR = 0.75           # Coefficient of restitution (z rebound)
MU = 0.6             # Friction coefficient at the surface

# --- Derived ---
S = PI * R ** 2      # Cross-sectional area [m^2]
I = (2.0 / 3.0) * M * R ** 2   # Hollow-sphere moment of inertia [kg m^2]
