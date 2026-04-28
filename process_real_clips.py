"""Eval our LSTM on all 21 TrackNet clips + build a real-clip fine-tune NPZ set.

For each clip id 0..20 in tennis_real.json:
  1. Load gt 3D + paper pred + Label.csv 2D
  2. DLT-recover camera (from gt 3D + Label 2D matched pairs)
  3. Save fine-tune NPZ:  paper_npz_rev1/real_tracknet_21/seq_NN.npz
     keys: uv, xyz, eot, intrinsics, extrinsic
  4. Run our model inference at 50fps (interp 25fps→50fps, subsample back)
  5. Metrics: ours vs gt, paper(pred_unrefined) vs gt, NRMSE
  6. Per-clip pred JSON + summary CSV + grid plot
"""
from __future__ import annotations
import os, re, sys, json, csv
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/media/skr/storage/ten_bad")
sys.path.insert(0, "/media/skr/storage/ten_bad/LSTM_Model")
from LSTM_Model.data.parameterization import pixel_to_plane_points
from LSTM_Model.pipeline import WhereIsTheBall

DATASET   = "/media/skr/storage/ten_bad/TrackNet/datasets/trackNet/Dataset"
DESC      = "/media/skr/storage/ten_bad/notes/screenshots/where_is_the_ball/data/tracknet_tid_desc.json"
REAL      = "/media/skr/storage/ten_bad/notes/screenshots/where_is_the_ball/data/tennis_real.json"
OUT_NPZ   = "/media/skr/storage/ten_bad/paper_npz_rev1/real_tracknet_21"
OUT_EVAL  = "/media/skr/storage/ten_bad/LSTM_Model/inference_output/eval_real21"
CKPT      = "/media/skr/storage/ten_bad/LSTM_Model/checkpoints_5k_v2/best.pt"
DEVICE    = "cuda"

os.makedirs(OUT_NPZ,  exist_ok=True)
os.makedirs(OUT_EVAL, exist_ok=True)


def load_jswrap(path):
    raw = open(path).read()
    m = re.match(r"\s*var\s+\w+\s*=\s*", raw)
    return json.loads(raw[m.end():].rstrip().rstrip(";"))


def dlt(uv, XYZ):
    n = len(uv); A = np.zeros((2*n, 12))
    for i in range(n):
        X, Y, Z = XYZ[i]; u, v = uv[i]
        A[2*i  ] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


def load_label_csv(clip_dir):
    labels = {}
    p = os.path.join(clip_dir, "Label.csv")
    with open(p) as f:
        for r in csv.DictReader(f):
            idx = int(r["file name"].split(".")[0])
            if int(r["visibility"]) == 1 and r["x-coordinate"] and r["y-coordinate"]:
                try:
                    labels[idx] = (float(r["x-coordinate"]),
                                   float(r["y-coordinate"]))
                except ValueError:
                    pass
    return labels


# Load metadata
desc = load_jswrap(DESC)
real = load_jswrap(REAL)

# Load model once
print(f"Loading model: {CKPT}")
net   = WhereIsTheBall(hidden=64).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
net.load_state_dict(state["model_state"])
net.eval()
print(f"  model epoch {state.get('epoch', '?')}")

results = []
all_preds = {}

for cid in sorted(desc.keys(), key=int):
    d = desc[cid]
    g, c, f0, f1 = d["g_name"], d["c_name"], d["f_start"], d["f_end"]
    N = f1 - f0 + 1
    clip_dir = os.path.join(DATASET, g, c)
    if not os.path.isdir(clip_dir):
        print(f"clip {cid}: dir missing, skipping"); continue

    # --- 3D gt + paper pred ---
    gt = np.asarray(real[cid]["gt"],            dtype=np.float64)
    pu = np.asarray(real[cid]["pred_unrefined"], dtype=np.float64)
    pr = np.asarray(real[cid]["pred_refined"],  dtype=np.float64)
    if len(gt) != N:
        print(f"clip {cid}: len mismatch  gt={len(gt)} expected={N}, skipping")
        continue
    # Detect placeholder zeros (clips 6-20): use pred_refined as pseudo-GT
    has_real_gt = bool(np.abs(gt).max() > 1e-6)
    target = gt if has_real_gt else pr

    # --- 2D Label.csv ---
    labels = load_label_csv(clip_dir)
    vis_uv, vis_xyz, vis_idx = [], [], []
    for i in range(N):
        fr = f0 + i
        if fr in labels:
            vis_uv.append(labels[fr]); vis_xyz.append(target[i]); vis_idx.append(i)
    if len(vis_uv) < 8:
        print(f"clip {cid}: only {len(vis_uv)} usable pairs, skipping")
        continue
    vis_uv  = np.asarray(vis_uv,  dtype=np.float64)
    vis_xyz = np.asarray(vis_xyz, dtype=np.float64)
    vis_idx = np.asarray(vis_idx, dtype=int)

    # --- DLT camera (OpenCV-style) → convert to OpenGL extrinsic ---
    P_proj = dlt(vis_uv, vis_xyz)
    K, R_cv, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P_proj)
    K   = K / K[2, 2]
    C   = (t_h[:3] / t_h[3]).flatten()
    t_cv = (-R_cv @ C)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsics = np.array([fx, cx, cy], dtype=np.float32)

    # reprojection sanity
    Rrod, _ = cv2.Rodrigues(R_cv)
    proj, _ = cv2.projectPoints(vis_xyz.reshape(-1, 1, 3),
                                 Rrod, t_cv, K, None)
    reproj_err_px = float(np.linalg.norm(proj.reshape(-1, 2) - vis_uv,
                                          axis=1).mean())

    # OpenGL extrinsic
    flip = np.diag([1, -1, -1])
    R_gl = flip @ R_cv
    t_gl = flip @ t_cv
    E_gl = np.eye(4); E_gl[:3, :3] = R_gl; E_gl[:3, 3] = t_gl
    extrinsic = E_gl.astype(np.float32)

    # --- Build full uv array (interpolated for invisible frames) ---
    uv_full = np.full((N, 2), np.nan)
    for i, idx in enumerate(vis_idx):
        uv_full[idx] = vis_uv[i]
    valid = ~np.isnan(uv_full[:, 0])
    iv = np.where(valid)[0]
    uv_full[:, 0] = np.interp(np.arange(N), iv, uv_full[iv, 0])
    uv_full[:, 1] = np.interp(np.arange(N), iv, uv_full[iv, 1])

    # --- eot from y-bounce local minima (use whichever 3D source we have) ---
    eot = np.zeros(N, dtype=np.uint8)
    y = target[:, 1]
    if N >= 3:
        for i in range(1, N - 1):
            if y[i] <= y[i-1] and y[i] <= y[i+1] and y[i] < 0.20:
                eot[i] = 1
    if y[0]  < 0.30: eot[0]  = 1
    if y[-1] < 0.30: eot[-1] = 1

    # --- Save fine-tune NPZ ONLY for clips with real GT ---
    if has_real_gt:
        npz_path = os.path.join(OUT_NPZ, f"seq_{int(cid):02d}.npz")
        np.savez(npz_path,
                 uv=uv_full.astype(np.float32),
                 xyz=gt.astype(np.float32),
                 eot=eot,
                 intrinsics=intrinsics,
                 extrinsic=extrinsic,
                 clip_id=np.int32(int(cid)),
                 game=g, clip=c, f_start=np.int32(f0), f_end=np.int32(f1))

    # --- Inference at 50fps (linear interp 25→50, subsample back) ---
    N50 = 2 * N - 1
    uv50 = np.zeros((N50, 2))
    uv50[::2]  = uv_full
    uv50[1::2] = 0.5 * (uv_full[:-1] + uv_full[1:])

    P = pixel_to_plane_points(uv50, intrinsics, extrinsic, convention="opengl")
    if np.isnan(P).any():
        print(f"clip {cid}: NaN plane-points, skipping inference")
        continue

    P_t = torch.from_numpy(P).float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(P)], dtype=torch.long)
    with torch.no_grad():
        out = net(P_t, lengths=lengths)
    xyz_50 = out["xyz_final"][0].cpu().numpy()
    xyz_25 = xyz_50[::2]   # (N, 3)

    # --- Metrics ---
    # vs target (real GT if avail, else paper pred_refined as pseudo-GT)
    err_ours_vs_target = np.linalg.norm(xyz_25 - target, axis=1)
    err_ours_vs_paper  = np.linalg.norm(xyz_25 - pu,     axis=1)
    target_range = float(np.linalg.norm(target.max(0) - target.min(0)))
    nrmse_o = float(np.sqrt((err_ours_vs_target**2).mean()) / max(target_range, 1e-9))
    if has_real_gt:
        err_paper_vs_target = np.linalg.norm(pu - gt, axis=1)
        nrmse_p = float(np.sqrt((err_paper_vs_target**2).mean()) / max(target_range, 1e-9))
        mean_err_paper = float(err_paper_vs_target.mean())
        max_err_paper  = float(err_paper_vs_target.max())
    else:
        nrmse_p = float("nan"); mean_err_paper = float("nan"); max_err_paper = float("nan")

    rec = dict(cid=int(cid), game=g, clip=c, N=N, visible=len(vis_uv),
               has_real_gt=int(has_real_gt),
               reproj_err_px=reproj_err_px, fx=float(fx),
               mean_err_ours_vs_target_m=float(err_ours_vs_target.mean()),
               mean_err_ours_vs_paper_m=float(err_ours_vs_paper.mean()),
               mean_err_paper_vs_target_m=mean_err_paper,
               max_err_ours_m=float(err_ours_vs_target.max()),
               max_err_paper_m=max_err_paper,
               nrmse_ours=nrmse_o, nrmse_paper=nrmse_p,
               target_range=target_range)
    results.append(rec)
    all_preds[int(cid)] = dict(xyz_ours=xyz_25, xyz_gt=target, xyz_paper=pu,
                                has_real_gt=has_real_gt)
    flag = "GT" if has_real_gt else "pseudo"
    paper_str = f"{mean_err_paper:.2f}m" if has_real_gt else "  -  "
    print(f"clip {cid:>2s}  {g:>6s}/{c:<8s}  N={N:>3d} vis={len(vis_uv):>3d}  {flag:>6s}  "
          f"reproj={reproj_err_px:>4.2f}px  "
          f"ours_vs_t={err_ours_vs_target.mean():>5.2f}m  "
          f"paper_vs_gt={paper_str}  NRMSE_o={nrmse_o:.3f}")

# --- Summary CSV ---
summary_csv = os.path.join(OUT_EVAL, "summary.csv")
with open(summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nwrote {summary_csv}")
print(f"wrote {len(results)} npz files to {OUT_NPZ}/")

# --- Aggregate stats ---
gt_recs = [r for r in results if r["has_real_gt"]]
all_recs = results

if gt_recs:
    err_o = np.array([r["mean_err_ours_vs_target_m"]  for r in gt_recs])
    err_p = np.array([r["mean_err_paper_vs_target_m"] for r in gt_recs])
    print(f"\nAggregate vs REAL GT (n={len(gt_recs)} clips):")
    print(f"  mean dist err  ours:  {err_o.mean():.3f} m  median: {np.median(err_o):.3f} m")
    print(f"  mean dist err  paper: {err_p.mean():.3f} m  median: {np.median(err_p):.3f} m")

err_op = np.array([r["mean_err_ours_vs_paper_m"] for r in all_recs])
print(f"\nAggregate vs PAPER pred_unrefined (n={len(all_recs)} clips):")
print(f"  mean dist ours-vs-paper: {err_op.mean():.3f} m  median: {np.median(err_op):.3f} m")

# --- Grid plot: y vs frame for all clips ---
ncol = 4
nrow = (len(results) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 2.4*nrow), sharey=False)
axes = np.array(axes).reshape(-1)
for ax, rec in zip(axes, results):
    cid = rec["cid"]
    p = all_preds[cid]
    label_t = "gt" if p["has_real_gt"] else "paper pred_refined (pseudo-gt)"
    ax.plot(p["xyz_gt"][:, 1],    "k-",   lw=1.4, label=label_t)
    ax.plot(p["xyz_paper"][:, 1], "C0--", lw=1.0, label="pred_unrefined")
    ax.plot(p["xyz_ours"][:, 1],  "C3-",  lw=1.0, label="ours")
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    err_str = f'{rec["mean_err_ours_vs_target_m"]:.1f}m'
    pap_str = (f' paper={rec["mean_err_paper_vs_target_m"]:.2f}m'
               if rec["has_real_gt"] else "")
    ax.set_title(f'{rec["game"]}/{rec["clip"]}  ours={err_str}{pap_str}', fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[len(results):]:
    ax.axis("off")
axes[0].legend(fontsize=7)
fig.suptitle(f"All 21 TrackNet clips — height (y) traces  |  ours v2 (ep{state.get('epoch','?')})", y=1.0)
fig.tight_layout()
plot_path = os.path.join(OUT_EVAL, "grid_y_vs_frame.png")
fig.savefig(plot_path, dpi=110, bbox_inches="tight")
print(f"wrote {plot_path}")

# --- Per-clip distance error grid ---
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 2.4*nrow), sharey=False)
axes = np.array(axes).reshape(-1)
for ax, rec in zip(axes, results):
    cid = rec["cid"]; p = all_preds[cid]
    err_o_seq = np.linalg.norm(p["xyz_ours"]  - p["xyz_gt"], axis=1)
    ax.plot(err_o_seq, "C3-", lw=1.0, label=f'ours μ={err_o_seq.mean():.2f}m')
    if p["has_real_gt"]:
        err_p_seq = np.linalg.norm(p["xyz_paper"] - p["xyz_gt"], axis=1)
        ax.plot(err_p_seq, "C0-", lw=1.0, label=f'paper μ={err_p_seq.mean():.2f}m')
    flag = "" if p["has_real_gt"] else " (pseudo-gt)"
    ax.set_title(f'{rec["game"]}/{rec["clip"]}{flag}', fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6)
for ax in axes[len(results):]:
    ax.axis("off")
fig.suptitle("All 21 clips — per-frame xyz distance error", y=1.0)
fig.tight_layout()
err_path = os.path.join(OUT_EVAL, "grid_err_per_clip.png")
fig.savefig(err_path, dpi=110, bbox_inches="tight")
print(f"wrote {err_path}")
