import json
import matplotlib.pyplot as plt
import numpy as np

log_path = "/media/skr/storage/ten_bad/LSTM_Model/checkpoints_5k/train_log.jsonl"

epochs, total, l_eps, l_3d, l_b = [], [], [], [], []
val_epochs, val_dist, val_h = [], [], []

with open(log_path) as f:
    for line in f:
        rec = json.loads(line)
        ep = rec["epoch"]
        epochs.append(ep)
        total.append(rec["total"])
        l_eps.append(rec["L_eps"])
        l_3d.append(rec["L_3D"])
        l_b.append(rec["L_B"])
        if "val_nrmse_distance" in rec:
            val_epochs.append(ep)
            val_dist.append(rec["val_nrmse_distance"])
            val_h.append(rec["val_nrmse_height"])

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Training Progress — 5k Dataset (Camera Randomization, σ_uv=5px)", fontsize=14, fontweight='bold')

# 1. Total loss (log scale)
ax = axes[0, 0]
ax.semilogy(epochs, total, alpha=0.4, linewidth=0.5, color='blue')
# smoothed
w = 20
if len(total) > w:
    smoothed = np.convolve(np.log10(np.clip(total, 1e-6, None)), np.ones(w)/w, mode='valid')
    ax.semilogy(epochs[w-1:], 10**smoothed, color='blue', linewidth=2, label=f'smoothed (w={w})')
ax.set_ylabel("Total Loss (log)")
ax.set_xlabel("Epoch")
ax.set_title("Total Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. L_3D (log scale)
ax = axes[0, 1]
ax.semilogy(epochs, l_3d, alpha=0.4, linewidth=0.5, color='red')
if len(l_3d) > w:
    smoothed = np.convolve(np.log10(np.clip(l_3d, 1e-6, None)), np.ones(w)/w, mode='valid')
    ax.semilogy(epochs[w-1:], 10**smoothed, color='red', linewidth=2, label=f'smoothed (w={w})')
ax.set_ylabel("L_3D (log)")
ax.set_xlabel("Epoch")
ax.set_title("L_3D (3D Position Loss)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. L_eps
ax = axes[1, 0]
ax.plot(epochs, l_eps, alpha=0.4, linewidth=0.5, color='green')
if len(l_eps) > w:
    smoothed = np.convolve(l_eps, np.ones(w)/w, mode='valid')
    ax.plot(epochs[w-1:], smoothed, color='green', linewidth=2, label=f'smoothed (w={w})')
ax.set_ylabel("L_eps")
ax.set_xlabel("Epoch")
ax.set_title("L_eps (Parameterization Loss)")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. L_B
ax = axes[1, 1]
ax.plot(epochs, l_b, alpha=0.4, linewidth=0.5, color='purple')
if len(l_b) > w:
    smoothed = np.convolve(l_b, np.ones(w)/w, mode='valid')
    ax.plot(epochs[w-1:], smoothed, color='purple', linewidth=2, label=f'smoothed (w={w})')
ax.set_ylabel("L_B")
ax.set_xlabel("Epoch")
ax.set_title("L_B (Bounce Loss)")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Val NRMSE distance
ax = axes[2, 0]
ax.plot(val_epochs, val_dist, 'o-', markersize=3, color='darkorange', linewidth=1)
best_idx = np.argmin(val_dist)
ax.axhline(val_dist[best_idx], color='red', linestyle='--', alpha=0.5, label=f'best={val_dist[best_idx]:.4f} (ep {val_epochs[best_idx]})')
ax.set_ylabel("NRMSE Distance")
ax.set_xlabel("Epoch")
ax.set_title("Validation NRMSE Distance")
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Val NRMSE height
ax = axes[2, 1]
ax.plot(val_epochs, val_h, 'o-', markersize=3, color='teal', linewidth=1)
best_idx_h = np.argmin(val_h)
ax.axhline(val_h[best_idx_h], color='red', linestyle='--', alpha=0.5, label=f'best={val_h[best_idx_h]:.4f} (ep {val_epochs[best_idx_h]})')
ax.set_ylabel("NRMSE Height")
ax.set_xlabel("Epoch")
ax.set_title("Validation NRMSE Height")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/media/skr/storage/ten_bad/training_5k_curves.png", dpi=150)
print(f"Saved. {len(epochs)} epochs logged, {len(val_epochs)} val checkpoints")
