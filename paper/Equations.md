# Equations & LSTM Architecture — *Where Is The Ball* (2025)

Verbatim distillation of the math and architecture descriptions from
Ponglertnapakorn & Suwajanakorn, *Where Is The Ball: 3D Ball Trajectory
Estimation From 2D Monocular Tracking*, arXiv 2506.05763 (Jun 2025).
Section numbers refer to the paper.

---

## Symbol & shape reference

Master lookup of every symbol in the paper with its mathematical domain
(per-step) and tensor shape (per-batch, per-sequence). Throughout:

- **B** = batch size (paper uses 256)
- **L** = sequence length (variable per sample; Synthetic Tennis: 64–822, mean 122)
- **N** = sequence length when discussing a single sequence (= L for that sample)
- All real-valued quantities are float32 in code.

### Geometric inputs

| Symbol | Per-step domain | Per-sequence shape | Per-batch shape | Description |
|---|---|---|---|---|
| `(u_t, v_t)` | `R²` | `(L, 2)` | `(B, L, 2)` | 2D pixel ball location at time `t` |
| `f` | `R` | `(1,)` | `(B, 1)` | focal length (intrinsic, per camera/sequence) |
| `(p_x, p_y)` | `R²` | `(2,)` | `(B, 2)` | principal point (intrinsic) |
| `E` | `SE(3) ⊂ R^{4×4}` | `(4, 4)` | `(B, 4, 4)` | camera extrinsic, world→camera |
| `E⁻¹` | `R^{4×4}` | `(4, 4)` | `(B, 4, 4)` | camera→world |
| `c` | `R³` | `(3,)` | `(B, 3)` | camera center (Eq. 1) — same for whole sequence if camera fixed |
| `d_t` | `R³` | `(L, 3)` | `(B, L, 3)` | ray direction at time `t` (Eq. 2) |
| `r_t(s) = c + s·d_t` | `R³` | – | – | parametric ray; `s ∈ R` is scalar |
| `ψ` | `R⁴ → R³` | – | – | dehomogenize: `ψ([x y z w]ᵀ) = [x y z]ᵀ` |
| `s_t*` | `R` | `(L,)` | `(B, L)` | ray parameter solving `r_t^y(s_t*) = h_t^refined` |

### Plane-points parameterization (canonical input)

| Symbol | Per-step domain | Per-sequence shape | Per-batch shape | Description |
|---|---|---|---|---|
| `p_ground,t`  (full 3D) | `R³` with `y ≡ 0` | `(L, 3)` | `(B, L, 3)` | ray ∩ ground plane `{y=0}` |
| `p_vertical,t`  (full 3D) | `R³` with `z ≡ 0` | `(L, 3)` | `(B, L, 3)` | ray ∩ vertical plane `{z=0}` |
| `p_ground,t`  (compact) | `R²` = `(x, z)` | `(L, 2)` | `(B, L, 2)` | y-coord dropped (always 0) |
| `p_vertical,t`  (compact) | `R²` = `(x, y)` | `(L, 2)` | `(B, L, 2)` | z-coord dropped (always 0) |
| `P_t` | `R⁴` | `(L, 4)` | `(B, L, 4)` | concat: `(p_g.x, p_g.z, p_v.x, p_v.y)` |
| `ΔP_t = P_{t+1} − P_t` | `R⁴` | `(L, 4)` | `(B, L, 4)` | temporal diff; last step zero-padded |

### EoT network outputs

| Symbol | Per-step domain | Per-sequence shape | Per-batch shape | Description |
|---|---|---|---|---|
| `ε_t` | `[0, 1] ⊂ R` | `(L, 1)` | `(B, L, 1)` | predicted EoT probability |
| `ε_t^gt` | `{0, 1}` | `(L, 1)` | `(B, L, 1)` | ground-truth EoT flag (uint8 / float) |
| `γ` | `[0, 1] ⊂ R` | scalar | scalar | class-balance weight in BCE (Eq. 4) |

### Height network outputs

| Symbol | Per-step domain | Per-sequence shape | Per-batch shape | Description |
|---|---|---|---|---|
| `h_t^f` | `R` | `(L, 1)` | `(B, L, 1)` | accumulated forward height; `h_0^f = 0` |
| `h_t^b` | `R` | `(L, 1)` | `(B, L, 1)` | accumulated backward height; `h_N^b = 0` |
| `Δh_t^f` | `R` | `(L, 1)` | `(B, L, 1)` | predicted forward height delta from LSTM^f |
| `Δh_t^b` | `R` | `(L, 1)` | `(B, L, 1)` | predicted backward height delta from LSTM^b |
| `w_t = (t−1)/(N−1)` | `[0, 1]` | `(L, 1)` | `(1, L, 1)` (broadcast) | ramp weight (Eq. 3) |
| `h_t` | `R` | `(L, 1)` | `(B, L, 1)` | combined height (Eq. 3) |
| `h_t^refined` | `R` | `(L, 1)` | `(B, L, 1)` | LSTM^height output |

### Refinement network outputs

| Symbol | Per-step domain | Per-sequence shape | Per-batch shape | Description |
|---|---|---|---|---|
| `r_t = (x_t, y_t, z_t)` | `R³` | `(L, 3)` | `(B, L, 3)` | lifted 3D coord from `h_t^refined` |
| `(δx_t, δy_t, δz_t)` | `R³` | `(L, 3)` | `(B, L, 3)` | LSTM^refine deltas |
| `(x_t, y_t, z_t)^final` | `R³` | `(L, 3)` | `(B, L, 3)` | final prediction `r_t + δ_t` |
| `(x_t, y_t, z_t)^gt` | `R³` | `(L, 3)` | `(B, L, 3)` | ground truth |

### Loss-related quantities

| Symbol | Domain | Shape | Description |
|---|---|---|---|
| `L_ε` | `R` | scalar | EoT BCE loss (Eq. 4) |
| `L_3D` | `R` | scalar | 3D L2 loss (Eq. 5) |
| `L_B` | `R` | scalar | below-ground penalty (Eq. 6) |
| `L_Total` | `R` | scalar | weighted total (Eq. 7) |
| `λ_ε, λ_3D, λ_B` | `R` | scalars | loss weights = (10, 1, 10) |
| `Y` | subset of `R` | `(\|Y\|,)`, `\|Y\| ≤ N·B` | predicted y-coords with `y < 0` |

### Per-network input/output tensor shapes (full pipeline)

| Network | Input shape | Output shape | Per-step input dim | Per-step output dim |
|---|---|---|---|---|
| `LSTM^ε` (Table 5) | `(B, L, 4)` = `ΔP_t` | `(B, L, 1)` = `ε_t` | 4 | 1 |
| `LSTM^f` (Table 6) | `(B, L, 6)` = `(ΔP_t, ε_t, h_t^f)` | `(B, L, 1)` = `Δh_t^f` | 6 = 4+1+1 | 1 |
| `LSTM^b` (Table 6) | `(B, L, 6)` = `(ΔP_t, ε_t, h_t^b)` | `(B, L, 1)` = `Δh_t^b` | 6 = 4+1+1 | 1 |
| `LSTM^height` (Table 7) | `(B, L, 5)` = `(h_t, P_t)` | `(B, L, 1)` = `h_t^refined` | 5 = 1+4 | 1 |
| `LSTM^refine` (Table 8) | `(B, L, 7)` = `(r_t, P_t)` | `(B, L, 3)` = `(δx, δy, δz)` | 7 = 3+4 | 3 |
| Auto-regr. (Table 9, IPL only) | `(B, L, 2)` = `(Δu, Δv)` | `(B, L, 2)` = `(Δu_{t+1}, Δv_{t+1})` | 2 | 2 |

### Internal recurrent / FC layer shapes

| Layer kind | Hidden | Output shape | Notes |
|---|---|---|---|
| BiLSTM (in main pipeline) | 64 per direction | `(B, L, 128)` | concat fwd+bwd → `2·64 = 128` |
| Unidirectional LSTM (`LSTM^f`, `LSTM^b`) | 64 | `(B, L, 64)` | one direction only |
| Residual sum | – | matches per-layer shape | output of layer 2 + output of layer 0 |
| FC head intermediate | – | `(B, L, 32)` | three LeakyReLU(0.01) layers |
| FC head final | – | `(B, L, out_dim)` | `out_dim ∈ {1, 3}`; sigmoid only on EoT |

### Hyperparameters (numeric)

| Symbol | Value | Description |
|---|---|---|
| `λ_ε` | 10 | EoT loss weight |
| `λ_3D` | 1 | 3D loss weight |
| `λ_B` | 10 | below-ground loss weight |
| `lr` | 1e-3 | Adam learning rate (constant) |
| `B` (batch) | 256 | training batch size |
| epochs | 1,400 | total training epochs |
| LeakyReLU slope | 0.01 | applied to all FC heads |
| Gaussian noise σ | 0–25 px | augmentation on `(u_t, v_t)` |

---

## §3.1.1  Input parameterization (ray + plane points)

For a 2D pixel `(u, v)` with intrinsics `(f, p_x, p_y)` and extrinsic
`E ∈ SE(3) ⊂ R^{4×4}`, the corresponding viewing ray
`r(s) = c + s·d` has

> **Eq. 1**  `c = ψ( E⁻¹ · [0, 0, 0, 1]ᵀ )`
>
> **Eq. 2**  `d = ψ( E⁻¹ · [u − p_x,  v − p_y,  f,  0]ᵀ )`

where `ψ : R⁴ → R³` is the dehomogenize op `ψ([x y z w]ᵀ) = [x y z]`.

The ray is then re-parameterized as its two intersections:

- `p_ground` = ray ∩ plane `{y = 0}`  (ground)
- `p_vertical` = ray ∩ plane `{z = 0}`  (a chosen vertical plane, e.g.
  coplanar with the tennis net)

Because `p_ground.y ≡ 0` and `p_vertical.z ≡ 0`, those coords are
dropped, leaving the canonical input

> `P = (p_ground, p_vertical) ∈ R⁴`
>
> ordered as  `(p_ground.x, p_ground.z, p_vertical.x, p_vertical.y)`.

---

## §3.1.2  End-of-trajectory (EoT) network — LSTM^ε

Input is the temporal difference

> `ΔP_t = P_{t+1} − P_t`         (zero-padded at the last step)

This relative form is invariant to absolute shifts in plane points:
a sequence starting at `P_1` and one starting at `P_1 + (a, b)` give
identical `ΔP_t`.

Output is the EoT probability `ε_t ∈ [0, 1]` — high at the time step
just before a force is applied (a hit) or the ball comes to rest.

Architecture: stack of **3 BiLSTMs with shortcut (residual)
connections** (inspired by [51]). The last hidden state goes through
**3 fully-connected layers** to produce `ε_t`.

---

## §3.1.3  Height prediction network — LSTM^f, LSTM^b, LSTM^height

Two unidirectional LSTMs predict per-step height *deltas*:

- **`LSTM^f` (forward)** takes `(ΔP_t, ε_t, h_t^f)` and predicts
  `Δh_t^f`. The forward height is accumulated:

  > `h_t^f = h_{t−1}^f + Δh_{t−1}^f`,    `h_0^f = 0`

- **`LSTM^b` (backward)** mirrors this, starting from `h_N^b = 0` and
  accumulating in reverse.

The two are combined with a **ramp-weighted sum** that trusts the
forward stream near the start and the backward stream near the end:

> **Eq. 3**  `h_t = (1 − w_t) · h_t^f + w_t · h_t^b`,
>
> where `w_t = (t − 1) / (N − 1)`.

This reduces long-aggregation drift.

Then the **bidirectional LSTM^height** refines the absolute heights
using the plane points as additional context:

> input  `(h_t, P_t) ∈ R⁵`   →   output  `h_t^refined ∈ R`

LSTM^height has the same architecture as LSTM^ε (3 BiLSTMs with shortcut
connections, then 3 FC layers).  It re-injects awareness of *absolute*
position that the relative-only forward/backward streams lose.

---

## §3.1.4  Refinement network — LSTM^refine

The refined height is lifted to a 3D coordinate by intersecting the ray
with the horizontal plane `y = h_t^refined`:

> Find `s_t*` such that  `r_t^y(s_t*) = h_t^refined`, then
>
> `r_t(s_t*) = (x_t, y_t, z_t)`.

Closed form using the plane points:
`t = h / p_vertical.y`,
`x = p_ground.x + t·(p_vertical.x − p_ground.x)`,
`y = h`,
`z = p_ground.z · (1 − t)`.

Because real 2D tracks are noisy, exact projection-consistency would
propagate that noise into 3D. The **refinement network (LSTM^refine)**
predicts small corrections:

> input  `(x_t, y_t, z_t, P_t) ∈ R⁷`   →   output  `(δx_t, δy_t, δz_t)`
>
> **Final**  `(x_t, y_t, z_t)_final = (x_t + δx_t,  y_t + δy_t,  z_t + δz_t)`

Predicting *deltas* (ResNet-style, [17]) makes the head start near
identity and focus on refinement. Architecture: stack of **3 BiLSTMs**,
mirroring LSTM^ε and LSTM^height.

---

## §3.2  Loss functions

### EoT loss (weighted BCE)

> **Eq. 4**
> `L_ε = − (1/N) · Σ_{t=1..N} [ γ · ε_t^gt · log ε_t  +  (1 − γ) · (1 − ε_t^gt) · log(1 − ε_t) ]`

where `ε_t^gt ∈ {0, 1}` and `γ` balances the two classes (positives
are rare).

### 3D reconstruction loss (L2)

> **Eq. 5**
> `L_3D = (1/N) · Σ_{t=1..N} ‖ (x_t, y_t, z_t)^gt − (x_t, y_t, z_t)^final ‖₂²`

### Below-ground penalty

> **Eq. 6**
> `L_B = (1/|Y|) · Σ_{y ∈ Y} y²`,
>
> where `Y = { y_t^final  |  y_t^final < 0 }` is the set of predicted
> y-coords that fell below the ground plane.

### Total loss

> **Eq. 7**
> `L_Total = λ_ε · L_ε  +  λ_3D · L_3D  +  λ_B · L_B`

Per §D, `(λ_ε, λ_3D, λ_B) = (10, 1, 10)`.

---

## §D  Implementation details

- **Optimizer:** Adam [24]
- **Learning rate:** constant `1e-3`
- **Batch size:** 256
- **Epochs:** 1,400
- **BPTT:** all LSTMs trained jointly with backpropagation through time.
- **Augmentation:** Gaussian noise added to each 2D input `(u_t, v_t)`
  to simulate noisy tracking. Different noise levels reported in the
  main paper (clean / ±5 / ±10 / ±15 / ±20 / ±25 px).
- **LeakyReLU slope:** 0.01 throughout.
- All four sub-networks are trained jointly.

### Table 5 — LSTM^ε (EoT prediction) architecture

| Layer | Activation | Output |
|---|---|---|
| Input | – | B × L × 4 |
| BiLSTM.0 | – | B × L × 2 × 64 |
| BiLSTM.1 | – | B × L × 2 × 64 |
| BiLSTM.2  (+ residual from BiLSTM.0) | – | B × L × 2 × 64 |
| Concat | – | B × L × 128 |
| FC.0 | LeakyReLU | B × L × 32 |
| FC.1 | LeakyReLU | B × L × 32 |
| FC.2 | LeakyReLU | B × L × 32 |
| FC.3 | Sigmoid | B × L × 1 |

### Table 6 — LSTM^{f,b} (height delta predictors) architecture

Same architecture for both forward and backward directions.

| Layer | Activation | Output |
|---|---|---|
| Input | – | B × L × 6 |
| LSTM.0 | – | B × L × 1 × 64 |
| LSTM.1 | – | B × L × 1 × 64 |
| LSTM.2 | – | B × L × 1 × 64 |
| Concat | – | B × L × 64 |
| FC.0 | LeakyReLU | B × L × 32 |
| FC.1 | LeakyReLU | B × L × 32 |
| FC.2 | LeakyReLU | B × L × 32 |
| FC.3 | – | B × L × 1 |

### Table 7 — LSTM^height (height refiner) architecture

| Layer | Activation | Output |
|---|---|---|
| Input | – | B × L × 5 |
| BiLSTM.0 | – | B × L × 2 × 64 |
| BiLSTM.1 | – | B × L × 2 × 64 |
| BiLSTM.2  (+ residual(output) from BiLSTM.0) | – | B × L × 2 × 64 |
| Concat | – | B × L × 128 |
| FC.0 | LeakyReLU | B × L × 32 |
| FC.1 | LeakyReLU | B × L × 32 |
| FC.2 | LeakyReLU | B × L × 32 |
| FC.3 | – | B × L × 1 |

### Table 8 — LSTM^refine (3D coord refiner) architecture

| Layer | Activation | Output |
|---|---|---|
| Input | – | B × L × 7 |
| BiLSTM.0 | – | B × L × 2 × 64 |
| BiLSTM.1 | – | B × L × 2 × 64 |
| BiLSTM.2  (+ residual(output) from BiLSTM.0) | – | B × L × 2 × 64 |
| Concat | – | B × L × 128 |
| FC.0 | LeakyReLU | B × L × 32 |
| FC.1 | LeakyReLU | B × L × 32 |
| FC.2 | LeakyReLU | B × L × 32 |
| FC.3 | – | B × L × 3 |

### Table 9 — Auto-regressive interpolator (IPL only, §D.1)

Per §D.1, an auto-regressive LSTM network is used as a **pre-processing
step** to fill in missing 2D track points in the IPL soccer dataset (where
some frames have no detection due to occlusion). It is *not* part of the
main 3D-trajectory pipeline.

- **Input:** temporal difference of 2D pixel coords `(Δu_t, Δv_t)`
- **Output:** predicted difference for the next time step `(Δu_{t+1}, Δv_{t+1})`
- **Setup:** 2 independent directional LSTMs (forward + backward), combined
  with linear ramp weighting analogous to Eq. 3 in the main paper.
- **Training:** teacher forcing, following Williams & Zipser [49].
- **Inference:** if a real track point is available at the current step,
  it is used directly; otherwise the auto-regressive prediction fills in.

Same architecture is used for both forward and backward directions.

| Layer | Activation | Output |
|---|---|---|
| Input | – | B × L × 2 |
| LSTM.0 | – | B × L × 64 |
| LSTM.1 | – | B × L × 64 |
| LSTM.2  (+ residual from LSTM.0) | – | B × L × 64 |
| LSTM.3  (+ residual from LSTM.0 and LSTM.1) | – | B × L × 64 |
| FC.0 | LeakyReLU | B × L × 64 |
| FC.1 | LeakyReLU | B × L × 32 |
| FC.2 | LeakyReLU | B × L × 16 |
| FC.3 | LeakyReLU | B × L × 8 |
| FC.4 | LeakyReLU | B × L × 4 |
| FC.5 | – | B × L × 2 |

### Notation in the tables

- **B** = batch size
- **L** = sequence length
- **2 × 64** = BiLSTM with hidden size 64 per direction (forward+backward)
- **1 × 64** = unidirectional LSTM with hidden size 64
- **Residual**: the third recurrent layer's output is added to the first
  layer's output before the projection head (ResNet-style shortcut).
- **Concat** rows: passthrough of the residual sum into the FC stack
  (the column simply makes the resulting tensor shape explicit;
  there is no tensor-axis concatenation in the BiLSTM stack itself).

---

## Pipeline summary (one forward pass)

```
P_t  =  (p_ground.x, p_ground.z, p_vertical.x, p_vertical.y)        # R^4
ΔP_t =  P_{t+1} − P_t                                               # R^4

ε_t  =  LSTM^ε(ΔP)                                                  # [0,1]
Δh_t^f = LSTM^f(ΔP, ε, h^f),     h_t^f = h_{t−1}^f + Δh_{t−1}^f     # forward
Δh_t^b = LSTM^b(ΔP, ε, h^b),     h_t^b = h_{t+1}^b + Δh_{t+1}^b     # backward
h_t  =  (1 − w_t) h_t^f + w_t h_t^b,   w_t = (t−1)/(N−1)            # Eq. 3

h_t^refined =  LSTM^height(h_t, P_t)                                # bidir
r_t  =  lift_to_3d(h_t^refined, P_t)                                # closed-form
δ_t  =  LSTM^refine(r_t, P_t)                                       # B×L×3

(x,y,z)^final = r_t + δ_t                                           # §3.1.4
```

Joint training minimizes `L_Total` (Eq. 7) over all four sub-networks
simultaneously.
