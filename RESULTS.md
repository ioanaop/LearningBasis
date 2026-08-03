# Results — frozen features, learn only the basis rotation

**Branch:** `exp/frozen-features-basis-only`
**Question:** the joint experiment showed a learned basis never beats `Q = I`.
Is the basis rotation *itself* worthless, or is the unsupervised joint objective
just unable to find a good one?

To answer that, this branch removes every confound: **descriptors are frozen**
(WKS or xyz, nothing learned about features), the supervision is the **ground-truth
correspondence**, and there is **no functional-map solver and no unsupervised loss**
in the objective. Only `Q` is learned. See `basis_opt/README.md` for the full
design rationale and `IMPLEMENTATION_REPORT.md` for the shared `Q` machinery.

---

## The objective

For a pair `(X, Y)` with mass-orthonormal eigenbases `Φ_x, Φ_y` and GT
corresponding vertices `corr_x[i] ↔ corr_y[i]`, learn orthonormal `Q_x, Q_y`:

```
L(Q_x, Q_y) = mean ‖ Φ_x[corr_x] Q_x  −  Φ_y[corr_y] Q_y ‖²
```

Corrected eigenfunctions should take equal values at corresponding points — when
they do, the GT functional map in the corrected basis is the identity.

**There is a closed-form optimum.** With `B_x = Φ_x[corr_x]`, `B_y = Φ_y[corr_y]`,
the loss equals `const − 2·tr(Q_xᵀ M Q_y)` for `M = B_xᵀ B_y`, because `‖B Q‖ = ‖B‖`
for orthonormal `Q`. So the global minimum is the SVD `M = U S Vᵀ ⇒ Q_x* = U,
Q_y* = V`. This **oracle** is free to compute and bounds what any method here can
achieve — it is the single most useful number on this branch.

**Evaluation** is deliberately harsh: once bases are aligned their rows are
aligned spectral coordinates, so correspondence is recovered by plain
nearest-neighbour between `Φ̃_x` and `Φ̃_y` rows. The basis is the *only*
descriptor; no fmap solve. `raw` (`Q = I`) is the baseline to beat.

---

## How to reproduce

```bash
cd DeepShapeMatchingKit
conda activate deepshapematchingkit
export CUDA_VISIBLE_DEVICES=1

# Stage 0 -- one pair, one free rotation per shape, no network, no descriptors.
# Sanity check that aligning the basis helps and that GD matches the SVD oracle.
python basis_opt/run_stage0.py --pair 0 1 --k 100

# Stage 1 -- shared CayleyONBCorrection predicts Q from frozen features.
# FAUST / WKS (the headline run, log: basis_opt/logs/stage1_k60.log)
python basis_opt/run_stage1.py --dataset faust --feature wks \
    --k 60 --rank 32 --steps 3000 --eval_every 250

# SMAL probes
python basis_opt/run_stage1.py --dataset smal --feature wks --k 60 --rank 32 --steps 800
python basis_opt/run_stage1.py --dataset smal --feature wks --k 60 --rank 64 --steps 700
python basis_opt/run_stage1.py --dataset smal --feature xyz --k 60 --rank 64 --steps 1500

# Comparison table across k for all methods (raw / classical / net / oracle)
python basis_opt/run_eval.py --dataset faust --k_list 20,40,60,100 --rank 32
```

Logs are committed under `basis_opt/logs/`, trained networks under
`basis_opt/ckpts/`.

---

## Results

### Stage 1 — FAUST, WKS features, k=60, rank 32, 3000 steps

Held-out test geodesic error of the NN-in-basis correspondence
(`basis_opt/logs/stage1_k60.log`):

| Method | geo error ↓ |
|---|---|
| `raw` (Q = I) | 0.5770 |
| `oracle` (closed-form SVD Q from GT) | **0.0195** |
| `net` (WKS → CayleyONBCorrection) @ step 0 / 1500 / 2500 / 3000 | 0.4644 → 0.4425 → 0.4320 → **0.3933** |

`align_loss` falls 1.88 → 1.67. Checkpoint: `ckpts/wks_onb_k60_r32.pth`.

**This is the key result of the whole project, and it cuts both ways:**

- **The premise is sound.** The oracle basis rotation is **~30× better than raw**
  (0.0195 vs 0.5770). A per-shape orthonormal change of basis genuinely can turn
  a near-useless spectral embedding into an almost-exact correspondence. The
  joint experiment's failure was *not* because basis rotation is a bad idea.
- **Predicting `Q` from features is the hard part.** The trained network closes
  only a fraction of the gap (0.577 → 0.393) and is still 20× away from the
  oracle it is chasing. It improves steadily and does not diverge — it is just
  slow and far short.

### Stage 1 — SMAL probes

| Run | feature | rank | raw | oracle | net (best) | log |
|---|---|---|---|---|---|---|
| `smal_k60_probe` | wks | 32 | 0.5313 | 0.0352 | 0.5725 | `logs/smal_k60_probe.log` |
| `smal_accel` | wks | 64 | 0.5191 | 0.0351 | 0.5820 | `logs/smal_accel.log` |
| `smal_xyz_k60` | xyz | 64 | 0.5313 | 0.0352 | 0.5757 | `logs/smal_xyz_k60.log` |

**Read:** on SMAL the oracle gap is even more dramatic (~15× better than raw),
but the network **never beats `raw`** on any of the three probes — not with more
rank, not with xyz features instead of wks. The align loss does decrease in every
run (e.g. 2.35 → 2.09), so it is optimizing something; it just is not the thing
that improves matching.

---

## Conclusion

The oracle proves a per-shape basis rotation is worth ~15–30× in geodesic error.
No configuration of `CayleyONBCorrection` conditioned on frozen per-shape
features gets close to it, and on SMAL it does not beat doing nothing at all.

The structural suspicion this raises — and the reason for the follow-up branches:
**the needed rotation is a *pairwise* quantity**. The oracle's `Q_x = U` and
`Q_y = V` come from an SVD of `B_xᵀB_y`, which depends on *both* shapes. A network
that sees one shape's features at a time may simply be unable to represent it,
no matter how it is trained.

Two follow-ups test that:
- **`exp/frozen-features-rotnet`** — is the *parametrization* the bottleneck?
  Swap the Cayley flow for a simpler, more expressive `expm(A − Aᵀ)` rotation net.
- **`exp/frozen-features-fmap-loss`** — is the *objective* the bottleneck?
  Re-express the same GT supervision in functional-map space, and try a hybrid.

> **Known gap.** The intended 8 000-step convergence runs (`logs/conv_*.log`)
> never produced results — they were launched while the branch was checked out
> without `basis_opt/` present and died immediately on a missing file. The
> longest completed run is the 3 000-step FAUST one above. Whether the net keeps
> closing the gap past 3 000 steps is untested.
