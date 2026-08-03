# Results — joint learning, baseline runs (FAUST / SMAL / brain)

**Branch:** `exp/joint-smal-configs`
**Question:** does learning the descriptor *and* the basis rotation `Q` jointly,
unsupervised, beat vanilla ULRSSM?

This branch adds the SMAL configs on top of the shared implementation (which
already carries the FAUST ones). It holds the results of **every joint run that
predates the Q→I regularizer** — FAUST, SMAL, and brain. The regularizer and its
sweep are on `exp/joint-features-and-basis`.

See `IMPLEMENTATION_REPORT.md` for what the model actually does.

---

## What this branch adds

| Config | dataset | feature `input_type` | `bypass` |
|---|---|---|---|
| `smal_onb.yaml` | SMAL | xyz | no (learned Q) |
| `smal_onb_bypass.yaml` | SMAL | xyz | **yes** (Q=I) |
| `smal_onb_wks.yaml` | SMAL | wks | no (learned Q) |

Plus `.gitignore` hygiene for training artifacts.

The `bypass` configs are the floor to beat: with `Q = I` the model is vanilla
ULRSSM running through the ONB code path, so any difference is attributable to
the learned basis and nothing else.

---

## How to reproduce

```bash
cd DeepShapeMatchingKit
conda activate deepshapematchingkit          # env on kjp211fc
export CUDA_VISIBLE_DEVICES=1                # pin the free GPU

# SMAL, xyz features
python train.py --opt options/ulrssm/train/smal_onb.yaml           # learned Q
python train.py --opt options/ulrssm/train/smal_onb_bypass.yaml    # Q=I floor
python train.py --opt options/ulrssm/train/smal_onb_wks.yaml       # wks features
python train.py --opt options/ulrssm/train/smal.yaml               # true vanilla

# FAUST (configs come from the shared base commit)
python train.py --opt options/ulrssm/train/faust_onb.yaml
python train.py --opt options/ulrssm/train/faust_onb_bypass.yaml
```

Console logs land in `logs/`; per-run checkpoints, TensorBoard events and the
validation log land in `experiments/ulrssm/<name>/`. Both are committed here
(final weights only — intermediate resume checkpoints are not).

Watch the `[ONB diag]` lines: `max|Q − I|` says whether the correction is
actually active, `max|Φ̃ᵀMΦ̃ − I|` says whether it is still orthonormal.

---

## Results

### FAUST (`num_evecs=200`, 15 epochs)

| Run | best val err ↓ | best AUC ↑ | final | `max\|Q−I\|` | `max\|Φ̃ᵀMΦ̃−I\|` |
|---|---|---|---|---|---|
| `faust_onb_bypass` (= vanilla ULRSSM) | **0.0160** | **0.9226** | 0.0163 | ~3.0e-7 (constant) | ~3.0e-7 |
| `faust_onb` (learned Q) | 0.0163 | 0.9211 | 0.0164 | 0.143 → **0.506** | ~7e-7 |

**Read:** the correction trains (Q drifts from ≈I to `max|Q−I|≈0.5`) and stays
perfectly orthonormal throughout (`~1e-7`), but it is **neutral-to-slightly-worse**
than the bypass. No accuracy gain from the joint learned basis on FAUST.

### SMAL (`num_evecs=200`, 20 epochs, `val_freq=2000`)

| Run | feature | best err ↓ | best AUC ↑ | final `max\|Q−I\|` | note |
|---|---|---|---|---|---|
| `smal` (pure `ULRSSM_Model`) | xyz | **0.0697** | **0.7810** | — | true vanilla baseline |
| `smal_onb_bypass` (Q=I) | xyz | 0.0748 | 0.7682 | ~3.4e-7 | ONB-path bypass ≈ vanilla |
| `smal_onb` (learned Q) | xyz | 0.4307 | 0.2615 | ~0.11 | **catastrophic** |
| `smal_onb_wks` (learned Q) | wks | 0.2684 | 0.5534 | ~0.15–0.20 | ≈ wks floor (no help) |

**Read — the core finding of this branch:**

1. **The unregularized learned basis is catastrophic on xyz**: 0.431 vs 0.075 for
   the bypass, a ~6× regression. Orthonormality stays exact (`~1e-7`), so the
   mechanism is not broken — the damage is *where* Q rotates. Q lowers the
   SURFMNet loss while **destroying** correspondence: the unsupervised proxy and
   the actual goal have come apart. This is what motivated the Q→I regularizer on
   the next branch.
2. **`input_type` dominates the basis correction.** xyz bypass 0.075 vs wks
   bypass 0.262 (~3.5×). On non-isometric SMAL, wks is simply the wrong
   descriptor floor, and the basis story is second-order to the descriptor choice.
3. **On wks the rotation never helped** either (learned 0.268 ≈ bypass 0.262) —
   there the ceiling is the features, not the basis.

### Brain (`mindboggle` 1k, `num_evecs=64`, 3 epochs)

`brain_onb` logged **losses only** — this dataset has no GT geodesic distances,
so there is no accuracy metric. `l_total` settles ≈ 2.0e2 (`l_bij≈80, l_orth≈75,
l_align≈42`); `max|Q−I|` grows 0.09 → ~0.40, orthonormality ~6e-7. Useful only as
evidence the pipeline runs on a third, very different dataset.

---

## Conclusion

Across FAUST, SMAL and brain, jointly learning the basis alongside the descriptor
**never beat the `Q = I` bypass**, and on SMAL-xyz it was far worse. The
orthonormality machinery is provably correct (self-tests at `1e-16`, live
diagnostics at `1e-7`), so this is a statement about the *objective*, not the
implementation: an unconstrained basis rotation can reduce the unsupervised loss
while making matching worse.

Two follow-ups came out of this:
- constrain Q toward identity → `exp/joint-features-and-basis`
- remove the descriptor problem entirely and supervise Q directly from GT, to
  measure what a basis rotation could achieve at best → `exp/frozen-features-basis-only`
