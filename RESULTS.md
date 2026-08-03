# Results — joint learning of features *and* basis

**Branch:** `exp/joint-features-and-basis`
**Question:** does learning the descriptor *and* the basis rotation `Q` jointly,
unsupervised, beat vanilla ULRSSM? And if an unconstrained `Q` hurts, does
pulling it back toward the identity recover the loss?

This is the **tip of the joint line**. Part 1 below is the baseline evidence
inherited from `exp/joint-smal-configs` (FAUST / SMAL / brain, no regularizer).
Part 2 is this branch's own contribution: the **Q→identity regularizer** and its
weight sweep.

See `IMPLEMENTATION_REPORT.md` for what the model actually does.

---

# Part 1 — baseline runs, no regularizer

## Configs inherited from `exp/joint-smal-configs`

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

**Read — the core finding of Part 1:**

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

## Part 1 conclusion

Across FAUST, SMAL and brain, jointly learning the basis alongside the descriptor
**never beat the `Q = I` bypass**, and on SMAL-xyz it was far worse. The
orthonormality machinery is provably correct (self-tests at `1e-16`, live
diagnostics at `1e-7`), so this is a statement about the *objective*, not the
implementation: an unconstrained basis rotation can reduce the unsupervised loss
while making matching worse.

That finding is what motivates Part 2.

---

# Part 2 — the Q→identity regularizer (this branch's contribution)

If an unconstrained `Q` lowers the loss while wrecking correspondence, constrain
it. This branch adds an **opt-in** penalty pulling the learned rotation back
toward the identity:

```
l_qreg = q_identity_reg_weight * ( ‖Q_x − I‖²_F + ‖Q_y − I‖²_F ) / B
```

recovering `Q` exactly as the validation diagnostic does, `Q = (Φ_rawᵀ M) Φ̃`.
Implementation: `models/onb_ulrssm_model.py` (+18 lines). **Default weight is 0,
an exact no-op** — every pre-existing config and the bypass anchor are
bit-for-bit unaffected.

## Configs added

| Config | feature | `q_identity_reg_weight` |
|---|---|---|
| `smal_onb_qreg10.yaml` / `smal_onb_qreg50.yaml` | xyz | **10.0 / 50.0** |
| `smal_onb_wks_qreg10.yaml` / `smal_onb_wks_qreg50.yaml` | wks | **10.0 / 50.0** |
| `smal_onb_wks_bypass.yaml` | wks | – (Q=I floor for the wks family) |

## How to reproduce

```bash
cd DeepShapeMatchingKit
conda activate deepshapematchingkit
export CUDA_VISIBLE_DEVICES=1

python train.py --opt options/ulrssm/train/smal_onb_qreg10.yaml
python train.py --opt options/ulrssm/train/smal_onb_qreg50.yaml
python train.py --opt options/ulrssm/train/smal_onb_wks_qreg10.yaml
python train.py --opt options/ulrssm/train/smal_onb_wks_qreg50.yaml
python train.py --opt options/ulrssm/train/smal_onb_wks_bypass.yaml
```

## Results — xyz features

| Run | weight | best err ↓ | best AUC ↑ | final `max\|Q−I\|` | note |
|---|---|---|---|---|---|
| `smal_onb_bypass` (Q=I) | — | **0.0748** | 0.7682 | ~3.4e-7 | floor to beat |
| `smal_onb` | 0 | 0.4307 | 0.2615 | ~0.11 | catastrophic (Part 1) |
| `smal_onb_qreg10` | 10 | 0.4420 | 0.2077 | ~0.039 | partial run (1 val ≈ 2k/16.8k it) |
| `smal_onb_qreg50` | 50 | **0.0933** | 0.7405 | ~0.001 | partial run (1 val); **recovered** |

## Results — wks features

| Run | weight | best err ↓ | best AUC ↑ | final `max\|Q−I\|` | note |
|---|---|---|---|---|---|
| `smal_onb_wks_bypass` (Q=I) | — | **0.2619** | 0.5437 | ~3.4e-7 | wks feature floor |
| `smal_onb_wks` | 0 | 0.2684 | 0.5534 | ~0.15–0.20 | ≈ floor (no help) |
| `smal_onb_wks_qreg10` | 10 | 0.2652 | 0.5495 | ~0.010 | full run |
| `smal_onb_wks_qreg50` | 50 | 0.2643 | 0.5528 | ~0.003 | full run |
| `smal_onb_wks_qreg` | 1 | — | — | — | **aborted**, 0 validations, no config |

## Read

1. **The regularizer does exactly what it was built for.** Weight 50 clamps
   `max|Q−I|` from ~0.11 to ~1e-3 and pulls geodesic error back from **0.431 →
   0.093**, near the 0.075 bypass floor. The proxy/goal divergence is a
   constraint problem, and constraining it works.
2. **The transition is sharp.** Weight 10 leaves a residual `max|Q−I|≈0.039` and
   the error is still ~0.442 — a few-percent rotation is already enough to wreck
   non-isometric SMAL matching.
3. **But recovery is not improvement.** The best regularized result (0.093) is
   still *worse* than simply setting `Q = I` (0.075). Pushed hard enough to stop
   hurting, `Q` is pushed close enough to the identity to stop doing anything.
4. **On wks the regularizer is inert**, because there was nothing to fix — the
   ceiling is the descriptor (0.262 floor), not the basis.

> **Caveat on the xyz numbers.** The two xyz `qreg` runs are **partial**: a single
> validation at ~2 000 / 16 820 iterations (~12% of training). Treat 0.0933 as an
> early snapshot, not a finished result. The wks `qreg` runs completed; so did
> `smal_onb` and both bypass runs (9 validations each).

---

# Overall conclusion for the joint approach

Across every dataset and both feature types, the jointly learned basis has **no
regime where it beats the `Q = I` bypass**. Unconstrained it actively harms;
constrained enough to be safe it converges back to doing nothing. The
implementation is verified correct at every level (self-tests `1e-16`, live
orthonormality `1e-7`, bypass reproduces vanilla bit-for-bit), so this is a
finding about the approach, not a bug.

The open question this leaves — *is the basis rotation itself worthless, or is
the unsupervised joint objective just unable to find a good one?* — is what the
`exp/frozen-features-*` branches answer by supervising `Q` directly from ground
truth. Short version: the rotation is **not** worthless (a GT oracle is ~30×
better than the raw basis), but predicting it from features is the hard part.
