# Direct basis-transformation optimization

**Branch:** `optimization-basis`

## Idea

The main pipeline (`ONB_ULRSSM_Model`) learns **two** things jointly and
unsupervised: a feature descriptor (DiffusionNet) *and* an orthonormal basis
correction `Q` (the Cayley flow), coupled through a functional-map solver and
SURFMNet losses. That couples two hard problems and differentiates through the
whole stack.

This experiment isolates the **basis** half in the simplest possible setting:

- **Fix the descriptors** — use WKS (or SHOT, or later a frozen pretrained net).
  Nothing is learned about features.
- **Use ground-truth correspondences** on a (near-)isometric dataset (FAUST).
- **Learn only the orthonormal change of basis `Q`**, supervised *directly* by
  the GT correspondence. No functional-map solver, no unsupervised losses, and we
  only differentiate through the rotation — none of the matching machinery.

### Objective: align bases so the GT functional map becomes the identity

For a pair `(X, Y)` with eigenbases `Φ_x, Φ_y` (mass-orthonormal) and GT
corresponding vertex pairs `corr_x[i] ↔ corr_y[i]`, learn orthonormal `Q_x, Q_y`:

```
    L(Q_x, Q_y) = mean ‖ Φ_x[corr_x] Q_x  −  Φ_y[corr_y] Q_y ‖²
```

i.e. corrected eigenfunctions should take **equal values at corresponding
points**. When that holds, the GT functional map in the corrected basis,
`C = lstsq(Φ̃_y[corr_y], Φ̃_x[corr_x])`, is the identity — hence
"align bases → GT fmap ≈ I".

> **Why not "match the descriptor-fmap to the GT-fmap"?** That objective is
> *invariant* to the rotation (both maps rotate the same way under `Q`), so it
> would optimize nothing unless combined with basis truncation. The alignment
> form above is the non-degenerate version.

### Closed-form oracle (free, exact)

With `B_x = Φ_x[corr_x]`, `B_y = Φ_y[corr_y]`, the loss equals
`const − 2·tr(Q_xᵀ M Q_y)` with `M = B_xᵀ B_y`, because `‖B Q‖ = ‖B‖` for
orthonormal `Q`. So the global optimum is the **SVD** `M = U S Vᵀ ⇒ Q_x* = U,
Q_y* = V`. We use this as an oracle to validate that gradient descent and the
WKS network actually reach the best achievable alignment. (Free rotations are
restricted to `SO(K)` via `expm`; the SVD oracle is full `O(K)`, so a reflection
gap of ≤ one sign is possible — negligible in practice.)

### Evaluation

Once the bases are aligned, their rows are aligned spectral coordinates, so the
correspondence is recovered by **plain nearest-neighbour between `Φ̃_x` and
`Φ̃_y` rows** — the basis itself is the only descriptor, no fmap solve. We report
the mean **geodesic error** (kit's `calculate_geodesic_error`) of that map.
`raw` (`Q = I`) is the baseline to beat.

## Staging

- **Stage 0 — `run_stage0.py`**: one pair, one *free* rotation per shape
  (`FreeRotation = expm(A−Aᵀ)`), no network, no descriptors. Sanity check that
  aligning the basis helps and that gradient descent matches the SVD oracle.
- **Stage 1 — `run_stage1.py`**: a *shared* `CayleyONBCorrection` network
  predicts `Q` from fixed **WKS** features (+ eigen-position), trained over FAUST
  train pairs, evaluated on held-out test pairs vs `raw` and the per-pair oracle.
- **Stage 2 (hook, not yet wired)**: replace WKS with a **frozen pretrained**
  DiffusionNet (DeepShapeMatching) and optimize only `Q`. Drop-in: swap how
  `feats` is produced in `dataio.py` / `run_stage1.py`; the objective is
  identical. This tells us what to expect before *jointly* optimizing descriptors
  and basis.

## Running

Use the `deepshapematchingkit` conda env and pin the free GPU. Run from the
`DeepShapeMatchingKit` directory (configs/data assume `../data`).

```bash
cd DeepShapeMatchingKit

# Stage 0: single-pair sanity check
CUDA_VISIBLE_DEVICES=1 conda run -n deepshapematchingkit \
    python basis_opt/run_stage0.py --pair 0 1 --k 100

# Stage 1: WKS-conditioned network over FAUST
CUDA_VISIBLE_DEVICES=1 conda run -n deepshapematchingkit \
    python basis_opt/run_stage1.py --k 100 --steps 2000
```

Key flags: `--k` (# eigenbasis modes), `--rank`/`--num_steps` (Cayley flow),
`--steps`/`--lr`, `--pair`, `--phase`.

## Files

| file | what |
|------|------|
| `core.py` | objective, `FreeRotation`, closed-form oracle, p2p recovery, geo error |
| `dataio.py` | load FAUST eigenbasis / mass / WKS / GT corr (reuses kit cache) |
| `run_stage0.py` | Stage 0 single-pair free-rotation experiment |
| `run_stage1.py` | Stage 1 WKS-conditioned `CayleyONBCorrection` training |

## What to expect / how to read the output

- `oracle` should beat `raw` clearly — that confirms the *idea* (a per-shape
  basis rotation derived from GT alignment improves matching). If it does not,
  the basis-alignment premise itself is weak for this data/`k`.
- `learned` (Stage 0) should ≈ `oracle`.
- Stage 1 `net` should move from ≈`raw` toward `oracle` as training proceeds, and
  generalize to held-out test shapes from WKS alone. The gap `net` vs `oracle`
  is the price of predicting `Q` from features instead of GT — exactly the signal
  we want before adding descriptor learning back in.
