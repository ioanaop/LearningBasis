# Learning Orthonormal Bases for Functional-Map Shape Matching

Can a **learned, exactly-orthonormal change of basis** `Q`, inserted between the
LBO eigenbasis and a functional-map solver, improve shape correspondence?

- **`main` is a clean baseline.** It contains only the two vendored upstream
  repos — [DeepShapeMatchingKit](DeepShapeMatchingKit/) and
  [Learning-ONBs](Learning-ONBs/) — plus [PLAN.md](PLAN.md). No experimental code.
- **Every experiment lives on its own `exp/…` branch.** Nothing is merged into
  `main`; the branches are deliberately kept as a record of what was tried.

Each branch carries two documents:

| File | What it is |
|---|---|
| `IMPLEMENTATION_REPORT.md` | The shared `Q` implementation — provenance, data flow, exact edits vs. the upstream model, self-test numbers. Same on every branch. |
| `RESULTS.md` | That branch's experiment: what it tried, exact commands to reproduce, the numbers, and the conclusion. |

Training logs, TensorBoard events and final checkpoints are committed alongside,
so every number below is verifiable without re-running anything.

---

## Branch map

All branches share one base commit that adds the `Q` implementation, kept off
`main` so `main` stays a pristine baseline:

```
main  ── init ── combined repos                      (vendored baseline only)
          └── Add learned ONB basis-correction implementation
                ├── exp/joint-smal-configs
                │     └── exp/joint-features-and-basis
                └── exp/frozen-features-basis-only
                      ├── exp/frozen-features-rotnet
                      └── exp/frozen-features-fmap-loss
```

The split is between **two ways of posing the problem**:

### A. Joint — learn features *and* the rotation together
Unsupervised (SURFMNet losses), descriptor and `Q` trained end-to-end. This is
the real target setting.

| Branch | What it adds | Headline result |
|---|---|---|
| `exp/joint-smal-configs` | SMAL configs; FAUST/SMAL/brain baseline runs | Learned `Q` **never beat** the `Q=I` bypass; on SMAL-xyz it regressed **0.075 → 0.431** geodesic error |
| `exp/joint-features-and-basis` | Q→identity regularizer + weight sweep | Weight 50 recovers **0.431 → 0.093**, but still worse than just `Q=I` (0.075) |

### B. Frozen features — learn *only* the rotation
Descriptors fixed (WKS/xyz), supervised directly by ground-truth correspondence,
no fmap solver in the loss. Built to measure the ceiling without the descriptor
confound.

| Branch | What it adds | Headline result |
|---|---|---|
| `exp/frozen-features-basis-only` | GT-supervised harness + closed-form SVD oracle | Oracle is **~30× better than raw** (0.0195 vs 0.577) — but the trained net only reaches 0.393 |
| `exp/frozen-features-rotnet` | Alternative `expm(A−Aᵀ)` rotation net | Ties `raw` (0.532 vs 0.531) → parametrization is *not* the bottleneck |
| `exp/frozen-features-fmap-loss` | Fmap-space + hybrid losses; downstream metric probe | Fmap loss is worse than nothing (0.664); **the gauge finding** (below) |

---

## What was learned

**1. The idea is sound — the basis rotation is worth a lot.**
The closed-form SVD oracle, which computes the optimal `Q` from ground truth,
improves FAUST geodesic error from **0.577 to 0.0195** (~30×), and ~15× on SMAL.
A per-shape orthonormal change of basis genuinely can turn a near-useless
spectral embedding into an almost-exact correspondence.

**2. Nothing learned ever got close to it.**
Across two datasets, two feature types, two rotation parametrizations, three
loss formulations and a regularizer sweep, **no configuration beat simply
setting `Q = I`.** The implementation is verified correct at every level
(self-tests at `1e-16`, live orthonormality at `1e-7`, and the bypass reproduces
vanilla ULRSSM bit-for-bit), so these are findings about the approach, not bugs.

**3. The gauge finding — the likely explanation.**
A functional-map solver **absorbs** an orthonormal `Q`: the solved map transforms
as `C = Q_yᵀ C_raw Q_x`, and `fmap2pointmap` is invariant to orthonormal changes
of basis. Worse, the resolvent regularizer actively *penalizes* `Q ≠ I`. So
downstream, **`Q = I` is already optimal** — an orthonormal per-shape rotation is
a gauge transformation the pipeline is built to ignore. This reframes every
negative result above: the learned `Q` was never able to help the solver, only to
hurt it.

**4. A second, structural suspicion.**
The oracle's rotations come from an SVD of `B_xᵀB_y` — a quantity depending on
**both** shapes. A network that sees one shape's features at a time may be unable
to represent it in principle. Consistent with this, two very different rotation
parametrizations failed identically.

---

## Where to continue

The most promising thread is on `exp/frozen-features-fmap-loss`: if the solver
absorbs orthonormal transforms, then the one thing it *cannot* undo is a
**non-orthonormal** one. `basis_opt/downstream.py` implements exactly that — a
shared, population-level, non-orthonormal metric `D` giving `Φ̃ = Φ Q D`,
initialized to the identity so training begins at the classical-fmap baseline,
trained with a contrastive loss whose negatives prevent `D → 0` collapse.

It is **implemented and wired to the CLI but not yet evaluated** — no training
log was captured. Running it is the natural next step.

---

## Setup

```bash
conda activate deepshapematchingkit
export CUDA_VISIBLE_DEVICES=1        # pin the free GPU

git checkout exp/<branch>
cat RESULTS.md                       # exact commands for that experiment
```

Datasets are reached through the `data/` symlink. Per-branch `RESULTS.md` gives
the precise command for every run whose numbers it reports.

> **Note on the self-tests:** they must run **CPU-only**
> (`CUDA_VISIBLE_DEVICES=""`). On the 2-GPU workstation, `num_gpu: auto` wraps
> the model in `DataParallel`, which breaks attribute access into the correction
> network. Details in `IMPLEMENTATION_REPORT.md` §5.
