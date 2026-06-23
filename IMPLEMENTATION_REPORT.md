# Learned Orthonormal-Basis (ONB) Correction — Implementation Report

> **Scope of this document.** It describes the *shared implementation* that this
> commit introduces — the Cayley basis correction `Q` and the ULRSSM model that
> uses it. It is deliberately **not** on `main`: `main` is the plain vendored
> DeepShapeMatchingKit + Learning-ONBs baseline. Every exploratory branch
> descends from this commit and therefore carries this file.
>
> **Per-experiment results are not here.** Each branch has its own `RESULTS.md`
> with what that branch tried, how to reproduce it, and what it found.
> `EXPERIMENTS.md` on `main` maps all of them.

---

## 0. One-paragraph summary

We insert a learned, exactly-orthonormal change of basis **Q** between the raw
LBO eigenbasis and the functional-map solver of **DeepShapeMatchingKit (DSMK)**.
Q is produced by the **rank-R Cayley flow** vendored from **Learning-ONBs**
(`EulerianIsometry`). The first (and currently only) build is **coefficient
space**: Q is a `K×K` rotation acting on spectral coefficients, conditioned on
per-mode DiffusionNet features plus an eigen-position embedding. Because raw LBO
eigenvectors are already mass-orthonormal (`ΦᵀMΦ = I`), the corrected
`Mk = QᵀIQ = I`, so the standard SURFMNet loss and the plain fmap solver stay
valid unchanged. A `bypass` flag makes `Q = I`, reproducing vanilla ULRSSM
bit-for-bit — this is the correctness anchor the whole design rests on.

---

## 1. What came from each repo

### 1.1 DeepShapeMatchingKit — reused **unchanged**

| Role | Class | File |
|------|-------|------|
| Feature extractor | `DiffusionNet_B` (510,336 params; `input_type: wks`) | `networks/diffusion_b_network.py:278` |
| Fmap solver | `FasterRegularizedFMNet` (0 params; closed-form, regularized) | `networks/fmap_network.py:77` |
| Spectral/orth. loss | `SURFMNetLoss` (w_bij, w_orth, w_lap) | `losses/fmap_loss.py:33` |
| Alignment loss | `SquaredFrobeniusLoss` | `losses/fmap_loss.py:8` |
| Sinkhorn / soft-corr head | `Similarity` (`tau=0.07`) | `networks/permutation_network.py:8` |
| p2p from features | `nn_query` | `utils/fmap_util.py:18` |
| fmap → p2p | `fmap2pointmap` | `utils/fmap_util.py:73` |
| Base training model | `ULRSSM_Model` (parent of our model) | `models/ulrssm_model.py` |

None of these files were edited. Our model **subclasses** `ULRSSM_Model` and
**re-registers** via the registry; the kit auto-scans `models/`, so no existing
kit file was touched to wire it in.

### 1.2 Learning-ONBs — vendored into `networks/onb/`

Vendored from `Learning-ONBs/infidictionary/…`. **Only edit = import-path
rewrites** (absolute `infidictionary.*` → package-relative `.`); the math is
untouched. Provenance:

| Vendored file (`networks/onb/…`) | Origin (`infidictionary/…`) | Provides |
|---|---|---|
| `eulerian.py` (411 ln) | `neural_isometries/eulerian.py` | `EulerianIsometry` — rank-R Cayley integrator (`pushforward`/`pullback`/`shuffle_model_state`, gradient checkpointing) |
| `isometry_base.py` (81 ln) | `neural_isometries/base.py` | `NeuralIsometry`, `IdentityIsometry` base classes |
| `fields/base.py` | `networks/base.py` | `TimeEvolvingField`, `MLPNeuralField`, `RMSNorm`, `_build_mlp`, `_init_orthogonal` |
| `fields/time_embedding.py` | `networks/time_embedding.py` | `SinusoidalTimeEmbedding` (used for both time- and mode-index embedding) |
| `fields/nerf_spatiotemporal.py` | `networks/nerf_spatiotemporal.py` | `NerfSpatioTemporalField` |
| `fields/latent_bilinear.py` | `networks/latent_bilinear.py` | `LatentBilinearSpatiotemporalField` |
| `fields/ff_neural_field.py`, `ntk_mlp_neural_field.py` | `networks/…` | alt. field generators (not used in coeff-space build) |
| `dictionaries/base.py`, `fourier.py`, `__init__.py` | `dictionaries/…` | `FourierDictionary` (atom-selector path of the isometry) |
| `domain_samplers.py` (85 ln) | `domain_samplers.py` | sampling helpers (unused in coeff space) |
| `utils.py` (45 ln) | `utils.py` | `pairwise_inner_product`, `norm2`, `parallel_inner_product` (measure-weighted inner products) |

`networks/onb/__init__.py` re-exports the public API and documents that the only
change is the import rewrite.

---

## 2. What is new

### 2.1 Written from scratch

| File | Lines | What |
|---|---|---|
| `networks/onb/correction.py` | **203** | The wrapper that turns the Cayley flow into a coefficient-space basis correction. |
| `models/onb_ulrssm_model.py` | **204** | `ONB_ULRSSM_Model(ULRSSM_Model)` — overrides `feed_data` (correction + rebuild), `validate_single`, `validation`. |
| `networks/onb/selftest_correction.py` | 128 | Standalone math anchors (orthonormality, bypass, convention). |
| `networks/onb/selftest_model_wiring.py` | 140 | Real-`feed_data` Q≠I basis-consistency anchor. |
| `networks/onb/selftest_validate.py` | 133 | Eval-determinism + bypass==vanilla anchor. |

### 2.2 Configs added

`options/ulrssm/train/faust_onb.yaml` and `faust_onb_bypass.yaml`, vendored and
edited from `faust.yaml`. See §6 for the exact diff. (SMAL configs and the Q→I
regularizer configs are added by later branches — see `EXPERIMENTS.md`.)

### 2.3 Key classes / functions and **actual signatures**

`networks/onb/correction.py`:

```python
@NETWORK_REGISTRY.register()
class CayleyONBCorrection(nn.Module):
    def __init__(self, feature_dim: int, rank: int = 16, num_steps: int = 20,
                 bypass: bool = False, base_acceleration: float = 1.0,
                 index_emb_freqs: int = 6, use_eigenvalue: bool = True,
                 hidden_dims: tuple = (128, 128), n_time_freqs: int = 16,
                 gradient_checkpointing: bool = False, eps: float = 1e-8): ...

    def _build_coords(self, evecs, mass, evals, feats):    # -> (K, coords_dim)
    def compute_rotation(self, evecs, mass, evals, feats): # -> Q (K, K), QᵀQ=I
    def _correct_one(self, evecs, mass, evals, feats):     # -> Φ̃ = Φ Q, (N, K)
    def forward(self, evecs, mass, evals, feats):          # batched or unbatched

class _SpectralConditionField(TimeEvolvingField):
    def __init__(self, coords_dim, output_dim, rank, time_emb_dim,
                 hidden_dims=(128, 128)): ...
    def forward(self, t_emb, x) -> torch.Tensor:           # -> (N, R, C)
```

Vendored core driven by the wrapper:

```python
class EulerianIsometry(NeuralIsometry):
    def __init__(self, coords_dim, channels_dim, rank, base_acceleration,
                 scalar_field_partial=None, gradient_checkpointing=False,
                 n_time_freqs=16, kr_hidden_dims=(64,64),
                 atom_shuffling_K=None, atom_selector_hidden_dims=(64,64),
                 use_power_law=False): ...
    def shuffle_model_state(self, num_steps=None)   # train: sorted-rand tspan; eval: linspace
    def pushforward(self, src_coords, src_logabsdet, src_field,
                    start_time, end_time, method="cayley")  # -> (coords, logabsdet, tgt_field)
```

**Parameter counts (live, from self-test/logs):** `CayleyONBCorrection`
bypass=True → **0 params**; bypass=False → **79,872 params**. Sits beside
`DiffusionNet_B` (510,336).

---

## 3. Data flow — one training step (`ONB_ULRSSM_Model.feed_data`)

Real variable names and shapes (batch `B=1`, `K = num_evecs = 200`, feature
channels `C = 256`):

```
data['first'] / data['second']  (to_device)
  ├ evecs        Φ_x   [B, Nx, K]   float32   (mass-orthonormal LBO basis)
  ├ mass         m_x   [B, Nx]      float32
  ├ evals        λ_x   [B, K]       float32
  └ (faces/verts/operators for aux losses)

feat_x = feature_extractor(data=data_x)         # [B, Nx, 256]   DiffusionNet/WKS

# ── correction (one SHARED onb instance applied to BOTH shapes) ──
evecs_x = onb(evecs_x, mass_x, evals_x, feat_x) # Φ̃_x [B, Nx, K]
  internally, per shape:
    coords = [ rms_norm(ΦᵀM·feat) , sin_emb(i/K) , λ_i/max|λ| ]   # (K, 256+13)
    logabsdet = 0 (K,)                          # uniform measure  ⇒  QᵀQ = I
    field     = I_K  (B=K, N=K, C=1)            # transport the identity basis
    Q = isometry.pushforward(coords, logabsdet, field, 0, 1) → (K, K)
    Φ̃ = Φ @ Q

# ── rebuild the two derived operators from the corrected basis ──
evecs_trans_x = evecs_x.transpose(1,2) * mass_x.unsqueeze(1)   # Φ̃ᵀ diag(m)  [B, K, Nx]
self.Mk_x     = bmm(evecs_trans_x, evecs_x)                    # Φ̃ᵀMΦ̃ ≈ I    [B, K, K]

# ── unchanged downstream ──
Cxy, Cyx = fmap_net(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
loss     = surfmnet_loss(Cxy, Cyx, evals_x, evals_y)          # l_bij + l_orth (w_lap=0)
Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
Cxy_est  = bmm(evecs_trans_y, bmm(Pyx, evecs_x))
loss['l_align'] = align_loss(Cxy, Cxy_est) (+ symmetric term)
```

The **only** tensors that change versus vanilla are `evecs_x/y` (now `Φ̃`) and the
**recomputed** `evecs_trans_x/y` and `Mk_x/y`. Everything fed to `fmap_net`,
`surfmnet_loss`, and `align_loss` is the unmodified ULRSSM call.

---

## 4. The exact edits vs the parent

The parent `ULRSSM_Model.feed_data` reads `evecs_trans` **straight from the data
dict**:

```python
# parent (models/ulrssm_model.py)
evecs_trans_x = data_x['evecs_trans']   # [B, K, Nx]   ← precomputed, raw basis
evecs_trans_y = data_y['evecs_trans']
```

Our override replaces those two lines with the correction + rebuild block
(`models/onb_ulrssm_model.py:51-63`):

```python
onb = self.networks['onb_correction']
evecs_x = onb(evecs_x, mass_x, evals_x, feat_x)               # NEW
evecs_y = onb(evecs_y, mass_y, evals_y, feat_y)               # NEW
evecs_trans_x = evecs_x.transpose(1,2) * mass_x.unsqueeze(1)  # rebuilt, not read
evecs_trans_y = evecs_y.transpose(1,2) * mass_y.unsqueeze(1)
self.Mk_x = torch.bmm(evecs_trans_x, evecs_x)                 # NEW (sanity/future)
self.Mk_y = torch.bmm(evecs_trans_y, evecs_y)
```

The rest of `feed_data` (fmap solve, SURFMNet, l_align, optional Dirichlet) is
**byte-identical** to the parent except it now operates on `Φ̃`.

`validate_single` is a near-mirror of the parent's, with three additions:
1. applies the **same shared** correction before the basis is used (else a
   trained model would be scored on the uncorrected basis);
2. uses `.squeeze()`'d unbatched tensors exactly like the parent;
3. accumulates two diagnostics into `self._diag_q` / `self._diag_orth`:
   `max|Q − I|` (is the correction active?) and `max|Φ̃ᵀMΦ̃ − I|` (still
   orthonormal?). `Q` is recovered as `Q = (Φ_rawᵀ M) Φ̃`.

`validation` (new override) resets the two accumulators, calls
`super().validation(...)`, then logs `[ONB diag] …` to the logger / TensorBoard /
wandb. In `.eval()` the isometry's `shuffle_model_state` takes the deterministic
`linspace` branch, so **test-time Q is fixed**, not resampled.

---

## 5. Anchor tests — actual numbers

Run CPU-only (see the caveat at the end of this section):

```bash
cd DeepShapeMatchingKit
CUDA_VISIBLE_DEVICES="" conda run -n deepshapematchingkit python networks/onb/selftest_correction.py
CUDA_VISIBLE_DEVICES="" conda run -n deepshapematchingkit python networks/onb/selftest_model_wiring.py
CUDA_VISIBLE_DEVICES="" conda run -n deepshapematchingkit python networks/onb/selftest_validate.py
```

`selftest_correction.py` (float64; passes):

| Check | One-line assertion | Printed residual |
|---|---|---|
| Input basis really mass-orthonormal | `max\|ΦᵀMΦ − I\|` | `6.661e-16` |
| (1) bypass=True returns input bit-for-bit | `torch.equal(out_bypass, Phi)` | `True` |
| (2) corrected basis mass-orthonormal | `max\|Φ̃ᵀMΦ̃ − I\|` | `6.661e-16` |
| (2) same instance, 2nd shape | `max\|Φ̃ᵀMΦ̃ − I\|` | `8.882e-16` |
| (3) zeroed generator ⇒ Q=I through full path | `max\|Q − I\|` | `0.000e+00` |
| (3) full path with Q=I == bypass, bit-for-bit | `torch.equal(out_id, out_bypass)` | `True` |
| (4) Q orthonormal | `max\|QᵀQ − I\|` | `6.661e-16` |
| (4) Q non-trivial (test not vacuous) | `max\|Q − I\|` | `2.908e-02` |
| (4) convention Φ̃=ΦQ ⇒ Φ̃ᵀM = Qᵀ·ΦᵀM | `max\|Φ̃ᵀM − Qᵀ·ΦᵀM\|` | `1.110e-16` |
| (4) the *wrong* convention (Q·ΦᵀM) | `max\|Φ̃ᵀM − Q·ΦᵀM\|` | `2.640e-02` |

`selftest_model_wiring.py` (real `feed_data`, GT-correspondent synthetic pair; passes on CPU):

| Check | Printed residual |
|---|---|
| Q non-trivial `max\|Q − I\|` | `6.280e-02` |
| `Mk_x` sanity `max\|Mk_x − I\|` | `4.768e-07` |
| `Cxy_est` under GT corr `max\|Cxy_est − I\|` | `4.768e-07` (consistent bases) |
| `Cyx_est` under GT corr `max\|Cyx_est − I\|` | `4.768e-07` |
| mismatch demo (1 raw projector) `max\|Cxy_bad − I\|` | `6.280e-02` (proves the test bites) |
| param counts | bypass=0, active=79,872 |

`selftest_validate.py` (passes on CPU):

| Check | Result |
|---|---|
| isometry in eval under `model.eval()` | `False` (= not training) |
| eval tspan == `linspace(0,1,L)` | `True` |
| Q deterministic across calls | `True` |
| bypass `validate_single` == vanilla p2p | `True` (bit-for-bit) |
| `max\|Cxy_vanilla − Cxy_onb\|` | `0.000e+00` |
| `max\|Pyx_vanilla − Pyx_onb\|` | `0.000e+00` |

> **Environment caveat (a real surprise):** `selftest_model_wiring` and
> `selftest_validate` **fail on the 2-GPU workstation** — `num_gpu: auto` wraps
> the model in `DataParallel` (so `model.networks['onb_correction'].bypass`
> raises `AttributeError`), and on a single visible GPU the synthetic CPU test
> tensors trip a device mismatch. They must be run **CPU-only**
> (`CUDA_VISIBLE_DEVICES=""`). The numbers above are from that CPU run.

---

## 6. Config specifics — `faust_onb.yaml` vs `faust.yaml`

Identical except:

| Field | `faust.yaml` (vanilla) | `faust_onb.yaml` |
|---|---|---|
| `type` | `ULRSSM_Model` | `ONB_ULRSSM_Model` |
| `networks.onb_correction` | *(absent)* | `type: CayleyONBCorrection`, `feature_dim: 256`, `rank: 16`, `num_steps: 20` (=L), `bypass: false` |
| `train.optims.onb_correction` | *(absent)* | `Adam`, `lr: 1.0e-3` |
| `train.schedulers.onb_correction` | *(absent)* | `CosineAnnealingLR`, `eta_min: 1.0e-4`, `T_max: 15` |

Unchanged and worth quoting: `num_evecs: 200`, `total_epochs: 15`,
`batch_size: 1`, feature extractor `DiffusionNet_B(in=128, out=256, input_type=wks)`,
`fmap_net: FasterRegularizedFMNet (bidirectional)`, `permutation: Similarity (tau=0.07)`,
loss `SURFMNetLoss(w_bij=1, w_orth=1, w_lap=0.0)` + `SquaredFrobeniusLoss(1.0)`.

`faust_onb_bypass.yaml` = `faust_onb.yaml` with **`bypass: true`** (the Q=I
ablation).

> **The `w_lap: 0.0` choice is load-bearing**, not incidental: the SURFMNet
> Laplacian-commutativity term is only well-defined on the LBO eigenbasis, so it
> is disabled off-basis. (Documented in `PLAN.md` and the config comment.)

---

## 7. Implementation notes not obvious from the plan

- **Self-tests are CPU-only on this box** (see §5 caveat) — `num_gpu: auto` +
  `DataParallel` breaks attribute access and device placement.
- **`shuffle_model_state` is re-seeded every forward** to resample the ODE
  time-span in training; the constructor seeds it once so the kit's per-network
  `.train()/.eval()` toggling (which calls it with no args) is safe. Any code
  that needs a *stable* Q — diagnostics, hybrid losses with two terms — must
  either call `.eval()` or compute Q once and reuse it.
- **Normalization is on the conditioning path only.** The RMS-norm of
  `ΦᵀM·feat` feeds the generator's coords, but the *transported field* (identity)
  and the *measure* (uniform `logabsdet=0`) never see it — this is what keeps the
  Q = I bypass anchor exact.
- **Convention is asserted, not assumed:** the self-test explicitly checks
  `Φ̃ = ΦQ ⇒ Φ̃ᵀM = Qᵀ·ΦᵀM` (`1e-16`) *and* that the swapped convention fails
  (`2.6e-2`), so a future refactor can't silently flip it.
- **`Mk` is computed but currently unused downstream** (it's ≈I by construction).
  It's kept as a sanity check and a hook for the deferred **vertex-space** build
  where `Mk ≠ I` and `HS_SURFMNetLoss` / `ExpandedResolventFMNet` would be needed
  (`PLAN.md` §"FIRST BUILD").
- **Vendored breadth > usage:** `domain_samplers.py`, the `dictionaries/` Fourier
  atom-selector path, and several alt. fields (`ff_neural_field`,
  `ntk_mlp_neural_field`, `nerf_spatiotemporal`, `latent_bilinear`) were vendored
  for completeness but the coefficient-space build only exercises
  `_SpectralConditionField` (MLP) + `SinusoidalTimeEmbedding`.
- **Training outputs are git-ignored** (`experiments/`, `wandb/`). Per-branch
  `logs/` and `ckpts/` are force-added as the evidence trail for `RESULTS.md`.
