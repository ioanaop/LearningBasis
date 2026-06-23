"""Coefficient-space learned orthonormal-basis correction (Cayley flow).

`CayleyONBCorrection` wraps the vendored `EulerianIsometry` and produces a
corrected eigenbasis Φ̃ = Φ Q, where Q is a K×K orthonormal rotation obtained
by flowing the identity I_K through the rank-R Cayley integrator.

Coefficient-space design (see PLAN.md):
  * The flow's "sample points" are the K modes (N = K), NOT the mesh vertices.
  * The measure is uniform (logabsdet = 0). Because a uniform measure makes the
    flow preserve the plain Euclidean inner product on the K coefficients, Q is
    Euclidean-orthonormal (QᵀQ = I). The raw basis already satisfies ΦᵀMΦ = I,
    so the corrected Mk = Qᵀ(ΦᵀMΦ)Q = QᵀQ = I exactly. Mass therefore enters
    ONLY through evecs_trans = ΦᵀM (used to project features and to rebuild the
    projector downstream); it never touches the flow.
  * The generator U is conditioned on a per-mode signal:
        coords[i] = concat( rms_norm(ΦᵀM·feat)[i] , eigpos[i] )
    where the first part is the intrinsic feature of mode i and
        eigpos[i] = [ sinusoidal_embedding(i / K) , evals[i] / max(evals) ]
    is an eigen-position signal so U knows where mode i sits in the spectrum.
    Normalization lives on the conditioning path ONLY — the transported field
    (identity) and the measure (uniform) never see it, preserving the exact
    Q = I bypass anchor.

A `bypass` flag returns evecs unchanged (bit-for-bit): this is the Q = I
ablation that must reproduce vanilla ULRSSM.
"""

from functools import partial

import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY

from .eulerian import EulerianIsometry
from .fields.base import TimeEvolvingField, _build_mlp
from .fields.time_embedding import SinusoidalTimeEmbedding


class _SpectralConditionField(TimeEvolvingField):
    """MLP generator field U(t, x): (N, emb) x (N, d) -> (N, R, C).

    A plain MLP on [t_emb, coords] — no random Fourier features. The kit's
    conditioning coords are already rich learned descriptors (projected
    DiffusionNet features + eigen-position), so the spectral-bias lifting that
    NeRF Fourier features provide for raw low-dim positions is unnecessary here.
    """

    def __init__(self, coords_dim, output_dim, rank, time_emb_dim,
                 hidden_dims=(128, 128)):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.R = rank
        self.C = output_dim
        self.mlp = _build_mlp(
            time_emb_dim + coords_dim,
            rank * output_dim,
            hidden_dims,
            nn.SiLU,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=False,
        )

    def forward(self, t_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        h = torch.cat([t_emb, x], dim=-1)
        return self.mlp(h).view(N, self.R, self.C)


@NETWORK_REGISTRY.register()
class CayleyONBCorrection(nn.Module):
    """Learned coefficient-space orthonormal basis correction Φ̃ = Φ Q.

    Args:
        feature_dim: channel count of the intrinsic (DiffusionNet) features.
        rank: rank R of the Cayley generator.
        num_steps: number of time samples L for the Euler integration
            (=> L-1 Cayley steps); T is fixed to [0, 1].
        bypass: if True, forward returns evecs unchanged (Q = I ablation).
        base_acceleration: global scale on the flow generator.
        index_emb_freqs: sinusoidal frequencies for the mode-index embedding.
        use_eigenvalue: append normalized eigenvalue as an eigen-position channel.
        hidden_dims: hidden widths of the generator MLP.
        n_time_freqs: sinusoidal time-embedding frequencies inside the isometry.
        gradient_checkpointing: checkpoint the unrolled Cayley steps (train only).
        eps: numerical floor for normalizations.
    """

    def __init__(
        self,
        feature_dim: int,
        rank: int = 16,
        num_steps: int = 20,
        bypass: bool = False,
        base_acceleration: float = 1.0,
        index_emb_freqs: int = 6,
        use_eigenvalue: bool = True,
        hidden_dims: tuple = (128, 128),
        n_time_freqs: int = 16,
        gradient_checkpointing: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.num_steps = num_steps
        self.bypass = bypass
        self.use_eigenvalue = use_eigenvalue
        self.eps = eps

        # Eigen-position embedding: sinusoidal(index/K) (+ optional eigenvalue).
        self.index_embedding = SinusoidalTimeEmbedding(index_emb_freqs)
        eigpos_dim = self.index_embedding.out_dim + (1 if use_eigenvalue else 0)
        coords_dim = feature_dim + eigpos_dim
        self.coords_dim = coords_dim

        # The flow is only built when not bypassing — keeps the Q=I ablation a
        # parameter-free passthrough.
        if not bypass:
            self.isometry = EulerianIsometry(
                coords_dim=coords_dim,
                channels_dim=1,
                rank=rank,
                base_acceleration=base_acceleration,
                scalar_field_partial=partial(_SpectralConditionField,
                                             hidden_dims=hidden_dims),
                gradient_checkpointing=gradient_checkpointing,
                n_time_freqs=n_time_freqs,
            )
            # Seed _num_steps so the isometry's train()/eval() override (which
            # calls shuffle_model_state() with no args) is safe; base_model
            # toggles .train()/.eval() on every network. We re-shuffle each
            # forward anyway to resample the ODE time-span.
            self.isometry.shuffle_model_state(num_steps=num_steps)
        else:
            self.isometry = None

    def _build_coords(self, evecs, mass, evals, feats):
        """Per-mode conditioning coords. evecs (N,K), mass (N,), evals (K,),
        feats (N,Cf) -> coords (K, coords_dim)."""
        K = evecs.shape[1]
        # A = ΦᵀM·feat  (per-mode intrinsic feature), shape (K, Cf).
        evecs_trans = evecs.t() * mass[None]          # (K, N)
        A = evecs_trans @ feats                       # (K, Cf)
        # rms-normalize each mode over the feature dim (conditioning path ONLY).
        A = A / (A.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps)

        # eigen-position: sinusoidal embedding of normalized mode index ...
        idx = torch.arange(K, device=evecs.device, dtype=evecs.dtype) / max(K, 1)
        eigpos = self.index_embedding(idx)            # (K, 2*index_emb_freqs)
        if self.use_eigenvalue:
            eval_norm = evals / (evals.abs().max() + self.eps)  # (K,)
            eigpos = torch.cat([eigpos, eval_norm[:, None]], dim=-1)

        return torch.cat([A, eigpos], dim=-1)         # (K, coords_dim)

    def compute_rotation(self, evecs, mass, evals, feats):
        """Learned K×K orthonormal rotation Q (QᵀQ = I) for a single shape.

        Obtained by flowing the identity I_K through the Cayley integrator,
        conditioned on this shape's per-mode features. Returns I_K when bypassing.
        evecs (N,K), mass (N,), evals (K,), feats (N,Cf) -> Q (K,K).
        """
        K = evecs.shape[1]
        if self.bypass:
            return torch.eye(K, device=evecs.device, dtype=evecs.dtype)

        coords = self._build_coords(evecs, mass, evals, feats)   # (K, coords_dim)
        logabsdet = torch.zeros(K, device=evecs.device, dtype=evecs.dtype)  # uniform measure
        # Transport the identity basis: field f_j = e_j, shape (B=K, N=K, C=1).
        field = torch.eye(K, device=evecs.device, dtype=evecs.dtype).unsqueeze(-1)

        self.isometry.shuffle_model_state(num_steps=self.num_steps)
        _, _, tgt = self.isometry.pushforward(coords, logabsdet, field, 0.0, 1.0)
        # tgt[j, n, 0] = Op[n, j]  (the orthonormal operator's column j), so
        # tgt.squeeze = Opᵀ and Q = Op = tgt.squeezeᵀ.
        return tgt.squeeze(-1).transpose(0, 1)        # (K, K), QᵀQ = I

    def _correct_one(self, evecs, mass, evals, feats):
        """Single-shape correction. evecs (N,K), mass (N,), evals (K,),
        feats (N,Cf) -> Φ̃ (N,K). Convention: Φ̃ = Φ Q, hence
        Φ̃ᵀ diag(mass) = Qᵀ (Φᵀ diag(mass)) and Φ̃ᵀ diag(mass) Φ̃ = QᵀQ = I."""
        Q = self.compute_rotation(evecs, mass, evals, feats)
        return evecs @ Q                              # Φ̃ = Φ Q, (N, K)

    def forward(self, evecs, mass, evals, feats):
        """Return corrected basis Φ̃.

        Accepts batched (B, N, K)/(B, N)/(B, K)/(B, N, Cf) or unbatched
        (N, K)/(N,)/(K,)/(N, Cf). Output matches the input batching of evecs.
        """
        # Bypass: exact passthrough (Q = I ablation), bit-for-bit.
        if self.bypass:
            return evecs

        batched = evecs.dim() == 3
        if not batched:
            return self._correct_one(evecs, mass, evals, feats)

        outs = []
        for b in range(evecs.shape[0]):
            outs.append(self._correct_one(evecs[b], mass[b], evals[b], feats[b]))
        return torch.stack(outs, dim=0)
