"""Conditioned-rotation basis correction (matrix-exp parametrization).

A simpler, more expressive alternative to CayleyONBCorrection for *standalone
supervised* basis alignment. Stage 0 showed a free K x K rotation
Q = expm(A - A^T) fits the alignment objective exactly (align -> ~0.03), while the
Cayley flow -- built for the differentiable joint pipeline -- could not fit it at
all with WKS or xyz. Here we keep the expm parametrization that works and simply
make the skew generator A a learned *function* of the shape's fixed per-mode
features, so Q generalizes across shapes while staying exactly orthonormal:

    coords[i] = concat( rms_norm(Phi^T M feat)[i] , eigpos[i] )          # (K, cond)
    E_p, E_q  = MLP(coords).split                                        # (K, d) each
    A         = scale * (E_p E_q^T - E_q E_p^T)                          # (K, K) skew
    Q         = expm(A)                                                  # orthonormal
    Phi~      = Phi Q

The bilinear E_p E_q^T builds a full K x K (rank up to d) data-dependent matrix
from per-mode embeddings; skew-symmetrizing + expm makes it a rotation. With
small MLP weights A ~ 0 so Q ~ I -- the same identity anchor as the Cayley
bypass, so training starts from "no correction".

Drop-in: forward(evecs, mass, evals, feats) -> Phi~, same signature as
CayleyONBCorrection, so basis_opt/run_stage1.py uses it unchanged.
"""

import torch
import torch.nn as nn

from networks.onb.fields.base import _build_mlp
from networks.onb.fields.time_embedding import SinusoidalTimeEmbedding


class ConditionedRotation(nn.Module):
    def __init__(self, feature_dim, emb_dim=128, index_emb_freqs=6,
                 use_eigenvalue=True, hidden_dims=(256, 256), init_scale=1.0,
                 eps=1e-8):
        super().__init__()
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.use_eigenvalue = use_eigenvalue
        self.eps = eps

        self.index_embedding = SinusoidalTimeEmbedding(index_emb_freqs)
        eigpos_dim = self.index_embedding.out_dim + (1 if use_eigenvalue else 0)
        cond_dim = feature_dim + eigpos_dim
        self.mlp = _build_mlp(cond_dim, 2 * emb_dim, hidden_dims, nn.SiLU,
                              use_batchnorm=False, use_rmsnorm=True, bias=False)
        # learnable global rotation magnitude; starts at init_scale.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def _build_coords(self, evecs, mass, evals, feats):
        """Per-mode conditioning (same recipe as CayleyONBCorrection). (K, cond)."""
        K = evecs.shape[1]
        evecs_trans = evecs.t() * mass[None]                 # (K, N)
        A = evecs_trans @ feats                              # (K, Cf): per-mode feature
        A = A / (A.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps)
        idx = torch.arange(K, device=evecs.device, dtype=evecs.dtype) / max(K, 1)
        eigpos = self.index_embedding(idx)                   # (K, 2*freqs)
        if self.use_eigenvalue:
            eval_norm = evals / (evals.abs().max() + self.eps)
            eigpos = torch.cat([eigpos, eval_norm[:, None]], dim=-1)
        return torch.cat([A, eigpos], dim=-1)                # (K, cond_dim)

    def compute_rotation(self, evecs, mass, evals, feats):
        """Learned K x K orthonormal Q from this shape's per-mode features."""
        coords = self._build_coords(evecs, mass, evals, feats)   # (K, cond)
        h = self.mlp(coords)                                     # (K, 2*emb)
        Ep, Eq = h[:, :self.emb_dim], h[:, self.emb_dim:]
        raw = Ep @ Eq.t()                                       # (K, K)
        Askew = self.scale * (raw - raw.t())                   # (K, K) skew-symmetric
        return torch.matrix_exp(Askew)                         # (K, K), Q^T Q = I

    def _correct_one(self, evecs, mass, evals, feats):
        return evecs @ self.compute_rotation(evecs, mass, evals, feats)

    def forward(self, evecs, mass, evals, feats):
        """Phi~ = Phi Q. Accepts batched (B,N,K)/... or unbatched (N,K)/..."""
        if evecs.dim() == 3:
            return torch.stack(
                [self._correct_one(evecs[b], mass[b], evals[b], feats[b])
                 for b in range(evecs.shape[0])], dim=0)
        return self._correct_one(evecs, mass, evals, feats)
