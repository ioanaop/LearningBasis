"""Downstream functional-map matching for the basis_opt probe.

Re-aims the learned basis to SERVE a DSMK functional-map solver instead of BEING
the matcher itself. Corrected basis:

    Phi~ = Phi @ Q @ D
      Q : per-shape, orthonormal, from CayleyONBCorrection (unchanged mechanism).
      D : ONE shared, population-level, NON-orthonormal K x K metric (same matrix
          for every shape/pair). Init = identity, so training starts exactly at
          the plain classical-fmap baseline.

Downstream matching:
    C   = FasterRegularizedFMNet(descriptors projected on Phi~)   # the SAME
          closed-form regularized solve the joint ONB model uses (imported,
          not reimplemented).
    p2p = fmap2pointmap(C, Phi~_x, Phi~_y)

Why D (and not Q) is the load-bearing part here: a gauge probe showed the solver
ABSORBS an orthonormal Q (C = Q_y^T C_raw Q_x, and fmap2pointmap is orthonormal-
invariant) and its resolvent regularizer PENALIZES any non-identity Q -- so Q=I
is already optimal downstream. A non-orthonormal D is the one transform the
solver cannot undo: it survives as a learned metric on the spectral matching,
`|| (Phi_x C^T[i] - Phi_y[j]) Q_y D ||`.

Training signal (differentiable; geo-error itself is not): a CONTRASTIVE
(InfoNCE) loss on the fmap-transported embeddings at GT-corresponding points --
corresponding points pulled together, non-corresponding pushed apart. The
contrastive negatives are what stop D from trivially collapsing (D -> 0), which
a plain `||transported - target||` alignment loss would allow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.fmap_network import FasterRegularizedFMNet
from utils.fmap_util import fmap2pointmap


class SharedMetric(nn.Module):
    """One shared (population-level), NON-orthonormal K x K reweighting D.

    The SAME matrix for every shape and every pair -- not per-shape, not
    per-pair. Initialized to identity so an untrained run reproduces the plain
    classical functional-map baseline exactly.

    kind:
        'none'    -> identity, no parameters (baseline / Q-only ablation).
        'diag'    -> D = diag(exp(log_d)), K params. A learned per-mode
                     reweighting (down/up-weight spectral modes). exp keeps it
                     positive; init log_d=0 -> D=I. (Global scale is a harmless
                     gauge: it cancels in C = D C_raw D^{-1}.)
        'lowrank' -> D = I + U V^T, rank-r off-diagonal mixing, 2*K*r params.
                     init U=V=0 -> D=I.
    """

    def __init__(self, k, kind='none', rank=8):
        super().__init__()
        self.k, self.kind = k, kind
        if kind == 'diag':
            self.log_d = nn.Parameter(torch.zeros(k))
        elif kind == 'lowrank':
            self.U = nn.Parameter(torch.zeros(k, rank))
            self.V = nn.Parameter(torch.zeros(k, rank))
        elif kind != 'none':
            raise ValueError(f'unknown shared-metric kind {kind!r}')

    def matrix(self, device, dtype):
        eye = torch.eye(self.k, device=device, dtype=dtype)
        if self.kind == 'none':
            return eye
        if self.kind == 'diag':
            return torch.diag(torch.exp(self.log_d))
        return eye + self.U @ self.V.t()

    def forward(self, phi):
        """phi (N,K) -> phi @ D (N,K)."""
        if self.kind == 'none':
            return phi
        return phi @ self.matrix(phi.device, phi.dtype)

    def cond_number(self):
        """Condition number of D (1.0 when D=I); a scalar diagnostic."""
        if self.kind == 'none':
            return 1.0
        D = self.matrix(self.log_d.device if self.kind == 'diag' else self.U.device,
                        torch.float32)
        s = torch.linalg.svdvals(D)
        return (s.max() / s.min().clamp_min(1e-12)).item()


# A single stateless solver instance (no learnable params -- just lmbda/gamma).
# lmbda=100 matches DSMK's FasterRegularizedFMNet default / the joint model.
_SOLVER = FasterRegularizedFMNet(lmbda=100, resolvant_gamma=0.5)


def solve_C(evx_c, evy_c, mass_x, mass_y, wks_x, wks_y, evals_x, evals_y):
    """Regularized descriptor functional map Cxy (X->Y) in the corrected basis.

    Uses the DSMK FasterRegularizedFMNet solve on the frozen descriptor (wks)
    projected onto the corrected basis Phi~. Differentiable in Phi~ (hence in
    Q and D). evals are the ORIGINAL LBO eigenvalues (the resolvent mask).
    """
    etx = (evx_c.t() * mass_x[None])[None]          # [1, K, Nx]
    ety = (evy_c.t() * mass_y[None])[None]          # [1, K, Ny]
    Cxy, _ = _SOLVER(wks_x[None], wks_y[None],
                     evals_x[None], evals_y[None], etx, ety)
    return Cxy[0]                                    # (K, K)


def downstream_p2p(evx_c, evy_c, mass_x, mass_y, wks_x, wks_y, evals_x, evals_y):
    """Y->X point map from the downstream fmap pipeline (for eval / geo-error)."""
    C = solve_C(evx_c, evy_c, mass_x, mass_y, wks_x, wks_y, evals_x, evals_y)
    return fmap2pointmap(C, evx_c, evy_c)


def downstream_contrastive_loss(evx_c, evy_c, mass_x, mass_y, wks_x, wks_y,
                                evals_x, evals_y, corr_x, corr_y,
                                n_samples=512, tau=0.1):
    """InfoNCE on the fmap-transported embeddings at GT-corresponding points.

    Zx = Phi~_x @ C^T  (transport X into Y's space, exactly as fmap2pointmap
    does), Zy = Phi~_y. For sampled GT pairs (corr_x[i], corr_y[i]), Zx[i] must
    be the nearest Zy among the sampled targets (and symmetric). Differentiable;
    the negatives prevent the metric D from collapsing.
    """
    C = solve_C(evx_c, evy_c, mass_x, mass_y, wks_x, wks_y, evals_x, evals_y)
    Zx = evx_c @ C.t()                              # (Nx, K)
    Zy = evy_c                                      # (Ny, K)

    P = corr_x.shape[0]
    if n_samples < P:
        idx = torch.randperm(P, device=evx_c.device)[:n_samples]
        cx, cy = corr_x[idx], corr_y[idx]
    else:
        cx, cy = corr_x, corr_y

    px, py = Zx[cx], Zy[cy]                          # (S, K) each
    logits = -(torch.cdist(px, py) ** 2) / tau       # (S, S)
    target = torch.arange(px.shape[0], device=px.device)
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.t(), target))
