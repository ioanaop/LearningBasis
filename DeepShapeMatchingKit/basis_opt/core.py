"""Core math for direct basis-transformation optimization.

All functions operate on a single shape pair (X, Y), unbatched.

Notation
--------
Phi_x  (Nx, K)   LBO eigenvectors of X, mass-orthonormal: Phi_x^T diag(mass) Phi_x = I.
Phi_y  (Ny, K)   same for Y.
corr_x (P,)      indices into X's vertices,  GT-corresponding to ...
corr_y (P,)      indices into Y's vertices.  (corr_x[i] <-> corr_y[i].)
Q_x,Q_y (K, K)   orthonormal change-of-basis.  Corrected basis  Phi~ = Phi Q.

Objective (basis alignment / GT-fmap -> identity)
-------------------------------------------------
    L(Q_x, Q_y) = || Phi_x[corr_x] Q_x  -  Phi_y[corr_y] Q_y ||_F^2  / (P*K)

Writing B_x = Phi_x[corr_x], B_y = Phi_y[corr_y] (P, K) and using that Q_x, Q_y
are orthonormal so ||B_x Q_x||_F = ||B_x||_F (rotation-invariant), the loss is

    const  -  2 * tr(Q_x^T (B_x^T B_y) Q_y).

So minimizing L  ==  maximizing tr(Q_x^T M Q_y) with M = B_x^T B_y, whose optimum
over orthonormal pairs is the SVD of M:  M = U S V^T  =>  Q_x* = U, Q_y* = V.
That closed form is the ORACLE we validate gradient descent / the WKS net against.

When Phi~_x[corr_x] == Phi~_y[corr_y], the GT functional map in the corrected
basis, C = lstsq(Phi~_y[corr_y], Phi~_x[corr_x]), equals the identity -- hence
"align bases -> GT fmap ~= I".
"""

import numpy as np
import torch
import torch.nn as nn

from utils.fmap_util import nn_query, regularized_fmap_solve
from metrics.geodist_metric import calculate_geodesic_error
from utils.tensor_util import to_numpy


# --------------------------------------------------------------------------- #
# Orthonormal parametrization for the single-pair "free rotation" (Stage 0)
# --------------------------------------------------------------------------- #
class FreeRotation(nn.Module):
    """A learnable orthonormal K x K matrix Q = expm(A - A^T).

    The skew-symmetric parametrization keeps Q exactly in SO(K) for every value
    of the free parameter, and initializing the parameter at 0 gives Q = I -- the
    same identity anchor the CayleyONBCorrection bypass uses, so optimization
    starts from "no correction".
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.zeros(k, k))

    def forward(self) -> torch.Tensor:
        a = self.weight - self.weight.t()        # skew-symmetric
        return torch.matrix_exp(a)               # in SO(K), exactly orthonormal


# --------------------------------------------------------------------------- #
# Objective + closed-form oracle
# --------------------------------------------------------------------------- #
def alignment_loss(Bx_c: torch.Tensor, By_c: torch.Tensor) -> torch.Tensor:
    """Mean-squared mismatch of corrected basis values at corresponding points.

    Bx_c = Phi_x[corr_x] @ Q_x  (P, K),   By_c = Phi_y[corr_y] @ Q_y  (P, K).
    """
    return (Bx_c - By_c).pow(2).mean()


def functional_map_diagnostic_loss(evecs_x, evecs_y, Qx, Qy, mass_x, mass_y,
                                   corr_x, corr_y, feats_x, feats_y,
                                   evals_x, evals_y, lambda_=1e-1,
                                   resolvant_gamma=0.5):
    """Supervised diagnostic loss in FUNCTIONAL-MAP space, ||C - C_gt||^2.

    `alignment_loss` supervises Qx, Qy in POINT space (P,K residuals at the
    P corresponding vertices). This is the same underlying supervision (GT
    correspondence, same Qx/Qy), but re-expressed as a single K x K
    functional-map comparison, to separate two possible failure modes of the
    standalone per-shape rotation network:
      (a) the point-level objective is too weak/indirect for gradient descent
          to find the rotation, vs
      (b) the per-shape network CANNOT REPRESENT the needed rotation at all
          (it is a relational/pairwise quantity, not a function of one
          shape's features -- see basis_opt/README.md).
    If training with THIS loss still can't beat the raw/bypass baseline, that
    points to (b): representability, not the objective. If it works where the
    point loss didn't, that points to (a): the objective was the bottleneck.

    Two K x K functional maps are built in the CORRECTED basis
    Phi~_x = Phi_x @ Qx, Phi~_y = Phi_y @ Qy (both a function of the current
    Qx, Qy, so this is fully differentiable end-to-end):

      C_gt : the ground-truth fmap, fit by least squares from the GT point
             correspondence (corr_x, corr_y) -- the "answer key" map, exactly
             analogous to `gt_fmap_identity_deviation`'s C but not compared to
             the identity, compared to...
      C    : the fmap solved the way the joint pipeline / classical baseline
             does it (`regularized_fmap_solve`, the same closed-form
             regularized least-squares solve `methods.match_classical_fmap`
             and DSMK's `FasterRegularizedFMNet` use) -- i.e. project the
             frozen descriptor (feats_x, feats_y; WKS/xyz/etc, NOT learned)
             onto the corrected basis and solve for the map. No permutation
             network / argmin is involved, so gradients flow cleanly into
             Qx, Qy.

    Both C and C_gt are pinned to the same "Y -> X" solve direction (By @ C
    ~= Bx), matching `gt_fmap_identity_deviation` in this file, so `C - C_gt`
    compares like with like.

    Args (all for a single, unbatched shape pair):
        evecs_x, evecs_y: (Nx,K), (Ny,K) raw (uncorrected) LBO eigenbases.
        Qx, Qy: (K,K) orthonormal rotations (e.g. from CayleyONBCorrection).
        mass_x, mass_y: (Nx,), (Ny,) mass-matrix diagonals.
        corr_x, corr_y: (P,) GT-corresponding vertex indices.
        feats_x, feats_y: (Nx,Cf), (Ny,Cf) FROZEN descriptors (e.g. WKS/xyz)
            used to solve `C` -- not the GT correspondence.
        evals_x, evals_y: (K,) eigenvalues, for the resolvent regularizer.
        lambda_, resolvant_gamma: regularization of the `C` solve (same
            defaults as `methods.match_classical_fmap`).

    Returns:
        scalar tensor, mean squared Frobenius error ||C - C_gt||^2 / K^2.
    """
    evx_c, evy_c = evecs_x @ Qx, evecs_y @ Qy                    # (Nx,K), (Ny,K)

    # C_gt: least-squares fmap from the GT correspondence, in the CORRECTED
    # basis. Solves By_c @ C_gt ~= Bx_c  (Y -> X direction).
    Bx_c, By_c = evx_c[corr_x], evy_c[corr_y]                    # (P,K) each
    C_gt = torch.linalg.lstsq(By_c, Bx_c).solution               # (K,K)

    # C: regularized fmap from the frozen descriptor, solved the same way the
    # joint pipeline / classical baseline does (closed-form, differentiable),
    # but through the CORRECTED basis's mass-weighted projector. Solved in the
    # same Y -> X direction as C_gt: regularized_fmap_solve(A,B,...) finds C
    # with C @ A ~= B, so pass A = ay (Y-side), B = ax (X-side).
    evecs_trans_x = evx_c.t() * mass_x[None]                     # (K,Nx)
    evecs_trans_y = evy_c.t() * mass_y[None]                     # (K,Ny)
    ax = evecs_trans_x @ feats_x                                 # (K,Cf)
    ay = evecs_trans_y @ feats_y                                 # (K,Cf)
    C = regularized_fmap_solve(ay[None], ax[None], evals_y[None], evals_x[None],
                               lambda_=lambda_, resolvant_gamma=resolvant_gamma)[0]

    return (C - C_gt).pow(2).mean()


def closed_form_rotations(Bx: torch.Tensor, By: torch.Tensor):
    """Oracle Q_x, Q_y minimizing the alignment loss, via SVD of B_x^T B_y.

    Bx, By: (P, K) raw eigenvectors restricted to GT-corresponding points.
    Returns (Q_x, Q_y) each (K, K), orthonormal.
    """
    M = Bx.t() @ By                              # (K, K)
    U, _, Vh = torch.linalg.svd(M)
    Qx = U                                       # (K, K)
    Qy = Vh.t()                                  # (K, K)
    return Qx, Qy


# --------------------------------------------------------------------------- #
# Diagnostics + correspondence recovery
# --------------------------------------------------------------------------- #
def gt_fmap_identity_deviation(Bx_c: torch.Tensor, By_c: torch.Tensor) -> float:
    """max | C_gt(corrected) - I |, where C_gt = lstsq(By_c, Bx_c).

    A direct read on the objective: 0 means the GT functional map in the
    corrected basis is exactly the identity.
    """
    K = Bx_c.shape[1]
    C = torch.linalg.lstsq(By_c, Bx_c).solution        # (K, K)
    eye = torch.eye(K, device=C.device, dtype=C.dtype)
    return (C - eye).abs().max().item()


def recover_p2p(evecs_x_c: torch.Tensor, evecs_y_c: torch.Tensor) -> torch.Tensor:
    """Correspondence Y->X by nearest neighbour in the *corrected* basis.

    Once the bases are aligned, their rows act as aligned spectral coordinates,
    so a plain NN between Phi~_x and Phi~_y rows recovers the map -- no
    descriptors and no functional-map solve involved. Returns p2p (Ny,).
    """
    return nn_query(evecs_x_c, evecs_y_c)


def geo_error(dist_x: np.ndarray, corr_x: torch.Tensor, corr_y: torch.Tensor,
              p2p: torch.Tensor) -> float:
    """Mean geodesic error of a Y->X map against GT, using X's geodesic matrix."""
    return float(calculate_geodesic_error(
        dist_x, to_numpy(corr_x), to_numpy(corr_y), to_numpy(p2p)))


def evaluate_pair(evecs_x_c, evecs_y_c, Bx_c, By_c, dist_x, corr_x, corr_y):
    """Bundle the two numbers we care about for a single pair.

    Returns dict(geo, align, fmap_id_dev).
    """
    p2p = recover_p2p(evecs_x_c, evecs_y_c)
    return {
        'geo': geo_error(dist_x, corr_x, corr_y, p2p),
        'align': alignment_loss(Bx_c, By_c).item(),
        'fmap_id_dev': gt_fmap_identity_deviation(Bx_c, By_c),
    }
