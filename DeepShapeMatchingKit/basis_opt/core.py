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

from utils.fmap_util import nn_query
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
