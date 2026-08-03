"""Matching methods compared in the K-sweep (basis_opt/run_eval.py).

Every method takes two per-shape dicts `sx, sy` (CPU/GPU tensors: evecs (N,Kmax),
mass (N,), evals (Kmax,), wks (N,128), corr (P,)) and a basis dimension K, and
returns a point-to-point map p2p (Y->X), shape (Ny,). Geodesic error is scored by
the caller. None of these methods (except `oracle`) see the GT correspondence.

The point of the comparison: does learning the basis (`net`) beat the strong
classical functional-map baseline (`classical`) at test time, especially at low K?
`oracle` is the GT-aligned ceiling; `raw` is the un-aligned strawman.
"""

import torch

from utils.fmap_util import (nn_query, fmap2pointmap, regularized_fmap_solve,
                             zoomout)
from basis_opt import core


def _evecs_trans(evecs_k, mass):
    """Phi^T diag(mass), (K, N)."""
    return evecs_k.t() * mass[None]


# --------------------------------------------------------------------------- #
def match_wks_nn(sx, sy, K, **kw):
    """Baseline B: WKS used directly as a descriptor, nearest neighbour.
    K-independent (WKS is fixed); reported once."""
    return nn_query(sx['wks'], sy['wks'])


def match_raw_basis(sx, sy, K, **kw):
    """Strawman: NN in the raw (un-aligned) LBO eigenbasis, Q = I."""
    return core.recover_p2p(sx['evecs'][:, :K], sy['evecs'][:, :K])


def match_classical_fmap(sx, sy, K, lambda_=1e-1, resolvant_gamma=0.5,
                         do_zoomout=False, k_zoomout=None, **kw):
    """Baseline A (the real number to beat): regularized functional map from
    fixed WKS, GeomFmaps-style, then fmap -> p2p (optionally + ZoomOut refine).

    Project WKS onto the K-dim basis -> coefficients a_x, a_y (K, 128); solve
    C_xy with  min_C ||C a_x - a_y||^2 + lambda ||D ⊙ C||^2  (D = resolvent mask);
    convert C_xy to p2p by NN in the spectral embedding.
    """
    evx, evy = sx['evecs'][:, :K], sy['evecs'][:, :K]
    ax = _evecs_trans(evx, sx['mass']) @ sx['wks']        # (K, 128)
    ay = _evecs_trans(evy, sy['mass']) @ sy['wks']        # (K, 128)
    Cxy = regularized_fmap_solve(
        ax[None], ay[None], sx['evals'][:K][None], sy['evals'][:K][None],
        lambda_=lambda_, resolvant_gamma=resolvant_gamma)[0]              # (K, K)
    p2p = fmap2pointmap(Cxy, evx, evy)
    if do_zoomout:
        kf = k_zoomout or sx['evecs'].shape[1]
        p2p = zoomout(p2p, sx['evecs'][:, :kf], sy['evecs'][:, :kf],
                      k_init=K, k_final=kf)
    return p2p


def match_oracle_basis(sx, sy, K, **kw):
    """Ceiling: GT-aligned basis (closed-form SVD Q from GT corr), NN-in-basis.
    Uses the ground-truth correspondence -- diagnostic upper bound, not a method."""
    evx, evy = sx['evecs'][:, :K], sy['evecs'][:, :K]
    Bx, By = evx[sx['corr']], evy[sy['corr']]
    Qx, Qy = core.closed_form_rotations(Bx, By)
    return core.recover_p2p(evx @ Qx, evy @ Qy)


def match_net_basis(sx, sy, K, net=None, **kw):
    """Ours: rotation Q predicted from WKS by the trained CayleyONBCorrection,
    NN-in-(corrected)-basis. No GT, no fmap solve at test time."""
    evx = net(sx['evecs'][:, :K], sx['mass'], sx['evals'][:K], sx['wks'])
    evy = net(sy['evecs'][:, :K], sy['mass'], sy['evals'][:K], sy['wks'])
    return core.recover_p2p(evx, evy)


def match_net_fmap_supervised(sx, sy, K, net=None, **kw):
    """Same architecture, same frozen WKS, same test-time matching as
    `match_net_basis` -- NN-in-(corrected)-basis, no GT/fmap solve at test
    time. The ONLY difference is which loss trained `net`'s checkpoint:
    `core.functional_map_diagnostic_loss` (functional-map space) here, vs
    `core.alignment_loss` (point space) for `match_net_basis`. This is a
    separate registry key purely so run_eval's table can show both
    checkpoints side by side under distinct, labeled columns; the matching
    logic itself is identical, so it just delegates.
    """
    return match_net_basis(sx, sy, K, net=net, **kw)


# registry used by run_eval (order = column order)
METHODS = {
    'wks_nn': match_wks_nn,
    'raw': match_raw_basis,
    'classical': match_classical_fmap,
    'net': match_net_basis,
    'net_fmap_supervised': match_net_fmap_supervised,
    'oracle': match_oracle_basis,
}
