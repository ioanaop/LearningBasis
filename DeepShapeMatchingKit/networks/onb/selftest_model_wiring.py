"""Phase 4 verification: both ONB configs instantiate, and feed_data keeps all
four basis tensors (evecs_x/y, evecs_trans_x/y) in a consistent corrected basis.

The basis-consistency check is the one the Q=I anchor cannot catch: it runs the
REAL ONB_ULRSSM_Model.feed_data with Q != I on a synthetic ground-truth
correspondence pair (shape Y = shape X relabelled by a known permutation, with
correspondingly permuted features). Under the GT permutation the closed-form
functional map Cxy_est must equal the identity in the corrected bases — which
holds iff evecs_trans_y, Pyx and evecs_x are all the corrected versions. If any
one were left raw, Cxy_est would come out as Q (not I); we demonstrate that too.

Run from anywhere:  python networks/onb/selftest_model_wiring.py
"""

import os
import sys

_KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _KIT_ROOT not in sys.path:
    sys.path.insert(0, _KIT_ROOT)

import torch

from utils.options import parse
from models import build_model


def instantiate(cfg_rel):
    opt = parse(cfg_rel, _KIT_ROOT, is_train=True)
    model = build_model(opt)
    onb = model.networks["onb_correction"]
    n_params = sum(p.numel() for p in onb.parameters())
    print(f"  [{cfg_rel}]  model={type(model).__name__}  "
          f"onb.bypass={onb.bypass}  onb#params={n_params}")
    return model


def make_mass_orthonormal_basis(N, K, dtype):
    mass = torch.rand(N, dtype=dtype) + 0.5
    X = torch.randn(N, K, dtype=dtype)
    Qe, _ = torch.linalg.qr(torch.sqrt(mass)[:, None] * X)
    Phi = Qe / torch.sqrt(mass)[:, None]
    return Phi, mass


def main():
    torch.manual_seed(0)

    print("== (A) both configs parse + model instantiates ==")
    model_bypass = instantiate("options/ulrssm/train/faust_onb_bypass.yaml")
    model = instantiate("options/ulrssm/train/faust_onb.yaml")     # bypass: false

    print("\n== (B) Q != I basis-consistency on the real feed_data ==")
    dtype = torch.float32
    N, K, Cf = 200, 16, 256       # Cf must match config feature_dim=256
    Phi_x, mass_x = make_mass_orthonormal_basis(N, K, dtype)
    perm = torch.randperm(N)
    Phi_y, mass_y = Phi_x[perm], mass_x[perm]      # GT-correspondent shape Y
    feats_x = torch.randn(N, Cf, dtype=dtype)
    feats_y = feats_x[perm]                          # features follow the corr.
    evals = torch.sort(torch.rand(K, dtype=dtype))[0]

    # GT permutation matrices: y_i <-> x_perm[i]
    Pyx = torch.zeros(N, N, dtype=dtype)
    Pyx[torch.arange(N), perm] = 1.0                 # (Ny, Nx)
    Pxy = Pyx.t().contiguous()                       # (Nx, Ny)

    def b(t):  # add batch dim
        return t.unsqueeze(0)

    data = {
        "first":  {"evecs": b(Phi_x), "evals": b(evals), "mass": b(mass_x)},
        "second": {"evecs": b(Phi_y), "evals": b(evals), "mass": b(mass_y)},
    }

    model.eval()  # deterministic time-span => reproducible Q

    # stub the feature extractor (returns feats in call order: x then y)
    feats_seq = [b(feats_x), b(feats_y)]
    calls = {"i": 0}

    def fake_feat(data=None):
        out = feats_seq[calls["i"]]
        calls["i"] += 1
        return out

    model.networks["feature_extractor"] = fake_feat
    # inject the ground-truth permutation
    model.compute_permutation_matrix = (
        lambda fx, fy, bidirectional=False, normalize=True:
        (b(Pxy), b(Pyx)) if bidirectional else b(Pyx)
    )
    # capture (C, C_est) passed to the align loss
    captured = []
    real_align = model.losses["align_loss"]

    def cap_align(a, c):
        captured.append((a.detach(), c.detach()))
        return real_align(a, c)

    model.losses["align_loss"] = cap_align

    model.feed_data(data)

    eyeK = torch.eye(K, dtype=dtype)
    Cxy, Cxy_est = captured[0]
    Cyx, Cyx_est = captured[1]
    err_xy = (Cxy_est[0] - eyeK).abs().max().item()
    err_yx = (Cyx_est[0] - eyeK).abs().max().item()

    # Q must be non-trivial, else the test is vacuous.
    Q = model.networks["onb_correction"].compute_rotation(
        b(Phi_x)[0], b(mass_x)[0], b(evals)[0], b(feats_x)[0])
    q_nontriv = (Q - eyeK).abs().max().item()

    # Mk sanity (init-time style check, loose float32 tolerance).
    mk_err = (model.Mk_x[0] - eyeK).abs().max().item()

    # Deliberate mismatch: raw evecs_trans_y with corrected evecs_x -> NOT I.
    et_y_raw = (Phi_y.t() * mass_y[None])           # raw projector (K,N)
    evecs_x_corr = Phi_x @ Q                          # corrected (N,K)
    Cxy_bad = et_y_raw @ Pyx @ evecs_x_corr
    err_bad = (Cxy_bad - eyeK).abs().max().item()

    print(f"  Q non-trivial            max|Q - I|        = {q_nontriv:.3e}")
    print(f"  Mk_x sanity              max|Mk_x - I|      = {mk_err:.3e}")
    print(f"  Cxy_est (GT corr)        max|Cxy_est - I|   = {err_xy:.3e}  (consistent bases)")
    print(f"  Cyx_est (GT corr)        max|Cyx_est - I|   = {err_yx:.3e}  (consistent bases)")
    print(f"  mismatch demo (1 raw)    max|Cxy_bad - I|   = {err_bad:.3e}  (would-be bug)")

    assert q_nontriv > 1e-2, "Q must be non-trivial or the test is vacuous"
    assert mk_err < 1e-3, f"Mk should be ~I in coefficient space: {mk_err}"
    assert err_xy < 1e-3, f"Cxy_est not identity under GT corr -> basis mismatch: {err_xy}"
    assert err_yx < 1e-3, f"Cyx_est not identity under GT corr -> basis mismatch: {err_yx}"
    assert err_bad > 1e-2, "mismatch demo should clearly fail (proves the test bites)"
    print("\nPASS")


if __name__ == "__main__":
    main()
