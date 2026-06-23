"""Phase 4b verification for the validate_single override.

(A) Eval-mode determinism: under .eval() the isometry uses the linspace
    time-span (not sorted-random), so test-time Q is fixed across calls.
(B) Bypass anchor: with bypass=True, ONB_ULRSSM_Model.validate_single must
    reproduce vanilla ULRSSM_Model.validate_single bit-for-bit (same instance,
    same stubbed features, same data => identical p2p / Cxy / Pyx, hence
    identical geodesic error).

Run from anywhere:  python networks/onb/selftest_validate.py
"""

import os
import sys

_KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _KIT_ROOT not in sys.path:
    sys.path.insert(0, _KIT_ROOT)

import torch

from utils.options import parse
from models import build_model
from models.ulrssm_model import ULRSSM_Model


class _Timer:  # minimal stand-in for AvgTimer
    def start(self):
        pass

    def record(self):
        pass


def make_mass_orthonormal_basis(N, K, dtype):
    mass = torch.rand(N, dtype=dtype) + 0.5
    X = torch.randn(N, K, dtype=dtype)
    Qe, _ = torch.linalg.qr(torch.sqrt(mass)[:, None] * X)
    Phi = Qe / torch.sqrt(mass)[:, None]
    return Phi, mass


def build(cfg_rel):
    return build_model(parse(cfg_rel, _KIT_ROOT, is_train=True))


def main():
    torch.manual_seed(0)
    dtype = torch.float32
    N, K, Cf = 200, 16, 256

    # ---- (A) eval-mode determinism (needs the active flow => bypass:false) ----
    print("== (A) eval-mode time discretization is deterministic ==")
    model_active = build("options/ulrssm/train/faust_onb.yaml")
    model_active.eval()
    onb = model_active.networks["onb_correction"]
    iso = onb.isometry
    print(f"  isometry.training under .eval()        = {iso.training}")
    Phi, mass = make_mass_orthonormal_basis(N, K, dtype)
    feats = torch.randn(N, Cf, dtype=dtype)
    evals = torch.sort(torch.rand(K, dtype=dtype))[0]
    Q1 = onb.compute_rotation(Phi, mass, evals, feats)
    tspan1 = iso.tspan.clone()
    Q2 = onb.compute_rotation(Phi, mass, evals, feats)   # re-shuffles internally
    tspan2 = iso.tspan.clone()
    expected = torch.linspace(0, 1, onb.num_steps)
    print(f"  tspan == linspace(0,1,L)               : {torch.allclose(tspan1, expected)}")
    print(f"  tspan stable across calls              : {torch.equal(tspan1, tspan2)}")
    print(f"  Q identical across calls (determinism) : {torch.equal(Q1, Q2)}")
    assert not iso.training, "isometry must be in eval mode under model.eval()"
    assert torch.allclose(tspan1, expected), "eval tspan must be linspace"
    assert torch.equal(Q1, Q2), "Q must be deterministic in eval mode"

    # ---- (B) bypass anchor: override == vanilla, bit-for-bit ----
    print("\n== (B) bypass=True validate_single reproduces vanilla ==")
    model = build("options/ulrssm/train/faust_onb_bypass.yaml")
    model.eval()

    Phi_x, mass_x = make_mass_orthonormal_basis(N, K, dtype)
    Phi_y, mass_y = make_mass_orthonormal_basis(N, K, dtype)
    feats_x = torch.randn(N, Cf, dtype=dtype)
    feats_y = torch.randn(N, Cf, dtype=dtype)
    evals_x = torch.sort(torch.rand(K, dtype=dtype))[0]
    evals_y = torch.sort(torch.rand(K, dtype=dtype))[0]

    def b(t):
        return t.unsqueeze(0)

    def make_data():
        return {
            "first": {
                "evecs": b(Phi_x), "evecs_trans": b(Phi_x.t() * mass_x[None]),
                "mass": b(mass_x), "evals": b(evals_x),
            },
            "second": {
                "evecs": b(Phi_y), "evecs_trans": b(Phi_y.t() * mass_y[None]),
                "mass": b(mass_y), "evals": b(evals_y),
            },
        }

    # deterministic stubbed feature extractor (returns x then y per call)
    feats_seq = [b(feats_x), b(feats_y)]
    state = {"i": 0}

    def fake_feat(data=None):
        out = feats_seq[state["i"] % 2]
        state["i"] += 1
        return out

    model.networks["feature_extractor"] = fake_feat

    # vanilla path: call the PARENT method on the same instance (reads raw basis)
    state["i"] = 0
    p2p_van, Pyx_van, Cxy_van = ULRSSM_Model.validate_single(model, make_data(), _Timer())
    # ONB override path (bypass=True -> correction is identity)
    state["i"] = 0
    p2p_onb, Pyx_onb, Cxy_onb = model.validate_single(make_data(), _Timer())

    p2p_equal = torch.equal(p2p_van, p2p_onb)
    cxy_err = (Cxy_van - Cxy_onb).abs().max().item()
    pyx_err = (Pyx_van - Pyx_onb).abs().max().item()
    print(f"  p2p identical (bit-for-bit)            : {p2p_equal}")
    print(f"  max|Cxy_vanilla - Cxy_onb|             = {cxy_err:.3e}")
    print(f"  max|Pyx_vanilla - Pyx_onb|             = {pyx_err:.3e}")

    assert p2p_equal, "bypass validate_single must reproduce vanilla p2p exactly"
    assert cxy_err == 0.0, f"Cxy must match vanilla bit-for-bit: {cxy_err}"
    assert pyx_err == 0.0, f"Pyx must match vanilla bit-for-bit: {pyx_err}"
    print("\nPASS")


if __name__ == "__main__":
    main()
