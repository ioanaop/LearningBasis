"""Standalone self-test for CayleyONBCorrection.

Checks two properties on random data:
  (1) bypass=True returns the input basis bit-for-bit;
  (2) bypass=False returns a mass-orthonormal basis, Φ̃ᵀ diag(mass) Φ̃ ≈ I.

Run from anywhere:  python networks/onb/selftest_correction.py
(adds the kit root to sys.path so `networks.onb.*` resolves).
"""

import os
import sys

# Make the kit root importable when run as a script.
_KIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _KIT_ROOT not in sys.path:
    sys.path.insert(0, _KIT_ROOT)

import torch

from networks.onb.correction import CayleyONBCorrection


def make_mass_orthonormal_basis(N, K, dtype):
    """Random mass>0 and Φ with Φᵀ diag(mass) Φ = I."""
    mass = torch.rand(N, dtype=dtype) + 0.5
    X = torch.randn(N, K, dtype=dtype)
    Y = torch.sqrt(mass)[:, None] * X            # M^{1/2} X
    Qe, _ = torch.linalg.qr(Y)                   # Qeᵀ Qe = I, (N, K)
    Phi = Qe / torch.sqrt(mass)[:, None]         # Φᵀ M Φ = Qeᵀ Qe = I
    return Phi, mass


def mass_orth_residual(Phi, mass, K, dtype):
    G = (Phi.t() * mass[None]) @ Phi
    return (G - torch.eye(K, dtype=dtype)).abs().max().item()


def main():
    torch.manual_seed(0)
    dtype = torch.float64        # math check: the Cayley map is exactly orthonormal
    N, K, Cf = 400, 16, 32

    Phi, mass = make_mass_orthonormal_basis(N, K, dtype)
    # sanity: the constructed input basis really is mass-orthonormal
    base_res = mass_orth_residual(Phi, mass, K, dtype)
    print(f"input Φ mass-orthonormality residual           = {base_res:.3e}")
    assert base_res < 1e-8, base_res

    evals = torch.sort(torch.rand(K, dtype=dtype))[0]   # increasing spectrum
    feats = torch.randn(N, Cf, dtype=dtype)

    # ---- (1) bypass: exact passthrough ----
    net_bypass = CayleyONBCorrection(feature_dim=Cf, rank=8, num_steps=20,
                                     bypass=True).to(dtype)
    out_bypass = net_bypass(Phi, mass, evals, feats)
    bypass_equal = torch.equal(out_bypass, Phi)
    print(f"(1) bypass=True  returns input bit-for-bit       : {bypass_equal}")

    # ---- (2) non-bypass: mass-orthonormal, basis actually changed ----
    net = CayleyONBCorrection(feature_dim=Cf, rank=8, num_steps=20,
                              bypass=False).to(dtype)
    net.eval()
    out = net(Phi, mass, evals, feats)
    changed = not torch.equal(out, Phi)
    res = mass_orth_residual(out, mass, K, dtype)
    print(f"(2) bypass=False basis changed                   : {changed}")
    print(f"(2) bypass=False max|Φ̃ᵀ diag(mass) Φ̃ - I|        = {res:.3e}")

    # shared-instance check: same module on two different inputs (X and Y)
    feats_y = torch.randn(N, Cf, dtype=dtype)
    Phi_y, mass_y = make_mass_orthonormal_basis(N, K, dtype)
    out_y = net(Phi_y, mass_y, evals, feats_y)
    res_y = mass_orth_residual(out_y, mass_y, K, dtype)
    print(f"    same instance on 2nd shape, residual         = {res_y:.3e}")

    # ---- (3) forced-identity through the FULL non-bypass path ----
    # Zero the generator so the flow returns Q = I via the real code (coords
    # build -> pushforward -> Φ̃ = ΦQ), then confirm it reproduces the
    # short-circuit bypass output bit-for-bit. This exercises the ΦQ plumbing,
    # which the trivial short-circuit bypass skips.
    net_id = CayleyONBCorrection(feature_dim=Cf, rank=8, num_steps=20,
                                 bypass=False).to(dtype)
    net_id.eval()
    with torch.no_grad():
        for p in net_id.parameters():
            p.zero_()
    Q_forced = net_id.compute_rotation(Phi, mass, evals, feats)
    q_id_res = (Q_forced - torch.eye(K, dtype=dtype)).abs().max().item()
    out_id = net_id(Phi, mass, evals, feats)
    id_equal = torch.equal(out_id, out_bypass)
    print(f"(3) forced-Q: max|Q - I|                         = {q_id_res:.3e}")
    print(f"(3) full path with Q=I == bypass, bit-for-bit    : {id_equal}")

    # ---- (4) composition-convention check, with Q != I ----
    # In eval mode the time-span is deterministic, so compute_rotation is
    # reproducible. Φ̃ = ΦQ  =>  Φ̃ᵀ diag(mass) = Qᵀ (Φᵀ diag(mass)).
    Q = net.compute_rotation(Phi, mass, evals, feats)
    q_orth = (Q.t() @ Q - torch.eye(K, dtype=dtype)).abs().max().item()
    q_nontriv = (Q - torch.eye(K, dtype=dtype)).abs().max().item()
    Phi_t = Phi @ Q
    et_direct = Phi_t.t() * mass[None]                 # Φ̃ᵀ diag(mass)  (direct)
    et_old = Phi.t() * mass[None]                       # Φᵀ diag(mass)
    err_QT = (et_direct - Q.t() @ et_old).abs().max().item()   # convention Φ̃=ΦQ
    err_Q = (et_direct - Q @ et_old).abs().max().item()        # the other one
    # forward() must reproduce Φ@Q in eval mode (same deterministic Q)
    fwd_matches = torch.allclose(net(Phi, mass, evals, feats), Phi_t, atol=1e-10)
    print(f"(4) Q orthonormal max|QᵀQ - I|                   = {q_orth:.3e}")
    print(f"(4) Q non-trivial max|Q - I|                     = {q_nontriv:.3e}")
    print(f"(4) evecs_trans:  ||Φ̃ᵀM - Qᵀ·ΦᵀM||  (CONVENTION) = {err_QT:.3e}")
    print(f"(4) evecs_trans:  ||Φ̃ᵀM - Q ·ΦᵀM||  (wrong one)  = {err_Q:.3e}")
    print(f"(4) forward() reproduces Φ@Q in eval mode        : {fwd_matches}")

    assert bypass_equal, "bypass must return the input exactly"
    assert changed, "non-bypass should modify the basis"
    assert res < 1e-4, f"mass-orthonormality residual too large: {res}"
    assert res_y < 1e-4, f"residual (2nd shape) too large: {res_y}"
    assert id_equal, "forced-identity full path must equal bypass bit-for-bit"
    assert q_orth < 1e-4, "forced Q must be orthonormal"
    assert q_nontriv > 1e-2, "Q should be non-trivial (test would be vacuous)"
    assert err_QT < 1e-5, f"convention Φ̃=ΦQ => Φ̃ᵀM = Qᵀ·ΦᵀM failed: {err_QT}"
    assert err_Q > 1e-2, "the wrong convention should clearly NOT match"
    assert fwd_matches, "forward() must reproduce Φ@Q"
    print("\nPASS")


if __name__ == "__main__":
    main()
