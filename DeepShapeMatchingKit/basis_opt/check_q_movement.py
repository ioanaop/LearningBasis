"""Diagnostic: did Q actually move during training, or is the optimizer
getting no gradient signal at all?

Compares each checkpoint's TRAINED Q against a FRESH, untrained net built with
the exact same seed + config (i.e. reconstructs "step 0" of that same run) --
both evaluated in .eval() mode, because CayleyONBCorrection.compute_rotation
calls shuffle_model_state(), which resamples the Cayley flow's ODE time-span
RANDOMLY in train() mode but uses a fixed linspace in eval() mode (see
networks/onb/eulerian.py). Comparing in train mode would make "Q moved" not
mean anything -- it could just be that per-call resampling noise, not
learning. eval() mode is required for a meaningful step0-vs-final comparison.

Answers: did fmap_supervised's Q move a lot (-> optimizer got a signal, the
flat loss is a degenerate valley) or barely at all (-> a real gradient-flow
bug / lr mismatch, since its loss scale is ~40x smaller than point_align's)?
point_align (which visibly learned: geo error and align loss both improved)
is also checked as a positive control on the methodology itself.

Run:
    python basis_opt/check_q_movement.py \
        --ckpt basis_opt/ckpts/faust_wks_onb_k60_r32_point_align.pth \
        --ckpt basis_opt/ckpts/faust_wks_onb_k60_r32_fmap_supervised.pth
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random

import numpy as np
import torch

from networks.onb.correction import CayleyONBCorrection
from basis_opt.dataio import load_shapes, to_device


def _net(feat_dim, a, device):
    return CayleyONBCorrection(
        feature_dim=feat_dim, rank=a['rank'], num_steps=a['num_steps'],
        bypass=False, base_acceleration=a['base_accel'],
        gradient_checkpointing=False,
    ).to(device)


def build_fresh_net(a, feat_dim, device):
    """Reconstruct the EXACT step-0 net: same seeding order as
    run_stage1.main() (random/np/torch seeded, THEN net constructed), so its
    initial weights are bit-identical to what step 0 of the real run had."""
    random.seed(a['seed'])
    np.random.seed(a['seed'])
    torch.manual_seed(a['seed'])
    return _net(feat_dim, a, device)


def load_trained_net(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device)
    a = blob['args']
    data_root = a['data_root'] or ('../data/SMAL_r' if a['dataset'] == 'smal'
                                   else '../data/FAUST_r')
    feat_key = a['feature']
    return blob['state_dict'], a, data_root, feat_key


@torch.no_grad()
def q_for_shapes(net, shapes, feat_key):
    """Q (K,K) per shape, in eval() mode for determinism (fixed tspan)."""
    net.eval()
    return [net.compute_rotation(s['evecs'], s['mass'], s['evals'], s[feat_key])
            for s in shapes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    help='checkpoint path; repeat for multiple checkpoints')
    ap.add_argument('--n_shapes', type=int, default=6,
                    help='# probe shapes, split half train / half test')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    for ckpt_path in args.ckpt:
        state_dict, a, data_root, feat_key = load_trained_net(ckpt_path, device)
        need_wks = (feat_key == 'wks')

        n_half = args.n_shapes // 2
        train_shapes = load_shapes(data_root, 'train', k=a['k'], with_dist=False,
                                   with_wks=need_wks, dataset=a['dataset'],
                                   indices=list(range(n_half)))
        test_shapes = load_shapes(data_root, 'test', k=a['k'], with_dist=False,
                                  with_wks=need_wks, dataset=a['dataset'],
                                  indices=list(range(n_half)))
        probe = [to_device(s, device) for s in (train_shapes + test_shapes)]
        feat_dim = probe[0][feat_key].shape[1]

        net0 = build_fresh_net(a, feat_dim, device)
        net_final = _net(feat_dim, a, device)
        net_final.load_state_dict(state_dict)

        Q0s = q_for_shapes(net0, probe, feat_key)
        Qfs = q_for_shapes(net_final, probe, feat_key)

        K = Q0s[0].shape[0]
        eye = torch.eye(K, device=device)
        print(f'\n=== {os.path.basename(ckpt_path)}  '
              f'(loss_mode={a.get("loss_mode", "?")}, dataset={a["dataset"]}, '
              f'feature={feat_key}, K={K}) ===')
        dQs = []
        for i, (q0, qf, s) in enumerate(zip(Q0s, Qfs, probe)):
            split = 'train' if i < n_half else 'test'
            dQ = (qf - q0).norm().item()
            d0 = (q0 - eye).norm().item()
            df = (qf - eye).norm().item()
            dQs.append(dQ)
            print(f'  [{split}] {s["name"]:14s}  ||Q_final-Q_0||={dQ:7.4f}   '
                  f'||Q_0-I||={d0:7.4f}   ||Q_final-I||={df:7.4f}')

        eye_norm = K ** 0.5
        print(f'  MEAN ||Q_final - Q_0|| = {np.mean(dQs):.4f}   '
              f'(for scale: ||I||_F = sqrt(K) = {eye_norm:.4f}; two INDEPENDENT '
              f'random orthogonal matrices differ by ~{eye_norm * 2**0.5:.4f} '
              f'on average)')


if __name__ == '__main__':
    main()
