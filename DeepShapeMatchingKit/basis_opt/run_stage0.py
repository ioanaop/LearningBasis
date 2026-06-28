"""Stage 0 -- single-pair, free-rotation sanity check.

Pure test of the idea, with NO network and NO descriptors: for one FAUST pair,
directly optimize two orthonormal matrices Q_x, Q_y so that the eigenbases align
at the ground-truth corresponding points. Compares three settings:

    raw      : Q = I               (vanilla LBO basis -- the thing we want to beat)
    oracle   : closed-form SVD Q   (global optimum of the alignment loss)
    learned  : gradient descent    (should recover the oracle)

For each we report mean geodesic error of the correspondence recovered by plain
nearest-neighbour in the (corrected) basis -- i.e. the basis itself is the only
"descriptor". If aligning the basis helps, `oracle`/`learned` beat `raw` by a lot.

Run (workstation):
    cd DeepShapeMatchingKit
    CUDA_VISIBLE_DEVICES=1 conda run -n deepshapematchingkit \
        python basis_opt/run_stage0.py --pair 0 1 --k 100
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch

from basis_opt import core
from basis_opt.dataio import load_shapes, to_device


def optimize_free_rotation(Bx, By, k, steps, lr, log_every=100):
    """Learn Q_x, Q_y (FreeRotation) minimizing the alignment loss on one pair."""
    device = Bx.device
    Qx = core.FreeRotation(k).to(device)
    Qy = core.FreeRotation(k).to(device)
    opt = torch.optim.Adam(list(Qx.parameters()) + list(Qy.parameters()), lr=lr)

    for step in range(steps):
        opt.zero_grad()
        loss = core.alignment_loss(Bx @ Qx(), By @ Qy())
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'    step {step:4d}  align_loss {loss.item():.6e}')

    with torch.no_grad():
        return Qx().detach(), Qy().detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../data/FAUST_r')
    ap.add_argument('--phase', default='test', choices=['train', 'test'])
    ap.add_argument('--pair', type=int, nargs=2, default=[0, 1],
                    help='shape indices within the phase (X Y)')
    ap.add_argument('--k', type=int, default=100, help='# eigenbasis modes')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    ix, iy = args.pair
    print(f'[stage0] loading FAUST {args.phase} shapes {ix},{iy}  k={args.k}  device={device}')
    shapes = load_shapes(args.data_root, args.phase, k=args.k,
                         with_dist=True, with_wks=False, indices=[ix, iy])
    sx, sy = to_device(shapes[0], device), to_device(shapes[1], device)
    print(f'         X={sx["name"]} (N={sx["evecs"].shape[0]})  '
          f'Y={sy["name"]} (N={sy["evecs"].shape[0]})  P={sx["corr"].shape[0]}')

    evx, evy = sx['evecs'], sy['evecs']
    Bx, By = evx[sx['corr']], evy[sy['corr']]          # (P, k)
    dist_x = sx['dist']
    cx, cy = sx['corr'], sy['corr']

    def report(tag, Qx, Qy):
        res = core.evaluate_pair(evx @ Qx, evy @ Qy, Bx @ Qx, By @ Qy, dist_x, cx, cy)
        print(f'  {tag:8s}  geo_err {res["geo"]:.4f}   align {res["align"]:.3e}   '
              f'|C_gt-I|max {res["fmap_id_dev"]:.3e}')
        return res

    eye = torch.eye(args.k, device=device)
    print('\nresults (geo_err = mean geodesic error of NN-in-basis correspondence):')
    report('raw', eye, eye)

    Qx_cf, Qy_cf = core.closed_form_rotations(Bx, By)
    report('oracle', Qx_cf, Qy_cf)

    print('  learned   optimizing free rotation ...')
    Qx_l, Qy_l = optimize_free_rotation(Bx, By, args.k, args.steps, args.lr)
    report('learned', Qx_l, Qy_l)


if __name__ == '__main__':
    main()
