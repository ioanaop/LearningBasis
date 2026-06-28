"""K-sweep evaluation: does learning the basis beat the classical fmap baseline?

For a fixed set of held-out FAUST test pairs, evaluates each matching method
(basis_opt/methods.py) across a list of basis dimensions K, and prints a table of
mean geodesic error. Training-free methods (wks_nn, raw, classical, oracle) need
no checkpoint; the `net` column is filled per-K from a trained CayleyONBCorrection
checkpoint (basis_opt/ckpts/wks_onb_k{K}_r{rank}.pth) if present, else shown as '-'.

Run (workstation):
    cd DeepShapeMatchingKit
    CUDA_VISIBLE_DEVICES=1 conda run -n deepshapematchingkit \
        python basis_opt/run_eval.py --k_list 20,40,60,100 --n_pairs 60
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random

import numpy as np
import torch

from networks.onb.correction import CayleyONBCorrection
from basis_opt import core
from basis_opt.methods import METHODS, match_classical_fmap
from basis_opt.dataio import load_shapes, to_device


def load_net(ckpt_dir, K, rank, device, dataset='faust'):
    path = os.path.join(ckpt_dir, f'{dataset}_wks_onb_k{K}_r{rank}.pth')
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location=device)
    num_steps = blob.get('args', {}).get('num_steps', 20)
    net = CayleyONBCorrection(feature_dim=128, rank=rank, num_steps=num_steps,
                              bypass=False, gradient_checkpointing=False).to(device)
    net.load_state_dict(blob['state_dict'])
    net.eval()
    return net


@torch.no_grad()
def mean_geo(method, shapes, pairs, K, device, **kw):
    errs = []
    for ix, iy in pairs:
        sx, sy = to_device(shapes[ix], device), to_device(shapes[iy], device)
        p2p = method(sx, sy, K, **kw)
        errs.append(core.geo_error(sx['dist'], sx['corr'], sy['corr'], p2p))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='faust', choices=['faust', 'smal'])
    ap.add_argument('--data_root', default=None,
                    help='default ../data/FAUST_r or ../data/SMAL_r by --dataset')
    ap.add_argument('--k_list', default='20,40,60,100')
    ap.add_argument('--rank', type=int, default=16, help='Cayley rank used for ckpt lookup')
    ap.add_argument('--n_pairs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--classical_lambda', type=float, default=1e-1,
                    help='resolvent regularization for the classical fmap baseline')
    ap.add_argument('--zoomout', action='store_true', help='add classical+ZoomOut column')
    ap.add_argument('--k_zoomout', type=int, default=None, help='ZoomOut final K (default Kmax)')
    ap.add_argument('--ckpt', default=os.path.join(os.path.dirname(__file__), 'ckpts'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_root = args.data_root or ('../data/SMAL_r' if args.dataset == 'smal'
                                   else '../data/FAUST_r')
    k_list = [int(x) for x in args.k_list.split(',')]
    kmax = max(k_list)
    print(f'[eval] {args.dataset} test, K in {k_list}, {args.n_pairs} pairs, device={device}')
    shapes = load_shapes(data_root, 'test', k=kmax, with_dist=True, with_wks=True,
                         dataset=args.dataset)
    n = len(shapes)

    pairs = []
    while len(pairs) < args.n_pairs:
        i, j = random.randrange(n), random.randrange(n)
        if i != j:
            pairs.append((i, j))

    cols = ['wks_nn', 'raw', 'classical']
    if args.zoomout:
        cols.append('classical+zo')
    cols += ['net', 'oracle']

    # wks_nn is K-independent: compute once.
    wks_val = mean_geo(METHODS['wks_nn'], shapes, pairs, kmax, device)

    print('\n' + 'K'.rjust(5) + ''.join(c.rjust(14) for c in cols))
    rows = {}
    for K in k_list:
        net = load_net(args.ckpt, K, args.rank, device, dataset=args.dataset)
        vals = {}
        vals['wks_nn'] = wks_val
        vals['raw'] = mean_geo(METHODS['raw'], shapes, pairs, K, device)
        vals['classical'] = mean_geo(METHODS['classical'], shapes, pairs, K, device,
                                     lambda_=args.classical_lambda)
        if args.zoomout:
            vals['classical+zo'] = mean_geo(
                match_classical_fmap, shapes, pairs, K, device,
                lambda_=args.classical_lambda,
                do_zoomout=True, k_zoomout=args.k_zoomout or kmax)
        vals['net'] = mean_geo(METHODS['net'], shapes, pairs, K, device, net=net) if net else None
        vals['oracle'] = mean_geo(METHODS['oracle'], shapes, pairs, K, device)
        rows[K] = vals
        cells = ''.join((f'{vals[c]:.4f}' if vals[c] is not None else '-').rjust(14) for c in cols)
        print(f'{K:5d}{cells}')

    print('\n(lower = better; net=ours predicted-Q NN-in-basis, oracle=GT-aligned ceiling)')


if __name__ == '__main__':
    main()
