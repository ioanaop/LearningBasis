"""Stage 1 -- learn the basis transformation as a function of WKS features.

Same objective as Stage 0, but instead of one free rotation per shape we train a
single shared network that *predicts* the rotation Q from fixed, per-mode WKS
descriptors (+ eigen-position). Descriptors are NOT learned; only the rotation
network is. This is the generalizing version: at test time it produces Q for
unseen shapes from their WKS alone.

We reuse the kit's `CayleyONBCorrection` (Phi~ = Phi Q, Q from the Cayley flow),
feeding it WKS as the conditioning `feats` instead of learned DiffusionNet
features. Loss = alignment loss on GT-corresponding points. No fmap solver, no
descriptor learning -- gradients flow only into the rotation network.

Stage 2 hook: swap `feats` from WKS to a *frozen pretrained* DiffusionNet
(DeepShapeMatching) -- pass `--feature shot` for SHOT, or wire a frozen net where
WKS is computed in dataio. Everything else here is unchanged.

Run (workstation):
    cd DeepShapeMatchingKit
    CUDA_VISIBLE_DEVICES=1 conda run -n deepshapematchingkit \
        python basis_opt/run_stage1.py --k 100 --steps 2000
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random

import numpy as np
import torch

from networks.onb.correction import CayleyONBCorrection
from basis_opt.rotnet import ConditionedRotation
from basis_opt import core
from basis_opt.dataio import load_shapes, to_device


def corrected_basis(net, shape, feat_key):
    """Phi~ = Phi Q with Q predicted from this shape's fixed `feat_key` descriptor."""
    return net(shape['evecs'], shape['mass'], shape['evals'], shape[feat_key])


@torch.no_grad()
def evaluate(net, test_shapes, pairs, device, feat_key):
    """Average geo error over `pairs` for raw / oracle / net."""
    net.eval()
    agg = {'raw': [], 'oracle': [], 'net': [], 'net_align': [], 'net_iddev': []}
    for ix, iy in pairs:
        sx, sy = to_device(test_shapes[ix], device), to_device(test_shapes[iy], device)
        evx, evy = sx['evecs'], sy['evecs']
        Bx, By = evx[sx['corr']], evy[sy['corr']]
        dist_x, cx, cy = sx['dist'], sx['corr'], sy['corr']
        eye = torch.eye(evx.shape[1], device=device)

        agg['raw'].append(core.evaluate_pair(evx, evy, Bx, By, dist_x, cx, cy)['geo'])

        Qx, Qy = core.closed_form_rotations(Bx, By)
        agg['oracle'].append(core.geo_error(dist_x, cx, cy, core.recover_p2p(evx @ Qx, evy @ Qy)))

        evx_c, evy_c = corrected_basis(net, sx, feat_key), corrected_basis(net, sy, feat_key)
        res = core.evaluate_pair(evx_c, evy_c, evx_c[cx], evy_c[cy], dist_x, cx, cy)
        agg['net'].append(res['geo'])
        agg['net_align'].append(res['align'])
        agg['net_iddev'].append(res['fmap_id_dev'])
    net.train()
    return {key: float(np.mean(val)) for key, val in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='faust', choices=['faust', 'smal'])
    ap.add_argument('--feature', default='wks', choices=['wks', 'xyz'],
                    help='fixed descriptor that conditions the rotation (xyz breaks '
                         'L/R symmetry; wks is intrinsic)')
    ap.add_argument('--data_root', default=None,
                    help='default ../data/FAUST_r or ../data/SMAL_r by --dataset')
    ap.add_argument('--anchor', type=int, default=0,
                    help='train-shape index to canonicalize against (fixed reference '
                         'frame, Q_ref=I); -1 = random distinct pairs (harder to optimize)')
    ap.add_argument('--net', default='rotnet', choices=['cayley', 'rotnet'],
                    help='rotation predictor: cayley (joint-pipeline flow) or '
                         'rotnet (conditioned expm skew, fits standalone alignment)')
    ap.add_argument('--emb_dim', type=int, default=128, help='rotnet bilinear embedding dim')
    ap.add_argument('--k', type=int, default=100, help='# eigenbasis modes')
    ap.add_argument('--rank', type=int, default=16, help='Cayley generator rank R')
    ap.add_argument('--num_steps', type=int, default=20, help='Cayley integration steps L')
    ap.add_argument('--base_accel', type=float, default=1.0,
                    help='global scale on the Cayley flow generator (larger = bigger rotations)')
    ap.add_argument('--steps', type=int, default=2000, help='training iterations')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--n_eval_pairs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ckpt', default=os.path.join(os.path.dirname(__file__), 'ckpts'))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_root = args.data_root or ('../data/SMAL_r' if args.dataset == 'smal'
                                   else '../data/FAUST_r')
    mode = f'anchor={args.anchor}' if args.anchor >= 0 else 'random-pairs'
    print(f'[stage1] {args.dataset}  k={args.k}  {mode}  device={device}')
    need_wks = (args.feature == 'wks')
    train_shapes = load_shapes(data_root, 'train', k=args.k, with_dist=False,
                               with_wks=need_wks, dataset=args.dataset)
    test_shapes = load_shapes(data_root, 'test', k=args.k, with_dist=True,
                              with_wks=need_wks, dataset=args.dataset)
    n_tr, n_te = len(train_shapes), len(test_shapes)
    feat_key = args.feature
    feat_dim = train_shapes[0][feat_key].shape[1]
    print(f'         {n_tr} train shapes, {n_te} test shapes, '
          f'feature={feat_key} (dim {feat_dim})')

    if args.net == 'rotnet':
        net = ConditionedRotation(
            feature_dim=feat_dim, emb_dim=args.emb_dim, init_scale=args.base_accel,
        ).to(device)
    else:
        net = CayleyONBCorrection(
            feature_dim=feat_dim,
            rank=args.rank, num_steps=args.num_steps, bypass=False,
            base_acceleration=args.base_accel,
            gradient_checkpointing=False,
        ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f'         net={args.net}  params: {sum(p.numel() for p in net.parameters())}')

    # fixed eval pairs (distinct shapes) for a stable learning curve
    eval_pairs = []
    while len(eval_pairs) < args.n_eval_pairs:
        i, j = random.randrange(n_te), random.randrange(n_te)
        if i != j:
            eval_pairs.append((i, j))

    base = evaluate(net, test_shapes, eval_pairs, device, feat_key)   # untrained ~ baseline
    print(f'  [eval @0]   raw {base["raw"]:.4f}   oracle {base["oracle"]:.4f}   '
          f'net {base["net"]:.4f}  (align {base["net_align"]:.2e})')

    # Anchored mode: canonicalize every shape to a FIXED reference frame -- the
    # reference shape's raw basis at its corresponding points (Q_ref = I). This
    # gives each shape a single stable target (Phi_i Q_i[corr_i] -> Phi_ref[corr_ref]),
    # which optimizes far better than random pairs (whose per-shape gradients
    # conflict and collapse Q to identity). Pairwise alignment of two test shapes
    # then follows by both mapping into the shared reference frame.
    ref_target = None
    if args.anchor >= 0:
        sref = to_device(train_shapes[args.anchor], device)
        ref_target = sref['evecs'][sref['corr']].detach()      # (P, k), fixed

    net.train()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        if ref_target is not None:
            i = random.randrange(n_tr)
            while i == args.anchor:
                i = random.randrange(n_tr)
            sx = to_device(train_shapes[i], device)
            opt.zero_grad()
            evx_c = corrected_basis(net, sx, feat_key)
            loss = core.alignment_loss(evx_c[sx['corr']], ref_target)
        else:
            i, j = random.randrange(n_tr), random.randrange(n_tr)
            while i == j:
                j = random.randrange(n_tr)
            sx, sy = to_device(train_shapes[i], device), to_device(train_shapes[j], device)
            opt.zero_grad()
            evx_c = corrected_basis(net, sx, feat_key)
            evy_c = corrected_basis(net, sy, feat_key)
            loss = core.alignment_loss(evx_c[sx['corr']], evy_c[sy['corr']])
        loss.backward()
        opt.step()
        run_loss += loss.item()

        if step % 50 == 0:
            print(f'  step {step:5d}  align_loss {run_loss / 50:.6e}')
            run_loss = 0.0
        if step % args.eval_every == 0:
            ev = evaluate(net, test_shapes, eval_pairs, device, feat_key)
            print(f'  [eval @{step}]  raw {ev["raw"]:.4f}   oracle {ev["oracle"]:.4f}   '
                  f'net {ev["net"]:.4f}  (align {ev["net_align"]:.2e}  '
                  f'|C-I| {ev["net_iddev"]:.2e})')

    os.makedirs(args.ckpt, exist_ok=True)
    out = os.path.join(args.ckpt, f'{args.dataset}_{feat_key}_{args.net}_k{args.k}.pth')
    torch.save({'state_dict': net.state_dict(), 'args': vars(args)}, out)
    print(f'  saved -> {out}')


if __name__ == '__main__':
    main()
