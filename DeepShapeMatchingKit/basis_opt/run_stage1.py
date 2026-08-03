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
from basis_opt import core
from basis_opt import downstream
from basis_opt.dataio import load_shapes, to_device


def corrected_basis(net, shape, feat_key):
    """Phi~ = Phi Q with Q predicted from this shape's fixed `feat_key` descriptor."""
    return net(shape['evecs'], shape['mass'], shape['evals'], shape[feat_key])


def corrected_basis_qd(net, shape, feat_key, metric):
    """Phi~ = Phi Q D  (per-shape orthonormal Q from `net`, shared metric D)."""
    return metric(corrected_basis(net, shape, feat_key))


def _pair_loss(loss_mode, net, feat_key, sx, sy, sy_is_anchor=False,
              lambda_fmap=1e-1, resolvant_gamma=0.5,
              hybrid_primary='fmap', hybrid_weight=0.01):
    """Training loss for one (sx, sy) pair, in any of point/fmap/hybrid mode.

    sy_is_anchor=True means sy is the FIXED reference shape used by anchored
    training (Q_y = I, sy is never passed through `net`) -- otherwise both
    shapes' rotations come from `net`. This one helper covers both the
    anchored and random-pair training loops so the loss-mode swap is
    controlled: same optimizer, same data, same rank/steps either way.

    Qx (and Qy, unless sy_is_anchor) are computed via ONE call each to
    `net.compute_rotation` and REUSED for whichever loss term(s) are needed.
    This matters: CayleyONBCorrection.compute_rotation calls
    shuffle_model_state(), which in train() mode resamples the Cayley flow's
    ODE time-span RANDOMLY on every call. Computing Q separately for a point
    term and a fmap term (e.g. via two `net(...)` / `net.compute_rotation(...)`
    calls) would silently use two DIFFERENT random Q's for the "same" step --
    wasteful (double flow evaluation) and inconsistent (a hybrid loss must add
    two terms evaluated at the SAME Q to mean anything).

    'point_align'    : core.alignment_loss (point-space, the baseline).
    'fmap_supervised': core.functional_map_diagnostic_loss (fmap-space,
                       diagnostic -- see core.py docstring for what it
                       localizes).
    'hybrid'         : primary loss (hybrid_primary) + hybrid_weight * the
                       other loss, both evaluated at the SAME Qx, Qy. Tests
                       whether a small dose of the OTHER supervision snaps
                       training out of a degenerate valley -- if so, the pure
                       single-loss failure was under-constraint, not pure
                       architectural incapacity.
    """
    Qx = net.compute_rotation(sx['evecs'], sx['mass'], sx['evals'], sx[feat_key])
    if sy_is_anchor:
        Qy = torch.eye(Qx.shape[0], device=Qx.device, dtype=Qx.dtype)
    else:
        Qy = net.compute_rotation(sy['evecs'], sy['mass'], sy['evals'], sy[feat_key])

    def point_loss():
        evx_c, evy_c = sx['evecs'] @ Qx, sy['evecs'] @ Qy
        return core.alignment_loss(evx_c[sx['corr']], evy_c[sy['corr']])

    def fmap_loss():
        return core.functional_map_diagnostic_loss(
            sx['evecs'], sy['evecs'], Qx, Qy, sx['mass'], sy['mass'],
            sx['corr'], sy['corr'], sx[feat_key], sy[feat_key],
            sx['evals'], sy['evals'], lambda_=lambda_fmap,
            resolvant_gamma=resolvant_gamma)

    if loss_mode == 'point_align':
        return point_loss()
    if loss_mode == 'fmap_supervised':
        return fmap_loss()
    if loss_mode == 'hybrid':
        pl, fl = point_loss(), fmap_loss()
        return (fl + hybrid_weight * pl) if hybrid_primary == 'fmap' \
            else (pl + hybrid_weight * fl)

    raise ValueError(f'unknown loss_mode {loss_mode!r}')


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


@torch.no_grad()
def evaluate_downstream(net, metric, test_shapes, pairs, device, feat_key):
    """fmap_downstream eval: geo-error of the DSMK-solver pipeline, plus the
    requested rotation diagnostics.

    Reports, averaged over `pairs`:
        raw_ds  : downstream geo with the plain LBO basis (Q=I, D=I) -- the
                  classical-fmap baseline this whole mode must beat.
        net_ds  : downstream geo with the learned Phi~ = Phi Q D.
        oracle  : NN-in-basis geo of the Procrustes oracle Q* (the OTHER
                  pipeline's ceiling, kept as an external reference point).
        q_dev   : mean ||Q - I||           (is Q moving?)
        q_orth  : mean ||Q^T Q - I||       (orthonormality mechanism intact?)
        d_cond  : condition number of D    (1.0 => D=I; >1 => D doing non-orth work)
        cos_q   : mean cosine(vec Q_net, vec Q_oracle*)  (gauge-ambiguous in
                  random-pair mode -- logged as requested, read with care)
    """
    net.eval()
    agg = {k: [] for k in ('raw_ds', 'net_ds', 'oracle', 'q_dev', 'q_orth', 'cos_q')}
    for ix, iy in pairs:
        sx, sy = to_device(test_shapes[ix], device), to_device(test_shapes[iy], device)
        evx, evy = sx['evecs'], sy['evecs']
        mx, my = sx['mass'], sy['mass']
        wx, wy = sx[feat_key], sy[feat_key]
        ex, ey = sx['evals'], sy['evals']
        dist_x, cx, cy = sx['dist'], sx['corr'], sy['corr']
        K = evx.shape[1]
        eye = torch.eye(K, device=device)

        # raw downstream (plain LBO basis) = classical fmap baseline
        p_raw = downstream.downstream_p2p(evx, evy, mx, my, wx, wy, ex, ey)
        agg['raw_ds'].append(core.geo_error(dist_x, cx, cy, p_raw))

        # learned Phi~ = Phi Q D downstream
        evx_c = corrected_basis_qd(net, sx, feat_key, metric)
        evy_c = corrected_basis_qd(net, sy, feat_key, metric)
        p_net = downstream.downstream_p2p(evx_c, evy_c, mx, my, wx, wy, ex, ey)
        agg['net_ds'].append(core.geo_error(dist_x, cx, cy, p_net))

        # external reference: NN-in-basis Procrustes oracle
        Qx_o, Qy_o = core.closed_form_rotations(evx[cx], evy[cy])
        agg['oracle'].append(core.geo_error(
            dist_x, cx, cy, core.recover_p2p(evx @ Qx_o, evy @ Qy_o)))

        # rotation diagnostics (Q only; D handled by d_cond below)
        Qx = net.compute_rotation(evx, mx, ex, wx)
        agg['q_dev'].append((Qx - eye).norm().item())
        agg['q_orth'].append((Qx.t() @ Qx - eye).norm().item())
        cos = torch.dot(Qx.flatten(), Qx_o.flatten()) / (
            Qx.flatten().norm() * Qx_o.flatten().norm() + 1e-12)
        agg['cos_q'].append(cos.item())
    net.train()
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out['d_cond'] = metric.cond_number()
    return out


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
    ap.add_argument('--k', type=int, default=100, help='# eigenbasis modes')
    ap.add_argument('--rank', type=int, default=16, help='Cayley generator rank R')
    ap.add_argument('--num_steps', type=int, default=20, help='Cayley integration steps L')
    ap.add_argument('--base_accel', type=float, default=1.0,
                    help='global scale on the Cayley flow generator (larger = bigger rotations)')
    ap.add_argument('--eval_mode', default='nn_in_basis',
                    choices=['nn_in_basis', 'fmap_downstream'],
                    help='nn_in_basis: rotation IS the matcher (existing baseline). '
                         'fmap_downstream: rotation SERVES a DSMK fmap solver -- '
                         'Phi~ = Phi Q D, solve C, fmap2pointmap, geo. Train by a '
                         'contrastive downstream loss. See basis_opt/downstream.py.')
    ap.add_argument('--use_shared_metric', default='none',
                    choices=['none', 'diag', 'lowrank'],
                    help='fmap_downstream only: shared non-orthonormal metric D '
                         '(the one knob the solver cannot absorb). none=identity.')
    ap.add_argument('--metric_rank', type=int, default=8,
                    help='rank for --use_shared_metric lowrank')
    ap.add_argument('--freeze_q', action='store_true',
                    help='fmap_downstream: keep Q=I (CayleyONBCorrection bypass) and '
                         'train ONLY the shared metric D -- isolates D\'s effect')
    ap.add_argument('--contrastive_samples', type=int, default=512,
                    help='fmap_downstream: # GT corr points per contrastive step')
    ap.add_argument('--contrastive_tau', type=float, default=0.1,
                    help='fmap_downstream: InfoNCE temperature')
    ap.add_argument('--loss_mode', default='point_align',
                    choices=['point_align', 'fmap_supervised', 'hybrid'],
                    help='point_align: core.alignment_loss (baseline, point space). '
                         'fmap_supervised: core.functional_map_diagnostic_loss '
                         '(diagnostic, functional-map space). hybrid: '
                         'hybrid_primary loss + hybrid_weight * the other -- see '
                         'core.py / _pair_loss docstrings')
    ap.add_argument('--hybrid_primary', default='fmap', choices=['fmap', 'point'],
                    help='hybrid mode only: which loss is the main term (weight 1.0)')
    ap.add_argument('--hybrid_weight', type=float, default=0.01,
                    help='hybrid mode only: weight on the SECONDARY (non-primary) loss')
    ap.add_argument('--lambda_fmap', type=float, default=1e-1,
                    help='resolvent regularization for the fmap_supervised loss\'s C solve')
    ap.add_argument('--resolvant_gamma', type=float, default=0.5,
                    help='resolvent mask exponent for the fmap_supervised loss\'s C solve')
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

    downstream_mode = (args.eval_mode == 'fmap_downstream')
    data_root = args.data_root or ('../data/SMAL_r' if args.dataset == 'smal'
                                   else '../data/FAUST_r')
    mode = f'anchor={args.anchor}' if args.anchor >= 0 else 'random-pairs'
    sig = (f'eval={args.eval_mode} metric={args.use_shared_metric} '
           f'freeze_q={args.freeze_q}' if downstream_mode
           else f'loss_mode={args.loss_mode}')
    print(f'[stage1] {args.dataset}  k={args.k}  {mode}  {sig}  device={device}')
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

    # In fmap_downstream, --freeze_q keeps Q=I (bypass) so only the shared metric
    # D is trained -- isolating D, the one transform the solver cannot absorb.
    bypass_q = downstream_mode and args.freeze_q
    net = CayleyONBCorrection(
        feature_dim=feat_dim,
        rank=args.rank, num_steps=args.num_steps, bypass=bypass_q,
        base_acceleration=args.base_accel,
        gradient_checkpointing=False,
    ).to(device)
    metric = downstream.SharedMetric(
        args.k, kind=(args.use_shared_metric if downstream_mode else 'none'),
        rank=args.metric_rank).to(device)
    params = list(net.parameters()) + list(metric.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f'         params: net={sum(p.numel() for p in net.parameters())} '
          f'metric[{metric.kind}]={sum(p.numel() for p in metric.parameters())}')

    # fixed eval pairs (distinct shapes) for a stable learning curve
    eval_pairs = []
    while len(eval_pairs) < args.n_eval_pairs:
        i, j = random.randrange(n_te), random.randrange(n_te)
        if i != j:
            eval_pairs.append((i, j))

    def run_eval(step):
        if downstream_mode:
            ev = evaluate_downstream(net, metric, test_shapes, eval_pairs, device, feat_key)
            print(f'  [eval @{step}]  raw_ds {ev["raw_ds"]:.4f}   net_ds {ev["net_ds"]:.4f}   '
                  f'oracle(nn) {ev["oracle"]:.4f}   |Q-I| {ev["q_dev"]:.3f}  '
                  f'|QtQ-I| {ev["q_orth"]:.1e}  cond(D) {ev["d_cond"]:.3f}  '
                  f'cos(Q,Q*) {ev["cos_q"]:.3f}')
            return ev
        ev = evaluate(net, test_shapes, eval_pairs, device, feat_key)
        print(f'  [eval @{step}]  raw {ev["raw"]:.4f}   oracle {ev["oracle"]:.4f}   '
              f'net {ev["net"]:.4f}  (align {ev["net_align"]:.2e}'
              + (f'  |C-I| {ev["net_iddev"]:.2e}' if step else '') + ')')
        return ev

    run_eval(0)   # untrained ~ baseline

    # Anchored mode: canonicalize every shape to a FIXED reference frame -- the
    # reference shape's raw basis at its corresponding points (Q_ref = I). This
    # gives each shape a single stable target (Phi_i Q_i[corr_i] -> Phi_ref[corr_ref]),
    # which optimizes far better than random pairs (whose per-shape gradients
    # conflict and collapse Q to identity). Pairwise alignment of two test shapes
    # then follows by both mapping into the shared reference frame.
    # fmap_downstream is inherently pairwise (the solver needs a pair), so it
    # always uses random distinct pairs -- the anchored reference frame is a
    # nn_in_basis-only construct.
    ref_shape = (to_device(train_shapes[args.anchor], device)
                 if (args.anchor >= 0 and not downstream_mode) else None)

    def downstream_step():
        i, j = random.randrange(n_tr), random.randrange(n_tr)
        while i == j:
            j = random.randrange(n_tr)
        sx, sy = to_device(train_shapes[i], device), to_device(train_shapes[j], device)
        evx_c = corrected_basis_qd(net, sx, feat_key, metric)
        evy_c = corrected_basis_qd(net, sy, feat_key, metric)
        return downstream.downstream_contrastive_loss(
            evx_c, evy_c, sx['mass'], sy['mass'], sx[feat_key], sy[feat_key],
            sx['evals'], sy['evals'], sx['corr'], sy['corr'],
            n_samples=args.contrastive_samples, tau=args.contrastive_tau)

    net.train()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        if downstream_mode:
            opt.zero_grad()
            loss = downstream_step()
        elif ref_shape is not None:
            i = random.randrange(n_tr)
            while i == args.anchor:
                i = random.randrange(n_tr)
            sx = to_device(train_shapes[i], device)
            opt.zero_grad()
            loss = _pair_loss(args.loss_mode, net, feat_key, sx, ref_shape,
                              sy_is_anchor=True, lambda_fmap=args.lambda_fmap,
                              resolvant_gamma=args.resolvant_gamma,
                              hybrid_primary=args.hybrid_primary,
                              hybrid_weight=args.hybrid_weight)
        else:
            i, j = random.randrange(n_tr), random.randrange(n_tr)
            while i == j:
                j = random.randrange(n_tr)
            sx, sy = to_device(train_shapes[i], device), to_device(train_shapes[j], device)
            opt.zero_grad()
            loss = _pair_loss(args.loss_mode, net, feat_key, sx, sy,
                              sy_is_anchor=False, lambda_fmap=args.lambda_fmap,
                              resolvant_gamma=args.resolvant_gamma,
                              hybrid_primary=args.hybrid_primary,
                              hybrid_weight=args.hybrid_weight)
        loss.backward()
        opt.step()
        run_loss += loss.item()

        if step % 50 == 0:
            if downstream_mode:
                tag = 'contrastive'
            else:
                tag = (f'hybrid[{args.hybrid_primary}+{args.hybrid_weight:g}*'
                      f'{"point" if args.hybrid_primary == "fmap" else "fmap"}]'
                      if args.loss_mode == 'hybrid' else args.loss_mode)
            print(f'  step {step:5d}  {tag}_loss {run_loss / 50:.6e}')
            run_loss = 0.0
        if step % args.eval_every == 0:
            run_eval(step)

    if downstream_mode:
        loss_tag = (f'fmap_downstream-{args.use_shared_metric}'
                    + ('-freezeQ' if args.freeze_q else '-Q'))
    else:
        loss_tag = (f'hybrid-{args.hybrid_primary}-w{args.hybrid_weight:g}'
                    if args.loss_mode == 'hybrid' else args.loss_mode)
    os.makedirs(args.ckpt, exist_ok=True)
    out = os.path.join(args.ckpt,
                       f'{args.dataset}_{feat_key}_onb_k{args.k}_r{args.rank}_{loss_tag}.pth')
    torch.save({'state_dict': net.state_dict(),
                'metric_state_dict': metric.state_dict(), 'args': vars(args)}, out)
    print(f'  saved -> {out}')


if __name__ == '__main__':
    main()
