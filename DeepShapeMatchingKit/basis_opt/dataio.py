"""Load FAUST shapes (eigenbasis, mass, WKS, GT correspondences) for the
direct basis-optimization experiments. Reuses the kit's cached spectral
operators -- nothing is recomputed if the diffusion cache exists.
"""

import numpy as np
import torch

from datasets.shape_dataset import SingleFaustDataset, SingleSmalDataset
from utils.geometry_util import compute_wks_autoscale


def _wks(evals, evecs, mass, n_descr=128, n_eig=128):
    """WKS descriptor (N, n_descr) from this shape's spectrum. Fixed, not learned."""
    feat = compute_wks_autoscale(evals[None], evecs[None], mass[None],
                                 n_descr=n_descr, n_eig=n_eig)
    return feat.squeeze(0)


def _build_dataset(dataset, data_root, phase, num_evecs, with_dist):
    """FAUST (near-isometric) or SMAL (non-isometric) single-shape dataset.

    Both expose per-shape GT correspondences to a shared template, so the
    alignment objective `Phi_x[corr_x] Q_x == Phi_y[corr_y] Q_y` is identical;
    SMAL just makes the pairs non-isometric (different species, same template).
    """
    common = dict(data_root=data_root, phase=phase, return_faces=True,
                  return_corr=True, return_evecs=True, num_evecs=num_evecs,
                  return_dist=with_dist)
    if dataset == 'faust':
        return SingleFaustDataset(**common)
    if dataset == 'smal':
        # category=True -> use {train,test}_cat.txt (cross-category pairs allowed)
        return SingleSmalDataset(category=True, **common)
    raise ValueError(f'unknown dataset {dataset!r} (expected faust|smal)')


def load_shapes(data_root, phase, k, num_evecs=200, with_dist=False,
                with_wks=True, indices=None, dataset='faust'):
    """Return a list of per-shape dicts of CPU tensors.

    Args:
        data_root: e.g. '../data/FAUST_r' or '../data/SMAL_r'.
        phase:     'train' / 'test'.
        k:         number of eigenbasis modes to keep for the correction.
        num_evecs: how many eigenpairs to pull from cache (>= max(k,128)).
        with_dist: also load the geodesic distance matrix (needed for geo error).
        with_wks:  also compute the WKS descriptor.
        indices:   optional subset of shape indices within the phase.
        dataset:   'faust' or 'smal'.

    Each dict has: name, evecs (N,k), evals (k), mass (N), corr (P)[, wks (N,128)]
    [, dist (N,N) numpy].
    """
    num_evecs = max(num_evecs, k, 128)
    ds = _build_dataset(dataset, data_root, phase, num_evecs, with_dist)

    idxs = range(len(ds)) if indices is None else indices
    shapes = []
    for i in idxs:
        item = ds[i]
        evecs = item['evecs'][:, :k].contiguous().float()      # (N, k)
        evals = item['evals'][:k].contiguous().float()         # (k,)
        mass = item['mass'].contiguous().float()               # (N,)
        # xyz coordinate "descriptor": centered + scaled to unit radius. Unlike
        # WKS this is EXTRINSIC (breaks left/right symmetry, carries pose/shape),
        # so it can condition the basis rotation where intrinsic WKS cannot.
        xyz = item['xyz'].float()
        xyz = xyz - xyz.mean(dim=0, keepdim=True)
        xyz = xyz / (xyz.norm(dim=1).max() + 1e-8)
        rec = {
            'name': item['name'],
            'evecs': evecs,
            'evals': evals,
            'mass': mass,
            'corr': item['corr'].long(),                       # (P,)
            'xyz': xyz.contiguous(),                           # (N, 3)
        }
        if with_wks:
            # WKS is computed from the first 128 modes regardless of k.
            rec['wks'] = _wks(item['evals'][:128].float(),
                              item['evecs'][:, :128].float(),
                              mass).contiguous()                # (N, 128)
        if with_dist:
            rec['dist'] = np.asarray(item['dist'], dtype=np.float32)   # (N, N)
        shapes.append(rec)
    return shapes


def to_device(rec, device):
    """Move a shape dict's torch tensors to device (leaves 'dist'/'name' as-is)."""
    out = {}
    for key, val in rec.items():
        out[key] = val.to(device) if torch.is_tensor(val) else val
    return out
