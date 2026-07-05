import numpy as np
import torch
import torch.nn.functional as F

from .ulrssm_model import ULRSSM_Model
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils import get_root_logger

# Importing the wrapper here both (a) makes CayleyONBCorrection available and
# (b) runs its @NETWORK_REGISTRY.register() so build_network can resolve it.
# This module is auto-scanned by models/__init__.py, so the registration fires
# without editing any existing kit file.
from networks.onb.correction import CayleyONBCorrection  # noqa: F401


@MODEL_REGISTRY.register()
class ONB_ULRSSM_Model(ULRSSM_Model):
    """ULRSSM with a learned coefficient-space orthonormal-basis correction.

    A single shared `CayleyONBCorrection` (network key ``onb_correction``) is
    applied to both shapes' raw eigenbases from their DiffusionNet features,
    producing Φ̃ = Φ Q with Q a K×K orthonormal rotation. The mass-weighted
    projector ``evecs_trans = Φ̃ᵀ diag(mass)`` and ``Mk = Φ̃ᵀ diag(mass) Φ̃`` are
    rebuilt from the corrected basis; everything downstream (fmap solver,
    SURFMNet loss, l_align) is the vanilla ULRSSM pipeline, unchanged.

    Coefficient-space invariant: because Q is Euclidean-orthonormal and Φ
    already satisfies ΦᵀMΦ = I, the corrected Mk = QᵀIQ = I. So SURFMNetLoss
    (whose orthogonality term assumes Mk = I) stays correct, and with the
    correction's ``bypass=True`` the whole pipeline reduces to vanilla ULRSSM.
    """

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data=data_x)  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data=data_y)  # [B, Ny, C]

        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        mass_x = data_x['mass']
        mass_y = data_y['mass']

        # ── learned orthonormal-basis correction (one shared instance) ──────────
        onb = self.networks['onb_correction']
        evecs_x_raw, evecs_y_raw = evecs_x, evecs_y           # keep Φ for the Q penalty
        evecs_x = onb(evecs_x, mass_x, evals_x, feat_x)  # Φ̃_x [B, Nx, K]
        evecs_y = onb(evecs_y, mass_y, evals_y, feat_y)  # Φ̃_y [B, Ny, K]

        # rebuild the mass-weighted projector evecs_trans = Φ̃ᵀ diag(mass)
        evecs_trans_x = evecs_x.transpose(1, 2) * mass_x.unsqueeze(1)  # [B, K, Nx]
        evecs_trans_y = evecs_y.transpose(1, 2) * mass_y.unsqueeze(1)  # [B, K, Ny]

        # rebuild Mk = Φ̃ᵀ diag(mass) Φ̃ (= I in coefficient space; kept for
        # sanity / future HS-loss & vertex-space variants).
        self.Mk_x = torch.bmm(evecs_trans_x, evecs_x)  # [B, K, K]
        self.Mk_y = torch.bmm(evecs_trans_y, evecs_y)  # [B, K, K]

        # fmap solver (unchanged)
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # SURFMNet loss (unchanged; valid because Mk = I in coefficient space)
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

        # permutation + closed-form alignment loss (unchanged, but on Φ̃)
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)
        if not self.partial:
            Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
            self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['operators']['L'], data_y['operators']['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)

        # ── optional Q→identity regularizer ─────────────────────────────────────
        # The unsupervised SURFMNet objective is too weak to supervise a free
        # learnable basis: Q can lower the loss by rotating the basis without
        # improving correspondence (proxy/goal divergence — wks ONB reached a
        # *lower* loss than baseline yet 3x worse geo-error). This term pulls the
        # learned rotation back toward I so low loss again implies good matching.
        # Q = Φ_rawᵀ M Φ̃  (since Φ̃ = Φ_raw Q and Φ_rawᵀ M Φ_raw = I). Weight 0
        # (default) is an exact no-op, so existing configs are unaffected.
        q_reg_w = self.opt['train'].get('q_identity_reg_weight', 0.0)
        if q_reg_w and q_reg_w > 0:
            B, _, K = evecs_x.shape
            eye = torch.eye(K, device=evecs_x.device, dtype=evecs_x.dtype)
            Qx = (evecs_x_raw.transpose(1, 2) * mass_x.unsqueeze(1)) @ evecs_x  # [B, K, K]
            Qy = (evecs_y_raw.transpose(1, 2) * mass_y.unsqueeze(1)) @ evecs_y  # [B, K, K]
            l_qreg = ((Qx - eye).pow(2).sum() + (Qy - eye).pow(2).sum()) / B
            self.loss_metrics['l_qreg'] = q_reg_w * l_qreg

    def validate_single(self, data, timer):
        """Mirror of ULRSSM_Model.validate_single, but the same shared basis
        correction is applied before the basis is used — otherwise a trained
        bypass=False model would be scored against the uncorrected basis.

        In eval mode (base_model.validation() calls self.eval() first) the
        isometry's shuffle_model_state takes the deterministic `linspace`
        branch, so test-time Q is fixed, not resampled per call. Under
        bypass=True the correction is identity, so this reproduces vanilla.
        """
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement (uses the overridden feed_data, so already corrected)
        if self.with_refine > 0:
            self.refine(data)

        # feature extractor
        feat_x = self.networks['feature_extractor'](data=data_x)
        feat_y = self.networks['feature_extractor'](data=data_y)

        # get spectral operators (unbatched, matching the parent's .squeeze())
        evecs_x_raw = data_x['evecs'].squeeze()
        evecs_y_raw = data_y['evecs'].squeeze()
        mass_x = data_x['mass'].squeeze()
        mass_y = data_y['mass'].squeeze()
        evals_x = data_x['evals'].squeeze()
        evals_y = data_y['evals'].squeeze()

        # ── learned orthonormal-basis correction (same shared instance) ─────────
        onb = self.networks['onb_correction']
        evecs_x = onb(evecs_x_raw, mass_x, evals_x, feat_x.squeeze(0))  # Φ̃_x [Nx, K]
        evecs_y = onb(evecs_y_raw, mass_y, evals_y, feat_y.squeeze(0))  # Φ̃_y [Ny, K]
        # rebuild the mass-weighted projector from the corrected basis
        evecs_trans_x = evecs_x.t() * mass_x[None]                  # [K, Nx]
        evecs_trans_y = evecs_y.t() * mass_y[None]                  # [K, Ny]

        # ── diagnostics (accumulated by validation()): is the correction active
        #    (max|Q - I|) and staying mass-orthonormal (max|Φ̃ᵀMΦ̃ - I|)? ────────
        if hasattr(self, '_diag_q'):
            with torch.no_grad():
                eye = torch.eye(evecs_x.shape[1], device=evecs_x.device, dtype=evecs_x.dtype)
                # Q = Φ_rawᵀ M Φ̃  (since Φ̃ = Φ_raw Q and Φ_rawᵀ M Φ_raw = I)
                Qx = (evecs_x_raw.t() * mass_x[None]) @ evecs_x
                Qy = (evecs_y_raw.t() * mass_y[None]) @ evecs_y
                q_dev = max((Qx - eye).abs().max().item(), (Qy - eye).abs().max().item())
                orth_dev = max((evecs_trans_x @ evecs_x - eye).abs().max().item(),
                               (evecs_trans_y @ evecs_y - eye).abs().max().item())
                self._diag_q.append(q_dev)
                self._diag_orth.append(orth_dev)

        if self.non_isometric:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()

            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze()
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)

            # convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # reset per-validation diagnostic accumulators (populated in validate_single)
        self._diag_q = []
        self._diag_orth = []

        # run the standard ULRSSM validation (geodesic error, pck, logging)
        super(ONB_ULRSSM_Model, self).validation(dataloader, tb_logger, update)

        # log the two ONB diagnostic scalars every validation step
        if len(self._diag_q) > 0:
            q_dev_mean = float(np.mean(self._diag_q))
            q_dev_max = float(np.max(self._diag_q))
            orth_dev_mean = float(np.mean(self._diag_orth))
            orth_dev_max = float(np.max(self._diag_orth))

            logger = get_root_logger()
            logger.info(f'[ONB diag] max|Q-I|: mean={q_dev_mean:.4e} max={q_dev_max:.4e} | '
                        f'max|Φ̃ᵀMΦ̃-I|: mean={orth_dev_mean:.4e} max={orth_dev_max:.4e}')

            step = self.curr_iter // self.opt['val']['val_freq'] if self.is_train else 0
            if tb_logger is not None:
                tb_logger.add_scalar('val_onb/q_dev_mean', q_dev_mean, global_step=step)
                tb_logger.add_scalar('val_onb/q_dev_max', q_dev_max, global_step=step)
                tb_logger.add_scalar('val_onb/orth_dev_mean', orth_dev_mean, global_step=step)
                tb_logger.add_scalar('val_onb/orth_dev_max', orth_dev_max, global_step=step)
            else:
                try:
                    import wandb
                    wandb.log({'onb/q_dev_mean': q_dev_mean, 'onb/q_dev_max': q_dev_max,
                               'onb/orth_dev_mean': orth_dev_mean, 'onb/orth_dev_max': orth_dev_max})
                except Exception:
                    pass
