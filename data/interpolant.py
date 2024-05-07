import copy
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

from data import so3_utils
from data import utils as du
from data import all_atom

def _centered_gaussian(num_batch, num_res, device):
    
    # TODO: remove this block and remove the generator from below
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    rng = torch.Generator(device=device)

    # (num_res * xyz) gaussian noise for each protein
    # centered at exactly (0,0,0) per protein
    noise = torch.randn(num_batch, num_res, 3, device=device, generator=rng)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    # uniformly randomly sampled rotation *matrices* in SO(3)
    # (one per residue per protein)
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        # isotropic Gaussian-equivalent on SO3 
        # (=truncated-series form of closed-form solution to Wiener diffusion process on SO3)
        # see FrameDiff paper, Appendix E
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch): 
       # theoretically in [0,1)
       # practically in [min_t, 1-min_t]
    #    t = torch.rand(num_batch, device=self._device)
    #    return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

        # TODO: revert this
        return torch.ones((num_batch,), device=self._device) * 0.5

    def _corrupt_trans(self, trans_1, t, res_mask):
        
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)

        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        
        # Kabsch pre-align noise to data to remove global rotation from ODE
        # see section 2.2 of FrameFlow paper  
        # TODO: reinstate
        # trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        
        # linearly interpolate between t=1 (data) and t=t
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        # Kabsch pre-align noise to data to remove global rotation from ODE
        # see section 2.2 of FrameFlow paper
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            # difference between complete noise (t=0) and correct matrix (t=1)
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        # interpolate linearly from 0 to t in tangent space and map back to SO(3)
        # equation (4) in FrameFlow paper
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):       
        
        # NOTE: WARNING !!! ======================================================
        # altered during fiddling with training
        # however, this should not have changed results, because we used all-linear
        # training schedules anyway
        # ======================================================================== 

        # to be called on structure batches only! (not during inference/sampling!)
        noisy_batch = copy.deepcopy(batch)
        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        # [B, 1]
        tau = self.sample_t(num_batch)[:, None]

        # NOTE: WARNING !!! newly added !!! ========================
        t = self.sample_kappa(tau, component='trans')
        # ==========================================================

        noisy_batch['t'] = t

        # apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        # NOTE: WARNING !!! newly added !!! ========================
        t_rot = self.sample_kappa(tau, component='rots')
        noisy_batch['t_rot'] = t_rot
        rotmats_t = self._corrupt_rotmats(rotmats_1, t_rot, res_mask)
        # ==========================================================

        # rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def sample_kappa(self, t, component='rots'):
        # ========================================================================
        # NOTE: WARNING !!! changed from 'sample_schedule' to 'train_schedule' !!!
        # ========================================================================
        if component == 'rots':
            schedule = self._rots_cfg.train_schedule
            if schedule == 'exp':
                exp_rate = self._rots_cfg.exp_rate
        elif component == 'trans':
            schedule = self._trans_cfg.train_schedule
            if schedule == 'exp':
                exp_rate = self._trans_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown component {component}. Must be one of "rots" or "trans".')

        if schedule == 'exp':
            return 1 - torch.exp(-t*exp_rate)
        elif schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid train schedule: {schedule}. Must be one of "exp" or "linear".')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):                                  # TODO: implement non-linear euler step for translations
        if self._trans_cfg.sample_schedule != 'linear':
            raise NotImplementedError(
                "Non-linear euler step for translations not implemented.")
            
        trans_vf = (trans_1 - trans_t) / (1 - t) # v_x in eq. (5) in FrameFlow paper
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            # for v_r in eq. (5) in FrameFlow paper
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            # for v_r in eq. (7) in FrameFlow paper
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    # TODO: consider implementing a better integrator (not Euler) - could maybe use scipy.integrate.odeint?
    # TODO: may want to alter so can handle multiple samples of different length in batch - padding in res_mask (and elsewhere?)
    def sample(self, batch, model):
        """
        This method is intended to be called on ab-parameter batches 
        during inference only. Each batch contains a single sample.
        """
        len_h, len_l = batch['len_h'].item(), batch['len_l'].item()
        num_res, num_batch = len_h + len_l, batch['len_h'].shape[0]

        idx_h = torch.arange(len_h, device=self._device) + 1
        idx_l = torch.arange(len_l, device=self._device) + 1001
        res_idx = torch.cat([idx_h, idx_l])
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        batch = {'res_idx': res_idx,    
             'res_mask': res_mask,
        }
        # start with a sample from the prior distribution
        # NOTE: during sampling we start from uniform on SO3
        #       whereas training used IGSO3
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        # set up time grid for integration
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]
        # propagate system forward in time
        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:
            # run model
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)
            # process model output
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1
            ''' 
            NOTE: We re-parametrised the loss s.t. the model predicts x_{t=1} and r_{t=1}, 
            rather than the vector fields v_x and v_r, going from eq. (4.5) to eq. (6) in FrameFlow paper.
            This means that during sampling, when we actually *need* the vector fields, because
            
            d/dt phi(x) = v(phi(x), t)
            
            we need to back-compute them from the final-t prediction using eq. (5) [or (7)] in FrameFlow paper.

            The re-parametrisation was carried out to be able to easily apply the auxiliary structural losses
            during training. See FlowModule->model_step() in models/flow_module.py for more details.
            It also means that we can easily see how well we've learnt the OT path by 
            looking at the final-t prediction from early sampling steps.
            We could potentially even truncate the sampling after a few steps and use the final-t prediction,
            if it's good enough.
            '''
            # take step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # NOTE: WARNING !!! ======================================================
        # I don't think this final step is necessary for FM, just diffusion.
        # ========================================================================

        # We integrated to 1, but all we have is a projection from 1-eps to 1.
        # now want to do one final predictions of x1 | x1_proj
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        
        return atom37_traj, clean_atom37_traj, clean_traj, res_idx