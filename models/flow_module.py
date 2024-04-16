from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)

        self.model = FlowModel(cfg.model)                   # CNF model proper
        self.interpolant = Interpolant(cfg.interpolant)     # handles noising and ODE

        self._interpolant_cfg = cfg.interpolant
        self._exp_cfg = cfg.experiment

        self._valid_sample_write_dir = self._exp_cfg.checkpointer.dirpath
        self._test_sample_write_dir = os.path.join(self._exp_cfg.checkpointer.dirpath, 'test')
        if 'inference' in self._exp_cfg:
            self._output_dir = self._exp_cfg.inference.output_dir
        os.makedirs(self._valid_sample_write_dir, exist_ok=True)
        os.makedirs(self._test_sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics, self.validation_epoch_samples = [], []
        self.test_epoch_metrics, self.test_epoch_samples = [], []
        self.save_hyperparameters() # saves to self.hparams (also in model checkpoints)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    
    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def model_step(self, noisy_batch: Any, stage: str = 'train'):
        # returns losses for a structure batch, as given in FrameDiff paper
        # see also eq. (6) in FrameFlow paper
        # every protein in a batch has the same length
        assert stage in ['train', 'valid', 'test', 'sample'], f'Unknown stage {stage}.'

        if stage == 'train':
            cfg = self._exp_cfg.training
        elif stage == 'valid':
            cfg = self._exp_cfg.validation
        elif stage == 'test':
            cfg = self._exp_cfg.testing
        elif stage == 'sample':
            cfg = self._exp_cfg.sampling
        
        loss_mask = noisy_batch['res_mask']
        if cfg.min_plddt_mask is not None:
            if 'res_plddt' in noisy_batch:
                plddt_mask = noisy_batch['res_plddt'] > cfg.min_plddt_mask
                loss_mask *= plddt_mask
            else:
                self._print_logger.warning(
                    "WARNING: 'res_plddt' not found in noisy_batch. Skipping pLDDT mask application.")
        num_batch, num_res = loss_mask.shape

        # ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # timestep used for normalization
        # NOTE: could also / additionally use the t_rot (if it differs) -> see. Interpolant.corrupt_batch()
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(cfg.t_normalize_clip))
        
        # fwd pass
        if stage == 'train':
            model_output = self.model(noisy_batch)
        else:
            with torch.no_grad():
                model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 # 3 bb atoms per included residue
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * cfg.trans_scale
        trans_loss = cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0] > cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        tot_loss = se3_vf_loss + auxiliary_loss
        if torch.isnan(tot_loss).any():
            raise ValueError('NaN loss encountered')
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "tot_loss": tot_loss
        }

    def process_struc_batch(self, struc_batch, stage):
        assert stage in ['train', 'valid', 'test', 'sample'], f'Unknown stage {stage}.'
        self.interpolant.set_device(struc_batch['res_mask'].device)

        # noise injection
        noisy_batch = self.interpolant.corrupt_batch(struc_batch)

        # generate edge-rep self-conditioning distogram w/ 50% prob, if specified
        # always use it during validation, testing, and sampling (if specified)
        if self._interpolant_cfg.self_condition:
            if stage in ['valid', 'test', 'sample'] or random.random() > 0.5:
                with torch.no_grad():
                    model_sc = self.model(noisy_batch)
                    noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        # forward pass and per-item losses
        batch_losses = self.model_step(noisy_batch, stage=stage)

        return batch_losses, noisy_batch
    
    def loss_agg_and_log(self, batch_losses, noisy_batch, stage):
        assert stage in ['train', 'valid', 'test'], f'Invalid stage {stage}.'

        # mean loss across loss types per item in batch
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = { 
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"{stage}/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # mean loss across time-bin; per loss type
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            f"{stage}/t",
            np.mean(du.to_numpy(t)), # mean sampled t across batch
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"{stage}/{k}", v, prog_bar=False, batch_size=num_batch)
                
        # throughput
        self._log_scalar(
            f"{stage}/length", noisy_batch['res_mask'].shape[1], 
            prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            f"{stage}/batch_size", num_batch, prog_bar=False)
        
        # total loss
        if stage == 'train':
            specified_loss = self._exp_cfg.training.loss
            on_epoch = False
            on_step = True
        elif stage == 'valid':
            specified_loss = self._exp_cfg.validation.loss
            on_epoch = True
            on_step = False
        elif stage == 'test':
            specified_loss = self._exp_cfg.testing.loss
            on_epoch = True
            on_step = False

        loss = (
            total_losses[specified_loss]
            # NOTE: WARNING =======================================================
            # TODO: this was already added in model_step (so we're adding it twice), 
            #       but this is how it was in the original code)
            #       deleting this line and setting aux_loss_weight: 2 in the config  
            #       should be equivalent
            +  total_losses['auxiliary_loss']
            # =====================================================================
        )
        self._log_scalar(f"{stage}/loss", loss, 
                         batch_size=num_batch,
                         on_step=on_step,
                         on_epoch=on_epoch)
        return loss
    
    def struc_step(self, struc_batch, stage='train'):
        """Training-like step (loss computation on structures)."""

        # noising, fwd pass, individual losses
        batch_losses, noisy_batch = self.process_struc_batch(struc_batch, f'{stage}')
        
        # loss logging and aggregation
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        loss = self.loss_agg_and_log(batch_losses, noisy_batch, f'{stage}')

        return loss, num_batch

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        struc_batch, _ = batch # only use structure batch during training

        train_loss, num_batch = self.struc_step(struc_batch, stage='train')

        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        return train_loss
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log('train/epoch_time_minutes', 
                 epoch_time,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False
        )
        self._epoch_start_time = time.time()

    def sample_step(self, len_batch):
        """Inference-like step (generation of new structures)."""

        self.interpolant.set_device(f'cuda:{torch.cuda.current_device()}')
        return self.interpolant.sample(len_batch, self.model)

    def struc_and_sample_step(self, batch: Any, batch_idx: int, stage='valid'):
        """
        Performs a training-like loss computation followed by sample generation
        on a joint batch of known structures and antibody parameters to generate with.
        Logs everything appropriately.
        """
        struc_batch, len_batch = batch
        len_struc_batch, len_len_batch = 0, 0
        loss = None

        # training-like loss computation
        if struc_batch is not None:
            loss, len_struc_batch = self.struc_step(struc_batch, stage=stage)

        # generation-based evaluation
        if len_batch != []:
            pass
            # TODO: implement sensible metrics to evaluate antibody-likeness or protein-likeness

            # samples, projections_traj, _, res_idx = self.sample_step(len_batch)

            # samples = samples[-1].numpy()       # just final state
            # len_len_batch = samples.shape[0]                                                # TODO: check this
            # res_idx = du.to_numpy(res_idx)[0]                                               # TODO: check this

            # if stage == 'valid':
            #     writedir = self._valid_sample_write_dir
            #     samples_list = self.validation_epoch_samples
            #     metrics_list = self.validation_epoch_metrics
            # elif stage == 'test':
            #     writedir = self._test_sample_write_dir
            #     samples_list = self.test_epoch_samples
            #     metrics_list = self.test_epoch_metrics

            # batch_metrics = []
            # for i in range(len_len_batch):

            #     # write out sample to PDB file
            #     final_pos = samples[i]
            #     saved_path = au.write_prot_to_pdb(
            #                     final_pos,
            #                     os.path.join(writedir,
            #                         f'sample_{i}_idx_{batch_idx}_len_{len(res_idx)}.pdb'),
            #                     res_idx=res_idx,
            #                     no_indexing=True # no file indexing
            #     )
            #     if isinstance(self.logger, WandbLogger):
            #         samples_list.append(
            #             [saved_path, self.global_step, wandb.Molecule(saved_path)]
            #         )

            #     # calculate evaluation metrics on batch
            #     mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            #     ca_idx = residue_constants.atom_order['CA']
            #     ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            #     batch_metrics.append((mdtraj_metrics | ca_ca_metrics)) # dictionary merge

            # metrics_list.append(pd.DataFrame(batch_metrics))
        
        return loss, len_struc_batch + len_len_batch
        
    def validation_step(self, batch: Any, batch_idx: int):
        step_start_time = time.time()
        
        val_loss, len_batch = self.struc_and_sample_step(batch, batch_idx, stage='valid')

        step_time = time.time() - step_start_time
        self._log_scalar("valid/examples_per_second", len_batch / step_time)

        return val_loss

    def test_step(self, batch: Any, batch_idx: int):
        step_start_time = time.time()

        test_loss, len_batch = self.struc_and_sample_step(batch, batch_idx, stage='test')

        step_time = time.time() - step_start_time
        self._log_scalar("test/examples_per_second", len_batch / step_time)

        return test_loss

    def end_val_test_epoch(self, stage='valid'):
        if stage == 'valid':
            samples_list = self.validation_epoch_samples
            metrics_list = self.validation_epoch_metrics
        elif stage == 'test':
            samples_list = self.test_epoch_samples
            metrics_list = self.test_epoch_metrics

        if len(samples_list) > 0:
            self.logger.log_table(
                key=f'{stage}/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=samples_list)
            samples_list.clear()

        if len(metrics_list) > 0:
            epoch_metrics = pd.concat(metrics_list)
            for metric_name, metric_val in epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f'{stage}/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(epoch_metrics),
                )
            metrics_list.clear()

    def on_validation_epoch_end(self):
        self.end_val_test_epoch(stage='valid')
    
    def on_test_epoch_end(self):
        self.end_val_test_epoch(stage='test')

    # TODO: may want to modify so that supports multiple samples in parallel
    #       -> would require changes to interpolant.sample() 
    def predict_step(self, batch):
        _, len_batch = batch # ignore any structure batch during inference

        assert len_batch != [], 'No samples to predict on.'
        
        # read off some constants
        len_h, len_l = len_batch['len_h'], len_batch['len_l']
        num_res = len_h + len_l
        sample_length = num_res.item()
        diffuse_mask = torch.ones(1, sample_length)
        sample_id = len_batch['sample_id'].item()
        
        # set up output directory path
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return
        
        # sample
        atom37_traj, model_traj, _, res_idx = self.sample_step(len_batch)

        # write results to file
        os.makedirs(sample_dir, exist_ok=True)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0)) # also puts on cpu
        _ = eu.save_traj(
            bb_traj[-1],        # final state as atom coords
            bb_traj,            # entire trajectory as atom coords
            # trajectory of final-state projections as atom coords
            np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            du.to_numpy(diffuse_mask)[0], # array of 1s (sample-length)
            du.to_numpy(res_idx)[0],      # residue index (sample-length)
            output_dir=sample_dir,
        )

        return