import os
import wandb
import GPUtil
from omegaconf import DictConfig, OmegaConf

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data_module import DataModule
from models.flow_module import FlowModule
from experiments import utils as eu

log = eu.get_pylogger(__name__) # multi-GPU-friendly python CLI logger
torch.set_float32_matmul_precision('high')

class ModelRun:

    def __init__(self, *, cfg: DictConfig, stage='train'):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment

        # initialise data module
        self._datamodule: LightningDataModule = DataModule(self._data_cfg)
        # initialise model
        if stage == 'train': 
            if self._exp_cfg.warm_start_cfg_override and self._exp_cfg.warm_start_weights is not None:
                self._model: LightningModule = FlowModule.load_from_checkpoint(
                    checkpoint_path=self._exp_cfg.warm_start_weights,
                    cfg=self._cfg,
                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())
                )
            else:
                self._model: LightningModule = FlowModule(self._cfg)
        else:
            if stage == 'test':
                ckpt_dir = self._exp_cfg.checkpointer.dirpath
                self.ckpt_name = os.path.join(ckpt_dir, 
                                              self._exp_cfg.testing.ckpt_name)
            elif stage == 'sample':
                self.ckpt_name =self._exp_cfg.inference.ckpt_path
            
            self._model: LightningModule = FlowModule.load_from_checkpoint(
                    checkpoint_path=self.ckpt_name,
                    cfg=self._cfg,
                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())
            )
            self._model.eval()

    def setup_log_and_debug(self):
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            self.logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.module.loaders.num_workers = 0
        else:
            self.logger = WandbLogger(**self._exp_cfg.wandb)

    def train_val(self):
        self.setup_log_and_debug()
 
        # set up checkpoint directory
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints and test samples will be saved to {ckpt_dir}.")
        
        # make sure we checkpoint the model
        callbacks = [ModelCheckpoint(**self._exp_cfg.checkpointer)]
        
        # save train-val-test config to file and to wandb
        cfg_path = os.path.join(ckpt_dir, 'trvalte_config.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        if isinstance(self.logger.experiment.config, wandb.sdk.wandb_config.Config):
            self.logger.experiment.config.update(flat_cfg)

        devices = GPUtil.getAvailable(order='memory', limit=self._exp_cfg.num_devices)
        log.info(f"Using devices: {devices}")
        if len(devices) > 1:
            self._exp_cfg.trainer.accumulate_grad_batches = self._exp_cfg.trainer.accumulate_grad_batches//len(devices)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            # detect_anomaly=True,
            # num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=self.logger,
            use_distributed_sampler=False,  # parallelism handled by CombinedDatasetBatchSampler internally
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

    def test(self):
        self.setup_log_and_debug()
        devices = GPUtil.getAvailable(order='memory', limit=self._exp_cfg.num_devices)
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
            # num_sanity_val_steps=0,
            logger=self.logger,
            use_distributed_sampler=False,  # parallelism handled by CombinedDatasetBatchSampler internally
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )
        trainer.test(
            model=self._model,
            datamodule=self._datamodule,        
        )
    
    def sample(self):
        self.setup_log_and_debug()

        # set-up directories to write samples and config to
        os.makedirs(self._exp_cfg.inference.output_dir, exist_ok=True)
        config_path = os.path.join(self._exp_cfg.inference.output_dir, 'sample_config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saved config and samples to {self._exp_cfg.inference.output_dir}')

        devices = GPUtil.getAvailable(order='memory', limit=self._exp_cfg.num_devices)
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
            # num_sanity_val_steps=0,
            logger=self.logger,
            use_distributed_sampler=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )
        trainer.predict(
            model=self._model,
            datamodule=self._datamodule,        
        )