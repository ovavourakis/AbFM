import os
import GPUtil
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.self.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data_module import DataModule
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb

log = eu.get_pyself.logger(__name__) # multi-GPU-friendly python CLI self.logger
torch.set_float32_matmul_precision('high')


class ModelRun:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = DataModule(self._data_cfg)
        self._model: LightningModule = FlowModule(self._cfg)

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

        devices = GPUtil.getAvailable(order='memory', 
                                      limit = 8)[:self._exp_cfg.num_devices]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
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

        ckpt_path = self._exp_cfg.checkpointer.dirpath
        self._model = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg
        )
        self._model.eval()

        devices = GPUtil.getAvailable(order='memory', 
                                      limit = 8)[:self._exp_cfg.num_devices]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
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
        pass
        # TODO: continue from here ------------------------------------

    
    





@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'trvalte_config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    run = ModelRun(cfg=cfg)
    run.train_val()

if __name__ == "__main__":
    main()