import hydra

from model_run import *
import experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="inference.yaml")
def sample(cfg: DictConfig) -> None:
    # load train/val/test config from checkpoint
    ckpt_path = cfg.experiment.inference.ckpt_path
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'trvalte_config.yaml'))

    # over-write relevant fields with inference config
    cfg = eu.merge_configs(ckpt_cfg, cfg)
    cfg.experiment.checkpointer.dirpath = './'

    run = ModelRun(cfg=cfg, stage='sample')
    run.sample()

if __name__ == "__main__":
    sample()