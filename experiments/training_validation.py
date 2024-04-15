import hydra

from model_run import *
import experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def train_val(cfg: DictConfig):
    cfg = eu.load_warmstart_config(cfg)
    run = ModelRun(cfg=cfg, stage='train')
    run.train_val()

if __name__ == "__main__":
    train_val()