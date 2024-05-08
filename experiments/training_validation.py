import hydra

from model_run import *
import experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def train_val(cfg: DictConfig):
    cfg = eu.load_warmstart_config(cfg)
    run = ModelRun(cfg=cfg, stage='train')
    run.train_val()

if __name__ == "__main__":
    try:
        train_val()
    finally:
        wandb.finish() 
        # should hopefully be called even upon SIGTERM
        # but it can still take a long time to finish, so may need to increase SLURM timeouts
        # to prevent SLURM freaking out and sending node into drain