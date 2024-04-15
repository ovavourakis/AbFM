import hydra

from model_run import *
import experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def test(cfg: DictConfig) -> None:
    cfg = eu.load_warmstart_config(cfg)
    run = ModelRun(cfg=cfg, stage='test')
    run.test()
    
if __name__ == "__main__":
    test()