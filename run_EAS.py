#
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from models.EAS.runner import Runner
logger = logging.getLogger(__name__)


@hydra.main(config_path="models/EAS/config", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))
    r = Runner(cfg)
    r.run()


if __name__ == "__main__":
    run()
