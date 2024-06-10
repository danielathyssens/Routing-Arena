#
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from models.BQ.runner import Runner
logger = logging.getLogger(__name__)


@hydra.main(config_path="models/BQ/config", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))
    rnr = Runner(cfg)
    rnr.run()


if __name__ == "__main__":
    run()

