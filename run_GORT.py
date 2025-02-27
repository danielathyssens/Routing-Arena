#
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from models.or_tools.runner import Runner

logger = logging.getLogger(__name__)


@hydra.main(config_path="models/or_tools/config", config_name="config", version_base="1.1")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))
    r = Runner(cfg)
    r.run()


if __name__ == "__main__":
    run()
