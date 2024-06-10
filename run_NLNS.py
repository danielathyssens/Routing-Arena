#
import logging
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from models.NLNS.runner import Runner

logger = logging.getLogger(__name__)


def lns_timelimit_validation(a, b, c):
    return a + (b / c)


OmegaConf.register_new_resolver("lns_timelimit_validation", lns_timelimit_validation)


@hydra.main(config_path="models/NLNS/config", config_name="config", version_base="1.1")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))
    r = Runner(cfg)
    r.run()


if __name__ == "__main__":
    run()
