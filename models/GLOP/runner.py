#
import os
import shutil
import logging
from abc import ABC
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
import warnings

import random
import numpy as np
import hydra
import torch
import time

from models.GLOP.GLOP.utils import load_model
from models.GLOP.GLOP.utils.functions import load_problem
from models.GLOP.GLOP.heatmap.cvrp.infer import load_partitioner
from models.GLOP.glop import eval_model
from data.tsp_dataset import TSPDataset
from data.cvrp_dataset import CVRPDataset
from models.runner_utils import get_stats, set_device, print_summary_stats, log_info, _adjust_time_limit, \
    eval_inference, set_passMark, get_time_limit
from formats import RPSolution
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS
from models.runners import BaseConstructionRunner

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


class Runner(BaseConstructionRunner, ABC):
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def __init__(self, cfg: DictConfig):

        super(Runner, self).__init__(cfg)

        # fix path aliases changed by hydra
        # self.cfg = update_path(cfg)
        # OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        self.acronym, self.acronym_ls = self.get_acronym(model_name="GLOP")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))
        # re-label model internal run_name (used to save ckpts)
        self.cfg.run_name = self.run_name

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup()
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'val', 'test', 'debug']")

    def _build_env(self, p_size_=None):
        pass

    def _build_model(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        cfg = self.cfg.copy()
        self.revisers = []
        revision_lens = cfg.policy_cfg.revision_lens
        # cfg.policy_cfg.

        for reviser_size in revision_lens:
            reviser_path = os.path.join(cfg.policy_cfg.reviser_path,"reviser_"+str(reviser_size)+"/epoch-299.pt")
            # f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
            reviser, _ = load_model(reviser_path, is_local=True)
            self.revisers.append(reviser)

        for reviser in self.revisers:
            reviser.to(self.device)
            reviser.eval()
            reviser.set_decode_type(cfg.policy_cfg.decode_strategy)

    def _run_model(self):
        return eval_model(
            data=self.ds.data,
            problem=self.cfg.problem,
            revisers=self.revisers,
            time_limit=self.per_instance_time_limit_constr,
            tester_cfg=self.cfg.tester_cfg,
            device=self.device,
            batch_size=self.cfg.tester_cfg.eval_batch_size,
        )

    def train(self, **kwargs):
        """Train the specified model."""
        logger.info(f"start training...")
        _ = train_model(
            problem=self.env,
            agent=self.policy,
            train_dataset=self.ds,
            validation_data=self.ds_val.data,
            opts=self.cfg.copy()
        )
        logger.info(f"training finished.")

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        self.setup()
        self.policy.load(self.cfg.test_cfg.checkpoint_load_path)
        epoch_resume = int(os.path.splitext(os.path.split(self.cfg.test_cfg.checkpoint_load_path)[-1])[0].split("-")[1])
        logger.info(f"Resuming after {epoch_resume}")
        self.policy.opts.epoch_start = epoch_resume + 1

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        self.train(**kwargs)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        # setup (data, env, ...)
        self.setup(compatible_problems=DATA_CLASS)
        # model-internal setting
        # self.cfg.eval_only = True
        # load policy
        # self.policy.load(self.cfg.test_cfg.checkpoint_load_path)

        results, summary = self.run_test()

    def get_acronym(self, model_name: str):
        acronym, acronym_ls = model_name, None
        acronym = model_name + "_it_" + str(self.cfg.policy_cfg.revision_iters[0])
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                # acronym_ls = 'GORT_' + str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                # if self.cfg.test_cfg.decode_type == 'greedy':
                acronym = acronym + '_' + acronym_ls
                # elif self.cfg.test_cfg.decode_type == 'sample':
        return acronym, acronym_ls

    def _update_path(self, cfg: DictConfig):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()

        if 'data_file_path' in list(cfg.keys()) and cfg.test_cfg.data_file_path is not None:
            cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.data_file_path)
            )
        # if cfg.val_env_cfg.data_file_path is not None:
        #     cfg.val_env_cfg.data_file_path = os.path.normpath(
        #         os.path.join(cwd, cfg.val_env_cfg.data_file_path)
        #     )
        # if cfg.tester_cfg.test_env_cfg.data_file_path is not None:
        #     cfg.tester_cfg.test_env_cfg.data_file_path = os.path.normpath(
        #         os.path.join(cwd, cfg.tester_cfg.test_env_cfg.data_file_path)
        #     )

        if 'single_large_instance' in list(cfg.env_kwargs.generator_args.keys()) and \
                cfg.env_kwargs.generator_args.single_large_instance is not None:
            cfg.env_kwargs.generator_args.single_large_instance = os.path.normpath(
                os.path.join(cwd, cfg.env_kwargs.generator_args.single_large_instance)
            )

        if cfg.test_cfg.saved_res_dir is not None:
            cfg.test_cfg.saved_res_dir = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.saved_res_dir)
            )

        if 'checkpoint_load_path' in list(cfg.keys()) and cfg.test_cfg.checkpoint_load_path is not None:
            cfg.test_cfg.checkpoint_load_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
            )

        if 'policy_cfg' in list(cfg.keys()):
            if 'reviser_path' in list(cfg.policy_cfg.keys()) and cfg.policy_cfg.reviser_path is not None:
                cfg.policy_cfg.reviser_path = os.path.normpath(
                    os.path.join(cwd, cfg.policy_cfg.reviser_path)
                )
        if 'policy_cfg' in list(cfg.keys()):
            if 'partitioner_path' in list(cfg.policy_cfg.keys()) and cfg.policy_cfg.partitioner_path is not None:
                cfg.policy_cfg.partitioner_path = os.path.normpath(
                    os.path.join(cwd, cfg.policy_cfg.partitioner_path)
                )

        if 'tester_cfg' in list(cfg.keys()):
            if 'ckpt_path_partitioner' in list(cfg.tester_cfg.keys()) and cfg.tester_cfg.ckpt_path_partitioner is not None:
                cfg.tester_cfg.ckpt_path_partitioner = os.path.normpath(
                    os.path.join(cwd, cfg.tester_cfg.ckpt_path_partitioner)
                )
        # if cfg.train_cfg.model_load.path is not None:

        return cfg


def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i + len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
