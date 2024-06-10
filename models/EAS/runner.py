#
import os
import shutil
import logging
from warnings import warn
from typing import Optional, Dict, Union, List, Tuple
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch
import time

from models.EAS.EAS.source.cvrp.grouped_actors import ACTOR as CVRP_ACTOR
from models.EAS.EAS.source.tsp.grouped_actors import ACTOR as TSP_ACTOR

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from models.EAS.eas import eval_model, train_model
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from formats import RPSolution
from models.runner_utils import _adjust_time_limit, merge_sols, print_summary_stats, get_stats, set_device, \
    set_passMark, eval_inference, get_time_limit
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS
from models.runners import BaseSearchRunner

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


class Runner(BaseSearchRunner):
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
        self.acronym, self.acronym_ls = self.get_acronym(model_name="EAS", eas_cfg=self.cfg.method)

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level etc. --> all handled in BaseRunner

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup(compatible_problems=DATA_CLASS)
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'val', 'test', 'debug']")

    # def _build_env(self):
    #     print('self.cfg.env_cfg', self.cfg.env_cfg)
    #     Env = TSPEnv if self.cfg.problem.upper() == "TSP" else CVRPEnv
    #     self.env = Env(self.ds, **self.cfg.env_cfg)

    def _build_model(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        self.model = TSP_ACTOR().to(self.device) if self.cfg.problem.upper() == "TSP" \
            else CVRP_ACTOR().to(self.device)
        # self.model = Model(**self.cfg.model_cfg)

    def _run_model(self) -> Tuple[Dict, List[RPSolution]]:
        """Run the model and get solutions for the RP - executed in run_inference"""

        return eval_model(
            method=self.cfg.method,
            grouped_actor=self.model,
            data=self.ds.data,
            config=self.cfg.tester_cfg,
            problem=self.cfg.problem.upper(),
            device=self.device,
        )

    def train(self, **kwargs):
        """Train the specified model."""

        cfg = self.cfg.copy()

        optimizer = Optimizer(self.model.parameters(), **self.cfg.train_cfg.optimizer_cfg.optimizer)
        scheduler = Scheduler(optimizer, **self.cfg.train_cfg.optimizer_cfg.scheduler)

        logger.info(f"start training...")
        # results, solutions
        _ = train_model(
            Trainer=TSPTrainer if cfg.problem.upper() == "tsp" else CVRPTrainer,
            env=self.env,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            train_cfg=cfg.train_cfg.copy()
        )
        logger.info(f"training finished.")

        # self.save_results({
        #     "solutions": solutions,
        #     "summary": summary
        # })
        # logger.info(summary)

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        self.setup()
        epoch_resume = int(os.path.splitext(os.path.split(self.cfg.checkpoint_load_path)[-1])[0].split("-")[1])
        logger.info(f"Resume Training after {epoch_resume} epochs...")

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        self.train(**kwargs)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        self.setup(compatible_problems=DATA_CLASS)
        # adjust time limit in tester_cfg for EAS
        print('self.per_instance_time_limit', self.per_instance_time_limit)
        if len(set([inst.graph_size for inst in self.ds.data])) == 1:
            self.cfg.tester_cfg["max_runtime"] = int(self.per_instance_time_limit)
        else:  # max run_time will be updated in looping over instances
            self.cfg.tester_cfg["max_runtime"] = None
        checkpoint = torch.load(self.cfg.test_cfg.checkpoint_load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)  # ['model_state_dict']
        self.model.eval()

        results, summary = self.run_test()

    def get_acronym(self, model_name: str, eas_cfg: str):
        acronym, acronym_ls = model_name, None
        print('str(eas_cfg)', str(eas_cfg))
        if self.cfg.run_type in ["val", "test"]:
            acronym = model_name + '_' + eas_cfg
            # if pomo_size == '1':
            #     model_name_aug = model_name_aug + '_greedy'
            # if self.cfg.test_cfg.add_ls:
            #     ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
            #     acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
            #     acronym = model_name_aug + '_' + acronym_ls
            # else:
            #     acronym = model_name_aug
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

        if cfg.test_cfg.saved_res_dir is not None:
            cfg.test_cfg.saved_res_dir = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.saved_res_dir)
            )

        if 'checkpoint_load_path' in list(cfg.keys()) and cfg.test_cfg.checkpoint_load_path is not None:
            cfg.test_cfg.checkpoint_load_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
            )

        if 'policy_cfg' in list(cfg.keys()):
            if 'exe_path' in list(cfg.policy_cfg.keys()) and cfg.policy_cfg.exe_path is not None:
                cfg.policy_cfg.exe_path = os.path.normpath(
                    os.path.join(cwd, cfg.policy_cfg.exe_path)
                )

        if cfg.run_type == "train":
            if cfg.train_cfg.model_load.path is not None:
                cfg.train_cfg.model_load.path = os.path.normpath(
                    os.path.join(cwd, cfg.train_cfg.model_load.path)
                )
            if cfg.env_kwargs.generator_args.single_large_instance is not None:
                cfg.env_kwargs.generator_args.single_large_instance = os.path.normpath(
                    os.path.join(cwd, cfg.env_kwargs.generator_args.single_large_instance)
                )

        return cfg

def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i + len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
