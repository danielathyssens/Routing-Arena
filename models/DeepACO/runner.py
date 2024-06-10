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

from models.DeepACO.deep_aco import eval_model
from models.DeepACO.DeepACO.tsp.net import Net as Net_tsp
from models.DeepACO.DeepACO.cvrp.net import Net as Net_cvrp
from models.DeepACO.DeepACO.tsp_nls.net import Net as Net_tsp_nls
from models.DeepACO.DeepACO.cvrp_nls.net import Net as Net_cvrp_nls
from models.DeepACO.DeepACO.tsp.aco import ACO as ACO_tsp
from models.DeepACO.DeepACO.cvrp_nls.aco import ACO as ACO_cvrp_nls
from models.DeepACO.DeepACO.tsp_nls.aco import ACO as ACO_tsp_nls
from models.DeepACO.DeepACO.cvrp.aco import ACO as ACO_cvrp

from data.tsp_dataset import TSPDataset
from data.cvrp_dataset import CVRPDataset
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
        self.acronym, self.acronym_ls = self.get_acronym(model_name="DeepACO")

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

    def _build_env(self):
        # set env - to be initialized in deep_aco.py
        if self.cfg.problem.lower() == "cvrp":
            self.env = ACO_cvrp if self.cfg.model in ["DeepACO", "ACO"] else ACO_cvrp_nls
        elif self.cfg.problem.lower() == "tsp":
            if self.cfg.model in ["DeepACO", "ACO"]:
                self.env = ACO_tsp  # if self.cfg.model == "DeepACO"
            elif self.cfg.model in ["DeepACO_NLS", "ACO_NLS"]:
                self.env = ACO_tsp_nls
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _build_model(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        # policy_cfg = self.cfg.policy_cfg.copy()
        if self.cfg.problem.lower() == "tsp":
            if self.cfg.model in ["DeepACO", "DeepACO_NLS"]:
                self.model = Net_tsp().to(self.device) if self.cfg.model in ["DeepACO", "ACO"]\
                    else Net_tsp_nls().to(self.device)
            else:
                self.model = None
        elif self.cfg.problem.lower() == "cvrp":
            if self.cfg.model in ["DeepACO", "DeepACO_NLS"]:
                self.model = Net_cvrp().to(self.device) if self.cfg.model == "DeepACO" \
                    else Net_cvrp_nls().to(self.device)
            else:
                self.model = None
        else:
            raise NotImplementedError
        logger.info(f"Builded Policy for {self.acronym}")

    def _run_model(self):
        """Run the model"""
        return eval_model(
            aco_env=self.env,
            problem=self.cfg.problem,
            model=self.model,
            data=self.ds.data,
            device=self.device,
            tester_cfg=self.cfg.tester_cfg,
            adjusted_time_budget=self.per_instance_time_limit_constr,
        )

    def train(self, **kwargs):
        """Train the specified model."""
        raise NotImplementedError
        # logger.info(f"start training...")
        # _ = train_model(
        #     problem=self.env,
        #     agent=self.policy,
        #     train_dataset=self.ds,
        #     validation_data=self.ds_val.data,
        #     opts=self.cfg.copy()
        # )
        # logger.info(f"training finished.")

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        raise NotImplementedError
        # self.setup()
        # self.policy.load(self.cfg.test_cfg.checkpoint_load_path)
        # epoch_resume = int(os.path.splitext(os.path.split(self.cfg.test_cfg.checkpoint_load_path)[-1])[0].split("-")[1])
        # logger.info(f"Resuming after {epoch_resume}")
        # self.policy.opts.epoch_start = epoch_resume + 1
        #
        # # remove the unnecessary new directory hydra creates
        # new_hydra_dir = os.getcwd()
        # if "resume" in new_hydra_dir:
        #     remove_dir_tree("resume", pth=new_hydra_dir)
        #
        # self.train(**kwargs)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        # setup (data, env, ...)
        self.setup(compatible_problems=DATA_CLASS)
        # load policy
        if self.cfg.model not in ["ACO", "ACO_NLS"]:
            logger.info(f'Loading model for testing from: {self.cfg.test_cfg.checkpoint_load_path}')
            try:
                self.model.load_state_dict(torch.load(self.cfg.test_cfg.checkpoint_load_path, map_location=self.device))
            except FileNotFoundError:
                if self.cfg.graph_size is None:
                    # there is no fixed graph size --> so there's "None" added to the path --> delete
                    # default to ckpt for size 100
                    self.cfg.test_cfg["checkpoint_load_path"] = self.cfg.test_cfg.checkpoint_load_path[:-7] + "100.pt"
                    logger.info(f'Updating path to: {self.cfg.test_cfg.checkpoint_load_path} - no fixed graph size...')
                    self.model.load_state_dict(torch.load(self.cfg.test_cfg.checkpoint_load_path,
                                                          map_location=self.device))
                else:
                    logger.info(f'DeepACO checkpoint for this graph-size does not exist... '
                                f'Default to ckpt for graph-size 100...')
                    self.cfg.test_cfg["checkpoint_load_path"] = os.path.join(os.path.dirname(
                        self.cfg.test_cfg.checkpoint_load_path), self.cfg.problem.lower() + "100.pt")
                    logger.info(f'Updating path to: {self.cfg.test_cfg["checkpoint_load_path"]}')
                    self.model.load_state_dict(torch.load(self.cfg.test_cfg.checkpoint_load_path,
                                                          map_location=self.device))

        results, summary = self.run_test()

    def get_acronym(self, model_name: str):
        acronym = model_name
        acronym_ls = None
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.model == "DeepACO":
                acronym = model_name
            elif self.cfg.model == "ACO":
                acronym = "ACO"
            elif self.cfg.model == "DeepACO_NLS":
                acronym = "DeepACO_NLS"

            if self.cfg.tester_cfg.t_aco is not None and self.cfg.tester_cfg.ignore_time_limit:
                acronym = acronym + "_t_aco_" + str(self.cfg.tester_cfg.t_aco[0])
            else:
                acronym = acronym
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                acronym = acronym + '_' + acronym_ls

        return acronym, acronym_ls

    def _update_path(self, cfg):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()

        if 'data_file_path' in list(cfg.keys()) and cfg.test_cfg.data_file_path is not None:
            cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.data_file_path)
            )

        if 'single_large_instance' in list(cfg.env_kwargs.generator_args.keys()) and \
                cfg.env_kwargs.generator_args.single_large_instance is not None:
            cfg.env_kwargs.generator_args.single_large_instance = os.path.normpath(
                os.path.join(cwd, cfg.env_kwargs.generator_args.single_large_instance)
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
