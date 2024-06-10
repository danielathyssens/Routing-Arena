#
import os
import time
import logging
from abc import ABC
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig
from pathlib import Path

import random
import numpy as np
import hydra
import torch

from data import TSPDataset, CVRPDataset, CVRPTWDataset
from formats import TSPInstance, CVRPInstance, RPSolution
# from models.NeuroLKH.neuro_lkh import cvrp_inference, tsp_inference, vrptw_inference
from models.NeuroLKH.neuro_lkh import inference
from models.runner_utils import _adjust_time_limit, print_summary_stats, get_stats, set_device, set_passMark, \
    eval_inference, get_time_limit
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS
from models.runners import BaseSearchRunner


logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset,
    'CVRPTW': CVRPTWDataset
}


class Runner(BaseSearchRunner, ABC):
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def __init__(self, cfg: DictConfig):
        super(Runner, self).__init__(cfg)

        # fix path aliases changed by hydra
        # self.cfg = update_path(cfg)

        # Model acronym
        self.acronym = 'NeuroLKH' if self.cfg.policy == 'NeuroLKH' else 'LKH'
        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level
        # if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
        #     self.debug = max(self.cfg.debug_lvl, 1)
        # else:
        #     self.debug = 0
        # if self.debug > 1:
        #     torch.autograd.set_detect_anomaly(True)

        # set device
        # cfg_ = self.cfg.copy()
        # cfg_['cuda'] = True if self.cfg.policy == 'NeuroLKH' else False --> handled in hydra config
        # self.device = set_device(cfg_)

        # init metric
        # self.metric = None
        # self.per_instance_time_limit = None
        # self.machine_info = None
        #
        # # set PassMark for eval
        # self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device)

        # if cfg.run_type in ["val", "test"]:
        #     # get Time Budget
        #     self.time_limit = get_time_limit(self.cfg)
        #     if self.time_limit is not None:
        #     # get normalized per instance Time Limit
        #         self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device)
        #         logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
        #                     f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
        #     else:
        #         self.per_instance_time_limit = None
        #         self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)
        #         logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")

    # def setup(self):
    #     """set up all entities."""
    #     self._dir_setup()
    #     self.seed_all(self.cfg.global_seed)
    #     self._build_problem()
    #     if self.cfg.data_file_path is not None and self.passMark is not None and self.cfg.test_cfg.eval_type != "simple":
    #         assert self.device in [torch.device("cpu"), torch.device("cuda")], \
    #             f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
    #             f"or set test_cfg.eval_type to 'simple'"
    #         self.init_metrics(self.cfg)

    # def _dir_setup(self):
    #     """Set up directories for logging, checkpoints, etc."""
        # self._cwd = os.getcwd()
        # # tb logging dir
        # # self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # # checkpoint save dir
        # # self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        # # val log dir
        # self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        # os.makedirs(self.cfg.log_path, exist_ok=True)

    # def init_metrics(self, cfg):
    #     self.metric = Metrics(BKS=self.ds.bks,
    #                           passMark=self.passMark,
    #                           TimeLimit_=self.time_limit,
    #                           passMark_cpu=self.CPU_passMark,
    #                           base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
    #                           scale_costs=10000 if os.path.basename(
    #                               cfg.data_file_path) in NORMED_BENCHMARKS else None,
    #                           cpu=False if self.device != torch.device("cpu") else True,
    #                           single_thread=True,
    #                           verbose=self.debug >= 1)
    #
    #     self.ds.metric = self.metric
    #     self.ds.adjusted_time_limit = self.per_instance_time_limit

    # def _build_problem(self):
    #     """Load dataset and create environment (problem state and data)."""
    #     cfg = self.cfg.copy()
    #
    #     if cfg.run_type in ["val", "test"]:
    #         self.ds = self.get_test_set(cfg)
    #     elif cfg.run_type in ["train", "resume"]:
    #         self.ds_val = self.get_train_val_set(cfg)
    #     else:
    #         raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
    #                                   f"Must be ['val', 'test', 'train', 'resume']")

    # def save_results(self, result: Dict, run_id: int = 0):
    #     pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
    #     torch.save(result, pth)

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

    def train(self, **kwargs):
        """Train the specified model."""
        raise NotImplementedError
        # logger.info(f"start training...")
        # results, solutions = train_dataset(...)
        # logger.info(f"training finished.")
        # logger.info(results)
        # solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        # self.callbacks['monitor'].save_results({
        #     "solutions": solutions,
        #     "summary": summary
        # }, 'val_results')
        # logger.info(summary)

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        raise NotImplementedError
        # ckpt_pth = self.cfg.get('checkpoint_load_path')
        # assert ckpt_pth is not None
        # state_dict = torch.load(ckpt_pth, map_location=self.device)
        # self.load_state_dict(state_dict)
        #
        # # remove the unnecessary new directory hydra creates
        # new_hydra_dir = os.getcwd()
        # if "resume" in new_hydra_dir:
        #     remove_dir_tree("resume", pth=new_hydra_dir)
        #
        # logger.info(f"resuming training from: {ckpt_pth}")
        # self.train(resume_from_log=True, **kwargs)

    def _run_model(self):
        # inference = self.cfg.problem.lower() + "_inference"
        # for NeuroLKH if instance set has different sizes, need to evaluate each instance in set separately
        is_X = self.cfg.test_cfg.data_file_path[-1] == "X"
        is_Golden = self.cfg.coords_dist == "golden"
        if (self.cfg.policy == "NeuroLKH" and is_X) or (self.cfg.policy == "NeuroLKH" and is_Golden):
            eval_single = True
        else:
            eval_single = False
        if eval_single:
            summary_res_all, rp_sols_all = [], []
            for data_instance in self.ds.data:
                summary_res, sols = inference(
                    problem=self.cfg.problem,
                    data=[data_instance],
                    is_normalized=self.cfg.normalize_data,
                    device=self.device,
                    method=self.cfg.policy,
                    TIME_LIMIT=self.per_instance_time_limit,
                    **self.cfg.policy_cfg
                )
                summary_res_all.extend(summary_res)
                rp_sols_all.extend(sols)
            return summary_res_all, rp_sols_all
        else:
            return inference(
                problem=self.cfg.problem,
                data=self.ds.data,
                is_normalized=self.cfg.normalize_data,
                device=self.device,
                method=self.cfg.policy,
                TIME_LIMIT=self.per_instance_time_limit,
                **self.cfg.policy_cfg
            )

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        self.setup(compatible_problems=DATA_CLASS)

        results, summary = self.run_test()

    def _update_path(self, cfg: DictConfig):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()
        if cfg.test_cfg.fixed_dataset:
            if cfg.test_cfg.data_file_path is not None:
                cfg.test_cfg.data_file_path = os.path.normpath(
                    os.path.join(cwd, cfg.test_cfg.data_file_path)
                )
        if cfg.policy_cfg.model_path is not None:
            cfg.policy_cfg.model_path = os.path.normpath(
                os.path.join(cwd, cfg.policy_cfg.model_path)
            )
        if cfg.policy_cfg.exe_path is not None:
            cfg.policy_cfg.exe_path = os.path.normpath(
                os.path.join(cwd, cfg.policy_cfg.exe_path)
            )

        if cfg.test_cfg.saved_res_dir is not None:
            cfg.test_cfg.saved_res_dir = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.saved_res_dir)
            )

        return cfg
