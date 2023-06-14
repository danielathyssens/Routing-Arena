#
import os
import time
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig
from pathlib import Path

import random
import numpy as np
import hydra
import torch

from data import TSPDataset, CVRPDataset
from formats import TSPInstance, CVRPInstance, RPSolution
from models.NeuroLKH.neuro_lkh import cvrp_inference
from models.runner_utils import _adjust_time_limit, print_summary_stats, get_stats, set_device, set_passMark, \
    eval_inference, get_time_limit
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)

        # Model acronym
        self.acronym = 'NeuroLKH' if self.cfg.policy == 'NeuroLKH' else 'LKH'
        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # set device
        cfg_ = self.cfg.copy()
        cfg_['cuda'] = True if self.cfg.policy == 'NeuroLKH' else False
        self.device = set_device(cfg_)

        # init metric
        self.metric = None
        self.per_instance_time_limit = None
        self.machine_info = None

        # set PassMark for eval
        self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device)

        if cfg.run_type in ["val", "test"]:
            # get Time Budget
            self.time_limit = get_time_limit(self.cfg)
            if self.time_limit is not None:
            # get normalized per instance Time Limit
                self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device)
                logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                            f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
            else:
                self.per_instance_time_limit = None
                self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)
                logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self._build_problem()
        self.seed_all(self.cfg.global_seed)
        if self.cfg.data_file_path is not None and self.passMark is not None and self.cfg.test_cfg.eval_type != "simple":
            assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                f"or set test_cfg.eval_type to 'simple'"
            self.init_metrics(self.cfg)

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        # self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        # self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.passMark,
                              TimeLimit_=self.time_limit,
                              passMark_cpu=self.CPU_passMark,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              single_thread=True,
                              verbose=self.debug >= 1)

        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _build_problem(self):
        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()

        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg)
        elif cfg.run_type in ["train", "resume"]:
            self.ds_val = self.get_train_val_set(cfg)
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def train(self, **kwargs):
        """Train the specified model."""
        raise NotImplementedError
        logger.info(f"start training...")
        results, solutions = train_dataset(...)
        logger.info(f"training finished.")
        logger.info(results)
        solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        self.callbacks['monitor'].save_results({
            "solutions": solutions,
            "summary": summary
        }, 'val_results')
        logger.info(summary)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        if not self.cfg.problem.upper() == "CVRP":
            raise NotImplementedError

        self.setup()
        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        logger.info(f"Run-time dependent parameters: {self.device} Device (threads: {self.cfg.policy_cfg.num_workers}),"
                    f" Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        if 1 < self.cfg.policy_cfg.num_workers < len(self.ds.data):
            logger.info(f"Parallelize search runs: running {self.cfg.policy_cfg.num_workers} instances "
                        f"in parallel.")
        for run in range(1, number_of_runs + 1):
            logger.info(f"running inference {run}/{number_of_runs}...")
            summary_res, solutions_ = self.run_inference()
            logger.info(f"Starting Evaluation for run {run}/{number_of_runs} "
                        f"with time limit {self.time_limit} for {self.acronym}")
            results, summary_per_instance, stats = self.eval_inference(run, number_of_runs, solutions_)
            results_all.append(results)
            stats_all.append(stats)
        if number_of_runs > 1:
            print_summary_stats(stats_all, number_of_runs)
            # save overall list of results (if just one run - single run is saved in eval_inference)
            if self.cfg.test_cfg.save_solutions:
                logger.info(f"Storing Overall Results for {number_of_runs} runs in {os.path.join(self.cfg.log_path)}")
                self.save_results(
                    result={
                        "solutions": results_all,
                        "summary": stats_all,
                    })

    def run_inference(self):
        logger.info(f"Run-time dependent parameters: {self.device} Device (threads: {self.cfg.policy_cfg.num_workers}),"
                    f" total adjusted Time Budget: {self.per_instance_time_limit}")
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
                summary_res, sols = cvrp_inference(
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
            return cvrp_inference(
                data=self.ds.data,
                is_normalized=self.cfg.normalize_data,
                device=self.device,
                method=self.cfg.policy,
                TIME_LIMIT=self.per_instance_time_limit,
                **self.cfg.policy_cfg
            )

    def eval_inference(self, curr_run: int, number_of_runs: int, RP_solutions: List[RPSolution]):
        return eval_inference(
            curr_run,
            number_of_runs,
            RP_solutions,
            self.ds,
            self.cfg.log_path,
            self.acronym,
            self.cfg.test_cfg,
            self.debug
        )

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        raise NotImplementedError
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        state_dict = torch.load(ckpt_pth, map_location=self.device)
        self.load_state_dict(state_dict)

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        logger.info(f"resuming training from: {ckpt_pth}")
        self.train(resume_from_log=True, **kwargs)

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

    def get_test_set(self, cfg):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")

        if cfg.test_cfg.eval_type != "simple":
            load_bks = True
            if cfg.test_cfg.eval_type == "wrap" or "wrap" in cfg.test_cfg.eval_type:
                load_base_sol = True
            else:
                load_base_sol = False
        else:
            load_bks, load_base_sol = False, False

        ds = dataset_class(
            store_path=cfg.test_cfg.data_file_path if 'data_file_path' in list(cfg.test_cfg.keys()) else None,
            distribution=cfg.coords_dist,
            dataset_size=cfg.test_cfg.dataset_size,
            normalize=cfg.normalize_data,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            TimeLimit=self.time_limit,
            machine_info=self.machine_info,
            load_base_sol=load_base_sol,
            load_bks=load_bks,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        return ds

    def get_train_val_set(self, cfg):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")
        ds_val = dataset_class(
            is_train=True,
            store_path=cfg.val_dataset,  # default is None --> so generate ds_val
            num_samples=cfg.val_size,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            device=self.device,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        return ds_val

    @staticmethod
    def seed_all(seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def update_path(cfg: DictConfig, fixed_dataset: bool = True):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()
    if fixed_dataset:
        if cfg.test_cfg.data_file_path is not None:
            cfg.test_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.data_file_path)
            )
    if cfg.policy_cfg.model_path is not None:
        cfg.policy_cfg.model_path = os.path.normpath(
            os.path.join(cwd, cfg.policy_cfg.model_path)
        )
    if cfg.policy_cfg.lkh_exe_path is not None:
        cfg.policy_cfg.lkh_exe_path = os.path.normpath(
            os.path.join(cwd, cfg.policy_cfg.lkh_exe_path)
        )

    if cfg.test_cfg.saved_res_dir is not None:
        cfg.test_cfg.saved_res_dir = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.saved_res_dir)
        )

    return cfg
