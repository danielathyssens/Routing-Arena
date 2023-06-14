#
import os
import time
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig
import warnings

import random
import numpy as np
import hydra
import torch

from data import TSPDataset, CVRPDataset
from formats import TSPInstance, CVRPInstance, RPSolution
# , eval_rp
from models.FILO.filo import cvrp_inference
from models.runner_utils import get_stats, _adjust_time_limit, print_summary_stats, eval_inference, set_passMark, set_device, get_time_limit
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'CVRP': CVRPDataset
}


#
class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)

        # Model acronym
        self.acronym = 'FILO'
        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level
        if self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0

        # set device
        self.device = set_device(self.cfg)  # torch.device("cpu")

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
        self.seed_all(self.cfg.global_seed)
        self._build_problem()
        # self._build_policy()
        if self.cfg.data_file_path is not None and self.passMark is not None and self.cfg.test_cfg.eval_type != "simple":
            assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                f"or set test_cfg.eval_type to 'simple'"
            self.init_metrics(self.cfg)


    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.CPU_passMark,
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
        self.ds = self.get_test_set(cfg)

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):
        """Test (evaluate) the (trained) model on specified dataset."""

        self.setup()

        # run test inference
        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        logger.info(f"Run-time dependent parameters: {self.device} Device (threads: {self.cfg.policy_cfg.num_workers}),"
                    f" Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        if 1 < self.cfg.policy_cfg.num_workers < len(self.ds.data):
            logger.info(f"Parallelize search runs: running {self.cfg.policy_cfg.num_workers} instances "
                        f"in parallel at a time.")
        for run in range(1, number_of_runs + 1):
            logger.info(f"running inference {run}/{number_of_runs}...")
            solutions_ = self.run_inference()

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
        policy_cfg = self.cfg.policy_cfg.copy()
        results, solutions_ = cvrp_inference(
            data=self.ds.data,
            is_normalized=self.cfg.normalize_data,
            TIME_LIMIT=self.per_instance_time_limit,
            **policy_cfg
        )
        return solutions_

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

    def get_test_set(self, cfg) -> CVRPDataset:
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['CVRP']")

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


def update_path(cfg: DictConfig, fixed_dataset: bool = True):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()
    if fixed_dataset:
        if cfg.test_cfg.data_file_path is not None:
            cfg.test_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.data_file_path)
            )
    if cfg.policy_cfg.filo_exe_path is not None:
        cfg.policy_cfg.filo_exe_path = os.path.normpath(
            os.path.join(cwd, cfg.policy_cfg.filo_exe_path)
        )

    if cfg.test_cfg.saved_res_dir is not None:
        cfg.test_cfg.saved_res_dir = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.saved_res_dir)
        )
    return cfg
