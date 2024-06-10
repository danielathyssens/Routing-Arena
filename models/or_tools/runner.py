#
import os
import time
import logging
from abc import ABC
from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch

# from lib.routing import RPDataset, eval_rp
from data import CVRPDataset, TSPDataset, CVRPTWDataset
from formats import RPSolution
from models.or_tools.or_tools import CVRPSolver, ParallelSolver
from models.runner_utils import get_stats, _adjust_time_limit, print_summary_stats, eval_inference, \
    set_passMark, set_device, get_time_limit
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
        # OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        self.acronym = 'GORT_' + str(self.cfg.policy)
        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # # debug level
        # if self.cfg.debug_lvl > 0:
        #     self.debug = max(self.cfg.debug_lvl, 1)
        # else:
        #     self.debug = 0
        #
        # set device
    #     self.device = set_device(self.cfg)  # torch.device("cpu")
    #
    #     # init metric
    #     self.metric = None
    #     self.per_instance_time_limit = None
    #     self.machine_info = None
    #
    #     # set PassMark for eval
    #     self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device, self.cfg.test_cfg.search_workers)
    #
    #     if cfg.run_type in ["val", "test"]:
    #         # get Time Budget
    #         self.time_limit = get_time_limit(self.cfg)
    #         if self.time_limit is not None:
    #             # get normalized per instance Time Limit
    #             self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device)
    #             logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
    #                         f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
    #         else:
    #             self.per_instance_time_limit = None
    #             self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)
    #             logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
    #
    #
    # def setup(self):
    #     """set up all entities."""
    #     self._dir_setup()
    #     self.seed_all(self.cfg.global_seed)
    #     self._build_problem()
    #     self._build_policy()
    #     if self.cfg.data_file_path is not None and self.passMark is not None and self.cfg.test_cfg.eval_type != "simple":
    #         assert self.device in [torch.device("cpu"), torch.device("cuda")], \
    #             f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
    #             f"or set test_cfg.eval_type to 'simple'"
    #         self.init_metrics(self.cfg)

    # def _dir_setup(self):
    #     """Set up directories for logging, checkpoints, etc."""
    #     self._cwd = os.getcwd()
    #     # tb logging dir
    #     self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
    #     # val log dir
    #     self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
    #     os.makedirs(self.cfg.log_path, exist_ok=True)

    # def init_metrics(self, cfg):
    #
    #     self.metric = Metrics(BKS=self.ds.bks,
    #                           passMark=self.CPU_passMark,
    #                           TimeLimit_=self.time_limit,
    #                           passMark_cpu=self.CPU_passMark,
    #                           base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
    #                           scale_costs=10000 if os.path.basename(
    #                               cfg.data_file_path) in NORMED_BENCHMARKS else None,
    #                           cpu=True,
    #                           single_thread=cfg.test_cfg.search_workers,
    #                           verbose=self.debug >= 1)
    #     self.ds.metric = self.metric
    #     self.ds.adjusted_time_limit = self.per_instance_time_limit
    #
    # def _build_problem(self):
    #     """Load dataset and create environment (problem state and data)."""
    #     cfg = self.cfg.copy()
    #
    #     assert cfg.run_type in ["val", "test"]
    #     self.ds = self.get_test_set(cfg)

    def _build_policy(self):
        """Load and prepare data and initialize GORT routing models."""
        policy_cfg = self.cfg.policy_cfg.copy()
        self.policy = ParallelSolver(
            problem=self.cfg.problem,
            solver_args=policy_cfg,
            time_limit=self.per_instance_time_limit,
            num_workers=self.cfg.test_cfg.batch_size,
            search_workers=self.cfg.test_cfg.search_workers,
            int_prec=policy_cfg.int_prec
        )

    # def save_results(self, result: Dict, run_id: int = 0):
    #     pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
    #     torch.save(result, pth)

    def _run_model(self) -> Tuple[Dict, List[RPSolution]]:
        sols = self.policy.solve(self.ds.data, normed_demands=self.cfg.normalize_data,
                                 distribution=self.cfg.coords_dist)
        return {}, sols

    def run(self):
        self.setup(compatible_problems=DATA_CLASS)

        results, summary = self.run_test()

        if self.cfg.save_as_base:
            logger.info(f'Saving solutions for {self.acronym} as Base Solution')
            logger.info(f'If multiple runs - saving first run results as Base Solution')
            self.save_BaseSol(results[0])

    def save_BaseSol(self, sols):
        """For CVRPTW GORT_Savings+GORT_SA is used as Base Solver in the Benchmark, this method can be used after test
           to store the BaseSol.pkl file in correct format in the dataset folder ( for new datasets )"""

        if self.ds.store_path[-3:] in ["pkl", ".pt"]:
            base_sol_path = os.path.join(os.path.dirname(self.ds.store_path), "BaseSol_"
                                         + os.path.basename(self.ds.store_path)[:-3] + "pkl")
        else:
            print('self.ds.store_path', self.ds.store_path)
            print('os.path.basename(self.ds.store_path)[:5]', os.path.basename(self.ds.store_path)[:5])
            base_sol_path = os.path.join(self.ds.store_path, "BaseSol_"
                                         + os.path.basename(self.ds.store_path)[:5] + ".pkl")
        print('base_sol_path', base_sol_path)
        base_sol_x = {}
        for rp_sol in sols:
            base_sol_x[str(rp_sol.instance.instance_id)] = (
            rp_sol.running_costs, rp_sol.running_times, self.acronym)
        torch.save(base_sol_x, base_sol_path)


        # default to a single run if number of runs not specified
        # number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        # results_all, stats_all = [], []
        # logger.info(f"Run-time dependent parameters: {self.device} Device (threads: {self.cfg.test_cfg.batch_size}),"
        #             f" Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        # if 1 < self.cfg.test_cfg.batch_size < len(self.ds.data):
        #     logger.info(f"Parallelize search runs: running {self.cfg.test_cfg.batch_size} instances in parallel.")
        # for run in range(1, number_of_runs + 1):
        #     logger.info(f"running inference {run}/{number_of_runs}...")
        #     solutions_ = self.run_inference()
        #     logger.info(f"Starting Evaluation for run {run}/{number_of_runs} "
        #                 f"with time limit {self.time_limit} for {self.acronym}")
        #     results, summary_per_instance, stats = self.eval_inference(run, number_of_runs, solutions_)
        #     results_all.append(results)
        #     stats_all.append(stats)
        # if number_of_runs > 1:
        #     print_summary_stats(stats_all, number_of_runs)
        #     # save overall list of results (if just one run - single run is saved in eval_inference)
        #     if self.cfg.test_cfg.save_solutions:
        #         logger.info(f"Storing Overall Results for {number_of_runs} runs in {os.path.join(self.cfg.log_path)}")
        #         self.save_results(
        #             result={
        #                 "solutions": results_all,
        #                 "summary": stats_all,
        #             })

    # def eval_inference(self, curr_run: int, number_of_runs: int, RP_solutions: List[RPSolution]):
    #     return eval_inference(
    #         curr_run,
    #         number_of_runs,
    #         RP_solutions,
    #         self.ds,
    #         self.cfg.log_path,
    #         self.acronym,
    #         self.cfg.test_cfg,
    #         self.debug
    #     )
    #
    # def get_test_set(self, cfg):
    #     if cfg.problem.upper() in DATA_CLASS.keys():
    #         dataset_class = DATA_CLASS[cfg.problem.upper()]
    #     else:
    #         raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
    #                                   f"Must be ['TSP', 'CVRP']")
    #
    #     if cfg.test_cfg.eval_type != "simple":
    #         load_bks = True
    #         if cfg.test_cfg.eval_type == "wrap" or "wrap" in cfg.test_cfg.eval_type:
    #             load_base_sol = True
    #         else:
    #             load_base_sol = False
    #     else:
    #         load_bks, load_base_sol = False, False
    #
    #     ds = dataset_class(
    #         store_path=cfg.test_cfg.data_file_path if cfg.test_cfg.data_file_path else None,
    #         distribution=cfg.coords_dist,
    #         graph_size=cfg.graph_size,
    #         dataset_size=cfg.test_cfg.dataset_size,
    #         normalize=cfg.normalize_data,
    #         seed=cfg.global_seed,
    #         verbose=self.debug >= 1,
    #         TimeLimit=self.time_limit,
    #         machine_info=self.machine_info,
    #         load_base_sol=load_base_sol,
    #         load_bks=load_bks,
    #         sampling_args=cfg.env_kwargs.sampling_args,
    #         generator_args=cfg.env_kwargs.generator_args
    #     )
    #     return ds

    @staticmethod
    def seed_all(seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _update_path(self, cfg: DictConfig, fixed_dataset: bool = True):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()
        if fixed_dataset:
            if 'data_file_path' in list(cfg.test_cfg.keys()):
                cfg.test_cfg.data_file_path = os.path.normpath(
                    os.path.join(cwd, cfg.test_cfg.data_file_path)
                )

            if cfg.test_cfg.saved_res_dir is not None:
                cfg.test_cfg.saved_res_dir = os.path.normpath(
                    os.path.join(cwd, cfg.test_cfg.saved_res_dir)
                )
        return cfg
