#
import os
import time
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch

from data import CVRPDataset
from formats import TSPInstance, CVRPInstance, RPSolution

# , eval_rp
from models.Savings.savings import eval_savings, run_savings
from models.runner_utils import get_stats, _adjust_time_limit, merge_sols, print_summary_stats, eval_inference, \
    set_passMark, set_device, get_time_limit
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS

logger = logging.getLogger(__name__)

DATA_CLASS = {
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
        OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        # option for construction models to run with local search on top
        self.acronym, self.acronym_ls = self.get_acronym(model_name="Savings")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0

        # set device
        self.device = set_device(self.cfg)  # torch.device("cpu")

        # init metric
        self.metric = None
        self.per_instance_time_limit_constr = None
        self.per_instance_time_limit_ls = None
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
        if self.cfg.data_file_path is not None and self.passMark is not None \
                and self.cfg.test_cfg.eval_type != "simple":
            assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                f"or set test_cfg.eval_type to 'simple'"
            self.init_metrics(self.cfg)
        if self.cfg.test_cfg.add_ls:
            self._build_policy_ls()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)

    def _build_problem(self):
        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()
        self.ds = self.get_test_set(cfg)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.CPU_passMark,
                              TimeLimit_=self.time_limit,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              is_cpu_search=cfg.test_cfg.add_ls,
                              single_thread=True,
                              verbose=self.debug >= 1)
        self.ds.metric = self.metric
        # self.ds.adjusted_time_limit = self.time_limit
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _build_policy_ls(self):
        """Load and prepare data and initialize GORT routing models."""
        from models.or_tools.or_tools import ParallelSolver
        policy_cfg = self.cfg.test_cfg.ls_policy_cfg.copy()
        self.policy_ls = ParallelSolver(
            problem=self.cfg.problem,
            solver_args=policy_cfg,
            time_limit=self.per_instance_time_limit,
            num_workers=self.cfg.test_cfg.ls_policy_cfg.batch_size
        )

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        self.setup()
        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        logger.info(f"Run-time dependent parameters: {self.device} Device "
                    f"(threads: {self.cfg.test_cfg.ls_policy_cfg.batch_size}),"
                    f" Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        if self.cfg.test_cfg.add_ls and 1 < self.cfg.test_cfg.ls_policy_cfg.batch_size < len(self.ds.data):
            logger.info(f"Parallelize search runs: running {self.cfg.test_cfg.ls_policy_cfg.batch_size} instances "
                        f"in parallel.")
        for run in range(1, number_of_runs + 1):
            logger.info(f"running inference {run}/{number_of_runs}...")
            solutions_ = self.run_inference()
            logger.info(f"Starting Evaluation for run {run}/{number_of_runs} "
                        f"with time limit {self.time_limit} for {self.acronym}")
            results, summary_per_instance, stats = self.eval_inference(run, number_of_runs, solutions_)
            results_all.append(results)
            stats_all.append(stats)

            if self.cfg.save_as_base:
                logger.info(f'Saving solutions for {self.acronym} as Base Solution')
                self.save_BaseSol(results)

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
        if self.cfg.test_cfg.add_ls:
            construct_name = self.acronym.replace("_" + self.acronym_ls, "")
            logger.info(f"running test inference for {construct_name} with additional LS: {self.acronym_ls}...")
            results, solutions_construct = eval_savings(
                data=self.ds.data,
                savings_function=self.cfg.method['savings_criterion'],
                is_normalized=self.cfg.normalize_data,
            )
            costs_constr = [sol_.cost for sol_ in solutions_construct]
            time_constr = [sol_.run_time for sol_ in solutions_construct]

            # Roughly check quality of constructed sols
            if None not in costs_constr:
                logger.info(f"Constructed sols with avg cost {np.mean(costs_constr)} in {np.mean(time_constr)}/inst")
            else:
                logger.info(f"{self.acronym} constructed inf. sols. Defaulting to GORT default construction (SAVINGS).")
            # check if still have time for search in Time Budget
            time_for_ls = self.per_instance_time_limit if self.per_instance_time_limit is not None \
                else np.mean([d.time_limit for d in self.ds.data])
            print('np.mean(time_constr)', np.mean(time_constr))
            print('np.mean([d.time_limit for d in self.ds.data])', np.mean([d.time_limit for d in self.ds.data]))
            if np.mean(time_constr) < time_for_ls:
                logger.info(f"\n finished construction... starting LS")
                sols_search = self.policy_ls.solve(self.ds.data,
                                                   normed_demands=self.cfg.normalize_data,
                                                   init_solution=solutions_construct,
                                                   distribution=self.cfg.coords_dist,
                                                   time_construct=float(np.mean(time_constr)))
                sols_ = merge_sols(sols_search, solutions_construct)
            else:
                sols_ = solutions_construct
                logger.info(f"\n {construct_name} used up runtime (on avg {np.mean(time_constr)}) for constructing "
                            f"(time limit {self.time_limit}). Using constructed solution for Evaluation.")
                self.acronym = construct_name
        else:
            logger.info(f"running test inference for {self.acronym}...")
            results, sols_ = eval_savings(
                data=self.ds.data,
                savings_function=self.cfg.method['savings_criterion'],
                is_normalized=self.cfg.normalize_data,
            )

        return sols_

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

    def save_BaseSol(self, sols):
        """Since Savings+GORT_SA is used as Base Solver in the Benchmark, this method can be used after test
           to store the BaseSol.pkl file in correct format in the dataset folder ( for new datasets )"""

        if self.ds.store_path[-4:] == ".pkl":
            base_sol_path = os.path.join(os.path.dirname(self.ds.store_path), "BaseSol_"
                                         + os.path.basename(self.ds.store_path)[:5] + ".pkl")
        else:
            base_sol_path = os.path.join(self.ds.store_path, "BaseSol_"
                                         + os.path.basename(self.ds.store_path)[:5] + ".pkl")
        print('base_sol_path', base_sol_path)
        base_sol_x = {}
        for rp_sol in sols:
            base_sol_x[str(rp_sol.instance.instance_id)] = (rp_sol.running_costs, rp_sol.running_times, self.acronym)
        torch.save(base_sol_x, base_sol_path)

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['val', 'test']")

    def get_test_set(self, cfg):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
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
            graph_size=cfg.graph_size,
            dataset_size=cfg.test_cfg.dataset_size,
            normalize=cfg.normalize_data,
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

    def get_acronym(self, model_name):
        acronym, acronym_ls = model_name, None
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.test_cfg.add_ls:
                # self.cfg.test_cfg.ls_policy_cfg.local_search_strategy
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                if self.cfg.method.savings_criterion == 'gaskell_pi':
                    acronym = model_name + '_gaskell_pi_' + acronym_ls
                elif self.cfg.method.savings_criterion == 'clarke_wright':
                    acronym = model_name + '_clarke_wright_' + acronym_ls
                elif self.cfg.method.savings_criterion == 'gaskell_lambda':
                    acronym = model_name + '_gaskell_lambda_' + acronym_ls
                else:
                    acronym = model_name + acronym_ls
            else:
                if self.cfg.method.savings_criterion == 'gaskell_pi':
                    acronym = 'Savings_gaskell_pi'
                elif self.cfg.method.savings_criterion == 'clarke_wright':
                    acronym = 'Savings_clarke_wright'
            return acronym, acronym_ls

    @staticmethod
    def seed_all(seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def update_path(cfg: DictConfig):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()

    if cfg.test_cfg.data_file_path is not None:
        cfg.test_cfg.data_file_path = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.data_file_path)
        )

    if cfg.test_cfg.saved_res_dir is not None:
        cfg.test_cfg.saved_res_dir = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.saved_res_dir)
        )
    return cfg
