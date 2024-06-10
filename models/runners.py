import os
import logging
import shutil
from abc import abstractmethod
import warnings
from typing import Optional, Dict, Union, List, Callable, Tuple
import omegaconf
from omegaconf import DictConfig, OmegaConf
from warnings import warn
import json

import random
import time
import numpy as np
import hydra
import torch
from metrics.metrics import Metrics
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from data.cvrptw_dataset import CVRPTWDataset
from models.runner_utils import _adjust_time_limit, merge_sols, print_summary_stats, set_device, set_passMark, \
    eval_inference, get_time_limit
from models.runner_utils import NORMED_BENCHMARKS
from formats import RPSolution

logger = logging.getLogger(__name__)


# class BaseRunner:
#
#     def __init__(self, cfg: DictConfig):
#
#         # fix path aliases changed by hydra
#         self.cfg = update_path(cfg)
#         OmegaConf.set_struct(self.cfg, False)
#
#         # debug level
#         if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
#             self.debug = max(self.cfg.debug_lvl, 1)
#         else:
#             self.debug = 0
#         if self.debug > 1:
#             torch.autograd.set_detect_anomaly(True)

# BaseRunner
class BaseConstructionRunner:
    """basic runner functionality for wrapping setup, training and testing of RA baselines"""

    def __init__(self, cfg: DictConfig):

        # super(BaseConstructionRunner, self).__init__(cfg)

        # fix path aliases changed by hydra
        self.cfg = self._update_path(cfg)
        OmegaConf.set_struct(self.cfg, False)

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        self.acronym_ls = None
        self.acronym = None
        self.model = None
        self.ds = None

        # set device
        self.device = set_device(self.cfg)

        # init metric
        self.metric = None
        self.per_instance_time_limit_constr = None
        self.per_instance_time_limit_ls = None
        self.machine_info = None

        # set PassMark for eval
        if cfg.run_type in ["val", "test"]:
            # get Time Budget
            self.time_limit = get_time_limit(self.cfg)
            if self.cfg.test_cfg.add_ls:
                self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device,
                                                                self.cfg.test_cfg.ls_policy_cfg.search_workers)
                # get normalized per instance Time Limit
                if self.time_limit is not None:
                    self.per_instance_time_limit_constr = _adjust_time_limit(self.time_limit, self.passMark,
                                                                             self.device)
                    self.per_instance_time_limit_ls = _adjust_time_limit(self.time_limit, self.CPU_passMark,
                                                                         torch.device("cpu"))
                    logger.info(f"Eval PassMark for this run: {self.passMark}. "
                                f"Adjusted Time Limit per Instance for Construction: {self.per_instance_time_limit_constr}."
                                f" PassMark for additional GORT Search: {self.CPU_passMark}."
                                f" Adjusted Time Limit per Instance for Search : {self.per_instance_time_limit_ls}.")
                else:
                    self.per_instance_time_limit_constr = None
                    self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, True)
                    logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
            else:
                self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device)
                if self.time_limit is not None:
                    self.per_instance_time_limit_constr = _adjust_time_limit(self.time_limit, self.passMark,
                                                                             self.device)
                    logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                                f"Adjusted Time Limit per Instance: {self.per_instance_time_limit_constr}.")
                else:
                    self.per_instance_time_limit_constr = None
                    logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
                    self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)

        else:
            self.passMark, self.CPU_passMark = None, None

    def setup(self, compatible_problems: dict = None, data_transformation: Callable = None):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_problem(compatible_problems=compatible_problems,
                            data_transform=data_transformation)  # aka build dataset
        self._build_env()
        self._build_model()
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.data_file_path is not None and self.passMark is not None \
                    and self.cfg.test_cfg.eval_type != "simple":
                assert self.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")], \
                    f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                    f"or set test_cfg.eval_type to 'simple'"
                self.init_metrics(self.cfg)
            if self.cfg.test_cfg.add_ls:
                self._build_policy_ls()

    def _dir_setup(self):
        """directories for logging, checkpoints, ..."""
        self._cwd = os.getcwd()
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)
        # ckpt dir
        try:
            self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
            os.makedirs(self.cfg.checkpoint_save_path, exist_ok=True)
        except omegaconf.errors.ConfigAttributeError:
            pass


    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.passMark,
                              TimeLimit_=self.time_limit,
                              passMark_cpu=self.CPU_passMark,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              is_cpu_search=cfg.test_cfg.add_ls,
                              single_thread=self.cfg.test_cfg.ls_policy_cfg.search_workers,
                              verbose=self.debug >= 1)
        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit_constr if not cfg.test_cfg.add_ls \
            else self.per_instance_time_limit_ls

    def _build_problem(self, compatible_problems: Dict = None, data_transform: Callable = None):
        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()
        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg=cfg, DATA_CLASS=compatible_problems)
        elif cfg.run_type in ["train", "resume"]:
            self.ds, self.val_data = self.get_train_val_set(cfg,
                                                            data_transform,
                                                            compatible_problems
                                                            )
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def _build_policy_ls(self):
        """Load and prepare data and initialize GORT routing models."""
        from models.or_tools.or_tools import ParallelSolver
        policy_cfg = self.cfg.test_cfg.ls_policy_cfg.copy()
        self.policy_ls = ParallelSolver(
            problem=self.cfg.problem,
            solver_args=policy_cfg,
            time_limit=self.per_instance_time_limit_ls,
            num_workers=self.cfg.test_cfg.ls_policy_cfg.batch_size,  # instance processing in parallel
            search_workers=policy_cfg.search_workers
        )

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def run_inference(self) -> List[RPSolution]:
        # run test inference
        if self.cfg.test_cfg.add_ls:
            logger.info(
                f"Run-time dependent parameters: {self.device} Device "
                f"(threads: {self.cfg.test_cfg.ls_policy_cfg.batch_size}),"
                f" Adjusted Time Budget for construction: {self.per_instance_time_limit_constr} / instance."
                f" Adjusted Time Budget for LS: {self.per_instance_time_limit_ls} / instance.")
            construct_name = self.acronym.replace("_" + self.acronym_ls, "")
            logger.info(f"running test inference for {construct_name} with additional LS: {self.acronym_ls}...")
            # needs to return RPSolution (solution, cost, time, instance)
            summary_dct, solutions_construct = self._run_model()
            costs_constr = [sol_.cost for sol_ in solutions_construct]
            time_constr = [sol_.run_time for sol_ in solutions_construct]
            if None not in costs_constr:
                logger.info(
                    f"Constructed solutions with average cost {np.mean(costs_constr)} in {np.mean(time_constr)}/inst")
            else:
                logger.info(f"{self.acronym} constructed inf. sols. Defaulting to GORT default construction (SAVINGS).")
            # check if not surpassed construction time budget and still have time for search in Time Budget
            # self.per_instance_time_limit_ls
            time_for_ls = self.per_instance_time_limit_ls if self.per_instance_time_limit_ls is not None \
                else np.mean([d.time_limit for d in self.ds.data])
            print('np.mean(time_constr)', np.mean(time_constr))
            print('np.mean([d.time_limit for d in self.ds.data])', np.mean([d.time_limit for d in self.ds.data]))
            if np.mean(time_constr) < time_for_ls:
                logger.info(f"\n finished construction... starting LS")
                normed_dem = self.cfg.env_kwargs.generator_args.normalize_demands if self.cfg.problem.upper() != "TSP" \
                    else False
                sols_search = self.policy_ls.solve(self.ds.data,
                                                   normed_demands=normed_dem,
                                                   init_solution=solutions_construct,
                                                   distribution=self.cfg.coords_dist,
                                                   grid_size=self.ds.grid_size,
                                                   time_construct=float(np.mean(time_constr)),
                                                   info_from_construct=summary_dct)

                sols_ = merge_sols(sols_search, solutions_construct)
            else:
                sols_ = solutions_construct
                logger.info(f"Model {construct_name} used up runtime (on avg {np.mean(time_constr)}) for constructing "
                            f"(adj. time limit {self.per_instance_time_limit_constr}). "
                            f"Using constructed solution for Evaluation.")
                self.acronym = construct_name
        else:
            # run test inference
            logger.info(f"Run-time dependent parameters: {self.device} Device, "
                        f"Adjusted Time Budget for construction: {self.per_instance_time_limit_constr} / instance.")
            logger.info(f"running test inference for {self.acronym} as construction method...")
            _, sols_ = self._run_model()

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

    def run_test(self) -> Tuple[List, List]:
        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []

        if self.cfg.test_cfg.add_ls and 1 < self.cfg.test_cfg.ls_policy_cfg.batch_size < len(self.ds.data):
            logger.info(
                f"Parallelize local search runs: running {self.cfg.test_cfg.ls_policy_cfg.batch_size} instances "
                f"in parallel.")
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
        return results_all, stats_all

    def get_test_set(self, cfg, DATA_CLASS: dict = None):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be {DATA_CLASS.keys()}")
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
            dataset_range=cfg.test_cfg.dataset_range,
            normalize=cfg.normalize_data,
            seed=cfg.global_seed,
            TimeLimit=self.time_limit,
            machine_info=self.machine_info,
            load_base_sol=load_base_sol,
            load_bks=load_bks,
            verbose=self.debug >= 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        if 'save_testset' in list(cfg.test_cfg.keys()) and cfg.test_cfg.save_testset is not None:
            test_data = ds.data
            print('test_data[0]', test_data[0])
            torch.save(test_data, "test_data_temp.pt")
        return ds

    def get_train_val_set(self, cfg, transform_function: Callable = None, DATA_CLASS: dict = None):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")
        ds = dataset_class(
            is_train=True,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            # device=self.device,
            transform_func=transform_function,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )

        # if cfg.train_cfg.save_dataset_object:
        #     torch.save(ds, "dataset.pt")

        if cfg.train_cfg.get_val_set:
            ds_val = dataset_class(
                is_train=True,
                store_path=cfg.val_dataset if 'val_dataset' in list(cfg.keys()) else None,
                # default is None --> so generate ds_val
                num_samples=cfg.val_size,
                distribution=cfg.coords_dist,
                graph_size=cfg.graph_size,
                # device=self.device,
                transform_func=transform_function,
                seed=cfg.global_seed,
                verbose=self.debug >= 1,
                sampling_args=cfg.env_kwargs.sampling_args,
                generator_args=cfg.env_kwargs.generator_args
            )
            val_data = ds_val.sample(cfg.val_size)
            torch.save(val_data, "val_dataset_for_train_run.pt")
        else:
            val_data = None
        return ds, val_data

    @abstractmethod
    def _build_env(self):
        pass # raise NotImplementedError

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _run_model(self):
        raise NotImplementedError

    @abstractmethod
    def _update_path(self, cfg):
        raise NotImplementedError

    @staticmethod
    def seed_all(seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class BaseSearchRunner:
    """basic runner functionality for wrapping setup, training and testing of RA Search baselines"""

    def __init__(self, cfg: DictConfig):

        # super(BaseConstructionRunner, self).__init__(cfg)

        # fix path aliases changed by hydra
        self.cfg = self._update_path(cfg)
        OmegaConf.set_struct(self.cfg, False)

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        self.acronym = None
        self.model = None
        self.metric = None
        self.machine_info = None
        self.per_instance_time_limit = None
        self.env = None
        self.val_env = None

        # set device
        self.device = set_device(self.cfg)

        # set PassMark for eval
        self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device, number_threads=1)

        if cfg.run_type in ["val", "test"]:
            # get Time Budget
            self.time_limit = get_time_limit(self.cfg)
            if self.time_limit is not None:
                # get normalized per instance Time Limit
                self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device)
                logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                            f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
            else:
                logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
                self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)

    def setup(self, compatible_problems: dict = None, data_transformation: Callable = None):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_problem(compatible_problems=compatible_problems,
                            data_transform=data_transformation)  # aka build dataset
        self._build_env()
        try:
            self._build_model()
        except NotImplementedError:
            pass
        try:
            self._build_policy()
        except NotImplementedError:
            pass
        self._build_collectors()
        self._build_callbacks()
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.data_file_path is not None and self.passMark is not None \
                    and self.cfg.test_cfg.eval_type != "simple":
                assert self.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")], \
                    f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                    f"or set test_cfg.eval_type to 'simple'"
                self.init_metrics(self.cfg)

    def _dir_setup(self):
        """ Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)
        # checkpoint save dir
        if 'checkpoint_save_path' in list(self.cfg.keys()) and self.cfg.checkpoint_save_path is not None:
            self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
            os.makedirs(self.cfg.checkpoint_save_path, exist_ok=True)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.passMark,
                              TimeLimit_=self.time_limit,
                              passMark_cpu=self.CPU_passMark,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              single_thread=True,  # self.cfg.policy_cfg.num_workers
                              verbose=self.debug >= 1)

        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _build_problem(self, compatible_problems: Dict = None, data_transform: Callable = None):
        """Load dataset and create environment (problem state and data)."""

        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()
        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg=cfg, DATA_CLASS=compatible_problems)
        elif cfg.run_type in ["train", "resume"]:
            self.ds, self.val_data = self.get_train_val_set(cfg,
                                                            data_transform,
                                                            compatible_problems
                                                            )
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def run_test(self):

        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        parallel_workers = self.cfg.policy_cfg.num_workers if 'num_workers' in list(self.cfg.keys()) else 0
        if 1 < parallel_workers < len(self.ds.data):
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
        return results_all, stats_all

    def run_inference(self) -> List[RPSolution]:
        # single_thread - not sure if CPU implementation of DACT is using only single thread.
        logger.info(f"Run-time dependent parameters: {self.device} Device, "
                    f"Adjusted Time Budget: {self.per_instance_time_limit} / instance.")

        _, solutions_ = self._run_model()
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

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def get_test_set(self, cfg, DATA_CLASS: dict = None, data_transform: Callable = None):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}. "
                                      f"Must be {DATA_CLASS.keys()}")

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
            dataset_range=cfg.test_cfg.dataset_range,
            normalize=cfg.normalize_data,
            transform_func=data_transform,
            seed=cfg.global_seed,
            TimeLimit=self.time_limit,
            machine_info=self.machine_info,
            load_base_sol=load_base_sol,
            load_bks=load_bks,
            verbose=self.debug >= 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        return ds

    def get_train_val_set(self, cfg, transform_function: Callable = None, DATA_CLASS: dict = None):
        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")
        ds = dataset_class(
            is_train=True,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            # device=self.device,
            transform_func=transform_function,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )

        if cfg.train_cfg.get_val_set:
            ds_val = dataset_class(
                is_train=True,
                store_path=cfg.val_dataset if 'val_dataset' in list(cfg.keys()) else None,
                # default is None --> so generate ds_val
                num_samples=cfg.val_size,
                distribution=cfg.coords_dist,
                graph_size=cfg.graph_size,
                # device=self.device,
                transform_func=transform_function,
                seed=cfg.global_seed,
                verbose=self.debug >= 1,
                sampling_args=cfg.env_kwargs.sampling_args,
                generator_args=cfg.env_kwargs.generator_args
            )
            val_data = ds_val.sample(cfg.val_size)
            torch.save(val_data, "val_dataset_for_train_run.pt")
        else:
            val_data = None
        return ds, val_data

    # @staticmethod
    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.val_env is not None:
            self.val_env.seed(seed + 1)

    @abstractmethod
    def _build_env(self):
        pass  # raise NotImplementedError

    @abstractmethod
    def _build_policy(self):
        pass  # raise NotImplementedError

    def _build_collectors(self):
        pass  # raise NotImplementedError

    def _build_callbacks(self):
        pass  # raise NotImplementedError

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _run_model(self):
        raise NotImplementedError

    @abstractmethod
    def _update_path(self, cfg):
        raise NotImplementedError



def update_path(cfg: DictConfig):
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


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
