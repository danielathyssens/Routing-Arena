#
import os
import shutil
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch
import time

from models.POMO.POMO.TSP.POMO.TSPEnv import TSPEnv
from models.POMO.POMO.TSP.POMO.TSPModel import TSPModel
from models.POMO.POMO.TSP.POMO.TSPTrainer import TSPTrainer
from models.POMO.POMO.TSP.POMO.TSPTester import TSPTester

from models.POMO.POMO.cvrp.POMO.CVRPEnv import CVRPEnv
from models.POMO.POMO.cvrp.POMO.CVRPModel import CVRPModel
from models.POMO.POMO.cvrp.POMO.CVRPTrainer import CVRPTrainer
from models.POMO.POMO.cvrp.POMO.CVRPTester import CVRPTester

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from models.POMO.pomo import eval_model, train_model
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from formats import RPSolution
from models.runner_utils import _adjust_time_limit, merge_sols, print_summary_stats, get_stats, set_device, set_passMark, eval_inference, get_time_limit
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
        OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        self.acronym, self.acronym_ls = self.get_acronym(model_name="POMO", pomo_cfg=self.cfg.tester_cfg)

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
                    logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                                f"Adjusted Time Limit per Instance for "
                                f"Construction: {self.per_instance_time_limit_constr}."
                                f" PassMark for additional GORT Search: {self.CPU_passMark}."
                                f" Adjusted Time Limit per Instance for Search : {self.per_instance_time_limit_ls}.")
                else:
                    self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, True)
                    logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
            else:
                self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device)
                if self.time_limit is not None:
                    # get normalized per instance Time Limit
                    self.per_instance_time_limit_constr = _adjust_time_limit(self.time_limit, self.passMark,
                                                                             self.device)
                    logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                                f"Adjusted Time Limit per Instance: {self.per_instance_time_limit_constr}.")
                else:
                    logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")
                    self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)

        else:
            self.passMark, self.CPU_passMark = None, None

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_problem()
        self._build_env()
        self._build_model()
        if self.cfg.run_type in ["val", "test"]:
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
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        os.makedirs(self.cfg.checkpoint_save_path, exist_ok=True)
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
                              is_cpu_search=cfg.test_cfg.add_ls,
                              single_thread=self.cfg.test_cfg.ls_policy_cfg.search_workers,
                              verbose=self.debug >= 1)
        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit_constr if not cfg.test_cfg.add_ls \
            else self.per_instance_time_limit_ls

    def _build_problem(self):
        """Load dataset and create environment (problem state)."""
        cfg = self.cfg.copy()

        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg)
        elif cfg.run_type in ["train", "resume"]:
            self.ds, self.ds_val = self.get_train_val_set(cfg)
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def _build_env(self):
        Env = TSPEnv if self.cfg.problem.upper() == "TSP" else CVRPEnv
        self.env = Env(self.ds, **self.cfg.env_cfg)

    def _build_model(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        Model = TSPModel if self.cfg.problem.upper() == "TSP" else CVRPModel
        self.model = Model(**self.cfg.model_cfg)

    def _build_policy_ls(self):
        """Load and prepare data and initialize GORT routing models."""
        from models.or_tools.or_tools import ParallelSolver
        policy_cfg = self.cfg.test_cfg.ls_policy_cfg.copy()
        self.policy_ls = ParallelSolver(
            problem=self.cfg.problem,
            solver_args=policy_cfg,
            time_limit=self.per_instance_time_limit_ls,
            num_workers=policy_cfg.batch_size,
            search_workers=policy_cfg.search_workers
        )

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

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

    def train(self, **kwargs):
        """Train the specified model."""

        cfg = self.cfg.copy()

        optimizer = Optimizer(self.model.parameters(), **self.cfg.train_cfg.optimizer_cfg.optimizer)
        scheduler = Scheduler(optimizer, **self.cfg.train_cfg.optimizer_cfg.scheduler)

        logger.info(f"start training...")
        # results, solutions
        _ = train_model(
            Trainer=TSPTrainer if cfg.problem.upper() == "TSP" else CVRPTrainer,
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

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        self.setup()

        checkpoint = torch.load(self.cfg.checkpoint_load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        if self.cfg.test_cfg.add_ls and 1 < self.cfg.test_cfg.ls_policy_cfg.batch_size < len(self.ds.data):
            logger.info(f"Parallelize local search runs: running {self.cfg.test_cfg.ls_policy_cfg.batch_size} instances "
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

    def run_inference(self):
        if self.cfg.test_cfg.add_ls:
            logger.info(
                f"Run-time dependent parameters: {self.device} Device (threads: {self.cfg.test_cfg.ls_policy_cfg.batch_size}),"
                f" Adjusted Time Budget for construction: {self.per_instance_time_limit_constr} / instance."
                f" Adjusted Time Budget for LS: {self.per_instance_time_limit_ls} / instance.")
            construct_name = self.acronym.replace("_" + self.acronym_ls, "")
            logger.info(f"running test inference for {construct_name} with additional LS: {self.acronym_ls}...")
            _, solutions_construct = eval_model(
                Tester=TSPTester if self.cfg.problem.upper() == "TSP" else CVRPTester,
                env=self.env,
                model=self.model,
                data=self.ds.data,
                device=self.device,
                tester_cfg=self.cfg.tester_cfg,
            )
            costs_constr = [sol_.cost for sol_ in solutions_construct]
            time_constr = [sol_.run_time for sol_ in solutions_construct]
            if None not in costs_constr:
                logger.info(
                    f"Constructed solutions with average cost {np.mean(costs_constr)} in {np.mean(time_constr)}")
            else:
                logger.info(f"{construct_name} constructed inf. sols. Default to GORT default construction (SAVINGS).")
            # check if not surpassed construction time budget and still have time for search in Time Budget
            time_for_ls = self.per_instance_time_limit_ls if self.per_instance_time_limit_ls is not None \
                else np.mean([d.time_limit for d in self.ds.data])
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
                logger.info(f"Model {construct_name} used up runtime (on avg {np.mean(time_constr)}) for constructing "
                            f"(time limit {self.time_limit}). Using constructed solution for Evaluation.")
                self.acronym = construct_name
        else:
            logger.info(f"Run-time dependent parameters: {self.device} Device, "
                        f"Adjusted Time Budget for construction: {self.per_instance_time_limit_constr} / instance.")
            logger.info(f"running test inference for {self.acronym} as construction method...")
            _, sols_ = eval_model(
                Tester=TSPTester if self.cfg.problem.upper() == "TSP" else CVRPTester,
                env=self.env,
                model=self.model,
                data=self.ds.data,
                device=self.device,
                tester_cfg=self.cfg.tester_cfg,
            )
            logger.info(f"finished.")

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
            raise NotImplementedError(f"Unknown problem class: '{cfg.problem.upper()}' for model {self.acronym}"
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

    def get_train_val_set(self, cfg):
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
            device=self.device,  # transform_func=make_cvrp_instance,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )

        ds_val = dataset_class(
            is_train=True,
            store_path=cfg.train_cfg.val_dataset if cfg.train_cfg.val_dataset is not None else None,
            num_samples=cfg.train_cfg.val_size,
            device=self.device,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            verbose=self.debug > 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        return ds, ds_val

    def get_acronym(self, model_name: str, pomo_cfg: DictConfig):
        acronym, acronym_ls = model_name, None
        augmentation = str(pomo_cfg.aug_factor) if pomo_cfg.augmentation_enable else 'no'
        pomo_size = str(pomo_cfg.pomo_size)
        if self.cfg.run_type in ["val", "test"]:
            model_name_aug = model_name + '_aug_' + augmentation
            # if augmentation != 'no':
            #     model_name_aug = model_name + '_aug_' + augmentation
            if pomo_size == '1':
                model_name_aug = model_name_aug + '_greedy'
                logger.info(f"Evaluating POMO with greedy rollout")
            else:
                logger.info(f"Evaluation POMO with sampling - sampling size set to {pomo_size}")
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                acronym = model_name_aug + '_' + acronym_ls
            else:
                acronym = model_name_aug
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
    if cfg.run_type in ["train", "resume"] and cfg.train_cfg.model_load.path is not None:
        cfg.train_cfg.model_load.path = os.path.normpath(
            os.path.join(cwd, cfg.train_cfg.model_load.path)
        )
    if cfg.checkpoint_load_path is not None:
        cfg.checkpoint_load_path = os.path.normpath(
            os.path.join(cwd, cfg.checkpoint_load_path)
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
