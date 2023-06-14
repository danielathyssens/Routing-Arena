#
import os
import shutil
import logging
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
import warnings

import random
import numpy as np
import hydra
import torch
import time

from models.DACT.DACT.problems.problem_vrp import CVRP
from models.DACT.DACT.problems.problem_tsp import TSP
from models.DACT.DACT.agent.ppo import PPO
from models.DACT.dact import train_model, eval_model
from data.tsp_dataset import TSPDataset
from data.cvrp_dataset import CVRPDataset
from models.runner_utils import get_stats, set_device, print_summary_stats, log_info, _adjust_time_limit, \
    eval_inference, set_passMark, get_time_limit
from formats import RPSolution
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
        self.acronym = self.get_acronym(model_name="DACT")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))
        # re-label model internal run_name (used to save ckpts)
        self.cfg.run_name = self.run_name

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # set device
        self.device = set_device(self.cfg)

        # set PassMark for eval
        self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device, number_threads=1)
        self.per_instance_time_limit = None
        self.machine_info = None

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

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_problem()
        self._build_env()
        self._build_policy()
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.data_file_path is not None and self.passMark is not None \
                    and self.cfg.test_cfg.eval_type != "simple":
                assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                    f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                    f"or set test_cfg.eval_type to 'simple'"
                self.init_metrics(self.cfg)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.passMark,
                              TimeLimit_=self.time_limit,
                              passMark_cpu=self.CPU_passMark,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              verbose=self.debug >= 1)

        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _dir_setup(self):
        """ Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # val log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        os.makedirs(self.cfg.checkpoint_save_path, exist_ok=True)

    def _build_problem(self):
        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()

        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg)
        elif cfg.run_type in ["train", "resume"]:
            self.ds, self.ds_val = self.get_train_val_set(cfg)

    def _build_env(self, p_size_=None):
        # init problem state
        cfg = self.cfg.env_cfg.copy()
        ENV = CVRP if self.cfg.problem.upper() == "CVRP" else TSP
        p_size = self.cfg.graph_size if self.cfg.graph_size is not None else self.ds.data[0].graph_size - 1
        if self.cfg.graph_size is None:
            warnings.warn(f"Setting p_size in DACT ENV to self.ds.data[0].graph_size={self.ds.data[0].graph_size}. "
                          f"DACT ENV is only working for sets of instances that are of the same problem size!")
        self.env = ENV(
            p_size=p_size if p_size_ is None else p_size_,
            step_method=cfg.step_method,
            init_val_met=cfg.init_val_met,
            with_assert=self.debug >= 1,
            P=cfg.perturb_eps,
            DUMMY_RATE=cfg.dummy_rate
        )
        logger.info(f"Builded Env for {self.acronym}")

    def _build_policy(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        cfg = self.cfg.copy()
        cfg.use_cuda = self.cfg.cuda and torch.cuda.is_available()
        cfg.world_size = torch.cuda.device_count()
        cfg.distributed = (torch.cuda.device_count() > 1) and (not cfg.no_DDP)
        cfg.device = self.device.type
        cfg.no_progress_bar = False
        self.policy = PPO(self.env.NAME, self.env.size, cfg)

        logger.info(f"Builded Policy for {self.acronym}")

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
        self.setup()
        # model-internal setting
        self.cfg.eval_only = True
        # load policy
        self.policy.load(self.cfg.test_cfg.checkpoint_load_path)

        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
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

    def run_inference(self) -> List[RPSolution]:
        # single_thread - not sure if CPU implementation of DACT is using only single thread.
        logger.info(f"Run-time dependent parameters: {self.device} Device, "
                    f"Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        eval_single = False
        if (self.cfg.test_cfg.data_file_path[-1] == "X") or (self.cfg.coords_dist == "golden"):
            eval_single = True
        if eval_single:
            summary_res_all, rp_sols_all = [], []
            for data_instance in self.ds.data:
                # self.env = \
                self._build_env(p_size_=data_instance.graph_size - 1)
                # policy =
                self._build_policy()
                summary_res, sols = eval_model(
                    data=[data_instance],
                    problem=self.env,
                    agent=self.policy,
                    time_limit=data_instance.time_limit,
                    opts=self.cfg.copy(),
                    dummy_rate=self.cfg.env_cfg.dummy_rate,
                    device=self.device,
                    batch_size=self.cfg.test_cfg.batch_size,
                )
                summary_res_all.extend(summary_res)
                rp_sols_all.extend(sols)
            return rp_sols_all
        else:

            _, solutions_ = eval_model(
                data=self.ds.data,
                problem=self.env,
                agent=self.policy,
                time_limit=self.per_instance_time_limit,
                opts=self.cfg.copy(),
                dummy_rate=self.cfg.env_cfg.dummy_rate,
                device=self.device,
                batch_size=self.cfg.test_cfg.batch_size,
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

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def get_acronym(self, model_name: str):
        acronym = model_name
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.T_max == 1000:
                acronym = model_name + '_1k'
            if self.cfg.T_max == 5000:
                acronym = model_name + '_5k'
            if self.cfg.T_max == 10000:
                acronym = model_name + '_10k'

        return acronym

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

        # BUILD TEST DATASET
        ds = dataset_class(
            store_path=cfg.test_cfg.data_file_path if 'data_file_path' in list(cfg.test_cfg.keys()) else None,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            dataset_size=cfg.test_cfg.dataset_size,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            TimeLimit=self.time_limit,
            machine_info=self.machine_info,
            load_bks=load_bks,
            load_base_sol=load_base_sol,
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
        # BUILD TRAIN AND VALIDATION DATASET CLASSES
        ds = dataset_class(
            is_train=True,
            distribution=self.cfg.coords_dist,
            graph_size=self.cfg.graph_size,
            device=self.device,
            seed=self.cfg.global_seed,
            verbose=self.debug >= 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args)

        ds_val = dataset_class(
            is_train=True,
            store_path=cfg.val_dataset if 'val_dataset' in list(cfg.keys()) else None,
            num_samples=cfg.val_size,
            distribution=cfg.coords_dist,
            device=self.device,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            verbose=self.debug >= 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        ).sample(**self.cfg.env_kwargs.sampling_args)
        torch.save(ds_val, "val_dataset_for_train_run.pt")
        return ds, ds_val

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
    # if 'data_file_path' in list(cfg.keys()) and cfg.test_cfg.data_file_path is not None:
    #     cfg.test_cfg.data_file_path = os.path.normpath(
    #         os.path.join(cwd, cfg.test_cfg.data_file_path)
    #     )
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

    if cfg.test_cfg.checkpoint_load_path is not None:
        cfg.test_cfg.checkpoint_load_path = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
        )
    # if cfg.save_dir is not None:
    #    cfg.save_dir = os.path.normpath(
    #         os.path.join(cwd, cfg.save_dir)
    #     )
    return cfg


def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i + len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
