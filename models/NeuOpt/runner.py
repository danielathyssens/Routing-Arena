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

from models.NeuOpt.NeuOpt.problems.problem_cvrp import CVRP
from models.NeuOpt.NeuOpt.problems.problem_tsp import TSP
from models.NeuOpt.NeuOpt.agent.ppo import PPO
from models.NeuOpt.neuopt import train_model, eval_model
from data.tsp_dataset import TSPDataset
from data.cvrp_dataset import CVRPDataset
from models.runner_utils import get_stats, set_device, print_summary_stats, log_info, _adjust_time_limit, \
    eval_inference, set_passMark, get_time_limit
from formats import RPSolution
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

        # Model acronym
        self.acronym = self.get_acronym(model_name="NeuOpt")

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

    def _build_env(self, p_size_=None):
        # init problem state
        cfg = self.cfg.env_cfg.copy()
        ENV = CVRP if self.cfg.problem.upper() == "CVRP" else TSP
        p_size = self.cfg.graph_size if self.cfg.graph_size is not None else self.ds.data[0].graph_size - 1
        if self.cfg.graph_size is None:
            warnings.warn(f"Setting p_size in NeuOpt ENV to self.ds.data[0].graph_size={self.ds.data[0].graph_size}. "
                          f"NeuOpt ENV is only working for sets of instances that are of the same problem size!")
        self.env = ENV(
            p_size=p_size if p_size_ is None else p_size_,
            # step_method=cfg.step_method,
            init_val_met=cfg.init_val_met,
            with_assert=self.debug >= 1,
            # P=cfg.perturb_eps,
            DUMMY_RATE=cfg.dummy_rate if self.cfg.problem.upper() == "CVRP" else None  # only for CVRP
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
        # self.policy = PPO(self.env.NAME, self.env.size, cfg)
        self.policy = PPO(self.env, cfg)

        logger.info(f"Builded Policy for {self.acronym}")

    def _run_model(self):
        eval_single = False
        if self.cfg.test_cfg.data_file_path is not None and \
                (self.cfg.test_cfg.data_file_path[-1] == "X") or (self.cfg.coords_dist == "golden"):
            eval_single = True
        if eval_single:
            summary_res_all, rp_sols_all = [], []
            for data_instance in self.ds.data:
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
            return {}, rp_sols_all
        else:
            return eval_model(
                data=self.ds.data,
                problem=self.env,
                agent=self.policy,
                time_limit=self.per_instance_time_limit,
                opts=self.cfg.copy(),
                dummy_rate=self.cfg.env_cfg.dummy_rate,
                device=self.device,
                batch_size=self.cfg.test_cfg.batch_size,
            )

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
        self.setup(compatible_problems=DATA_CLASS)
        # model-internal setting
        self.cfg.eval_only = True
        # load policy
        self.policy.load(self.cfg.test_cfg.checkpoint_load_path)

        results, summary = self.run_test()

        # default to a single run if number of runs not specified
        # number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        # results_all, stats_all = [], []
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

    # def run_inference(self) -> List[RPSolution]:
    #     # single_thread - not sure if CPU implementation of DACT is using only single thread.
    #     logger.info(f"Run-time dependent parameters: {self.device} Device, "
    #                 f"Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
    #     eval_single = False
    #     if (self.cfg.test_cfg.data_file_path[-1] == "X") or (self.cfg.coords_dist == "golden"):
    #         eval_single = True
    #     if eval_single:
    #         summary_res_all, rp_sols_all = [], []
    #         for data_instance in self.ds.data:
    #             # self.env = \
    #             self._build_env(p_size_=data_instance.graph_size - 1)
    #             # policy =
    #             self._build_policy()
    #             summary_res, sols = eval_model(
    #                 data=[data_instance],
    #                 problem=self.env,
    #                 agent=self.policy,
    #                 time_limit=data_instance.time_limit,
    #                 opts=self.cfg.copy(),
    #                 dummy_rate=self.cfg.env_cfg.dummy_rate,
    #                 device=self.device,
    #                 batch_size=self.cfg.test_cfg.batch_size,
    #             )
    #             summary_res_all.extend(summary_res)
    #             rp_sols_all.extend(sols)
    #         return rp_sols_all
    #     else:
    #
    #         _, solutions_ = eval_model(
    #             data=self.ds.data,
    #             problem=self.env,
    #             agent=self.policy,
    #             time_limit=self.per_instance_time_limit,
    #             opts=self.cfg.copy(),
    #             dummy_rate=self.cfg.env_cfg.dummy_rate,
    #             device=self.device,
    #             batch_size=self.cfg.test_cfg.batch_size,
    #         )
    #     return solutions_
    #
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
    # def save_results(self, result: Dict, run_id: int = 0):
    #     pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
    #     torch.save(result, pth)

    def get_acronym(self, model_name: str):
        acronym = model_name
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.T_max == 100:
                acronym = model_name + '_100'
            if self.cfg.T_max == 1000:
                acronym = model_name + '_1k'
            if self.cfg.T_max == 5000:
                acronym = model_name + '_5k'
            if self.cfg.T_max == 10000:
                acronym = model_name + '_10k'

        return acronym

    # def get_test_set(self, cfg):
    #     if cfg.problem.upper() in DATA_CLASS.keys():
    #         dataset_class = DATA_CLASS[cfg.problem.upper()]
    #     else:
    #         raise NotImplementedError(f"Unknown problem class: '{cfg.problem.upper()}' for model {self.acronym}"
    #                                   f"Must be ['TSP', 'CVRP']")
    #     if cfg.test_cfg.eval_type != "simple":
    #         load_bks = True
    #         if cfg.test_cfg.eval_type == "wrap" or "wrap" in cfg.test_cfg.eval_type:
    #             load_base_sol = True
    #         else:
    #             load_base_sol = False
    #     else:
    #         load_bks, load_base_sol = False, False
    #
    #     # BUILD TEST DATASET
    #     ds = dataset_class(
    #         store_path=cfg.test_cfg.data_file_path if 'data_file_path' in list(cfg.test_cfg.keys()) else None,
    #         distribution=cfg.coords_dist,
    #         graph_size=cfg.graph_size,
    #         dataset_size=cfg.test_cfg.dataset_size,
    #         seed=cfg.global_seed,
    #         verbose=self.debug >= 1,
    #         TimeLimit=self.time_limit,
    #         machine_info=self.machine_info,
    #         load_bks=load_bks,
    #         load_base_sol=load_base_sol,
    #         sampling_args=cfg.env_kwargs.sampling_args,
    #         generator_args=cfg.env_kwargs.generator_args
    #     )
    #     return ds
    #
    # def get_train_val_set(self, cfg):
    #     if cfg.problem.upper() in DATA_CLASS.keys():
    #         dataset_class = DATA_CLASS[cfg.problem.upper()]
    #     else:
    #         raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
    #                                   f"Must be ['TSP', 'CVRP']")
    #     # BUILD TRAIN AND VALIDATION DATASET CLASSES
    #     ds = dataset_class(
    #         is_train=True,
    #         distribution=self.cfg.coords_dist,
    #         graph_size=self.cfg.graph_size,
    #         device=self.device,
    #         seed=self.cfg.global_seed,
    #         verbose=self.debug >= 1,
    #         sampling_args=cfg.env_kwargs.sampling_args,
    #         generator_args=cfg.env_kwargs.generator_args)
    #
    #     ds_val = dataset_class(
    #         is_train=True,
    #         store_path=cfg.val_dataset if 'val_dataset' in list(cfg.keys()) else None,
    #         num_samples=cfg.val_size,
    #         distribution=cfg.coords_dist,
    #         device=self.device,
    #         graph_size=cfg.graph_size,
    #         seed=cfg.global_seed,
    #         verbose=self.debug >= 1,
    #         sampling_args=cfg.env_kwargs.sampling_args,
    #         generator_args=cfg.env_kwargs.generator_args
    #     ).sample(**self.cfg.env_kwargs.sampling_args)
    #     torch.save(ds_val, "val_dataset_for_train_run.pt")
    #     return ds, ds_val
    #
    # @staticmethod
    # def seed_all(seed: int):
    #     """Set seed for all pseudo random generators."""
    #     # will set some redundant seeds, but better safe than sorry
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

    def _update_path(self, cfg: DictConfig, fixed_dataset: bool = True):
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
