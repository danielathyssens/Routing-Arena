#
import os
import shutil
import logging
from warnings import warn
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import hydra
import torch
import time

from formats import CVRPInstance, RPSolution
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from models.NLNS.nlns import eval_model, train_model, prep_data_NLNS
from models.runner_utils import _adjust_time_limit, print_summary_stats, set_device, set_passMark, eval_inference, get_time_limit
from formats import CVRPInstance, RPSolution
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS

from models.NLNS.NLNS.actor import VrpActorModel
from models.NLNS.NLNS.critic import VrpCriticModel

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
        self.acronym = self.get_acronym("NLNS")
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
        self.per_instance_time_limit = None
        self.machine_info = None

        # set PassMark for eval
        self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device,
                                                        self.cfg.lns_nb_cpus)
        if cfg.run_type in ["val", "test"]:
            # get Time Budget
            self.time_limit = get_time_limit(self.cfg)
            if self.time_limit is not None:
                # get normalized per instance Time Limit
                self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device,
                                                                  cfg.lns_nb_cpus)
                logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                            f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
            else:
                self.per_instance_time_limit = None
                self.machine_info = (self.passMark, self.CPU_passMark, self.device, cfg.lns_nb_cpus, False)
                logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")


    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self.seed_all(self.cfg.global_seed)
        self._build_problem()
        self._build_model()
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.data_file_path is not None and self.passMark is not None \
                    and self.cfg.test_cfg.eval_type != "simple":
                assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                    f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                    f"or set test_cfg.eval_type to 'simple'"
                self.init_metrics(self.cfg)

    def _dir_setup(self):
        """ Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
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
                              single_thread=cfg.lns_nb_cpus == 1,
                              verbose=self.debug >= 1)

        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _build_problem(self):
        """Load dataset and create environment (problem state and data)."""
        cfg = self.cfg.copy()

        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg)
        elif cfg.run_type in ["train", "resume"]:
            self.ds, self.ds_val = self.get_train_val_set(cfg)
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def _build_model(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        cfg = self.cfg.copy()
        cfg.use_cuda = self.cfg.cuda and torch.cuda.is_available()
        cfg.distributed = False
        cfg.no_saving = True
        cfg.device = self.device.type
        cfg.no_progress_bar = False

        if cfg.run_type == "train":
            self.train_cfg = cfg.train_opts_cfg
            self.train_cfg['output_path'] = cfg.checkpoint_save_path

        self.actor_model = VrpActorModel(device=cfg.device,
                                         hidden_size=cfg.pointer_hidden_size).to(cfg.device)

        self.critic_model = VrpCriticModel(cfg.critic_hidden_size).to(cfg.device)

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def train(self, **kwargs):
        """Train the specified model."""

        # agent.start_training(problem, opts.val_dataset, tb_logger)

        logger.info(f"start training...")
        results, solutions = train_model(
            train_dataclass=self.ds,
            val_dataclass=self.ds_val,
            actor=self.actor_model,
            critic=self.critic_model,
            train_cfg=self.train_cfg,
            run_id=self.cfg.train_opts_cfg.run_name
        )
        logger.info(f"training finished.")
        logger.info(results)
        # solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        # self.save_results({
        #     "solutions": solutions,
        #     "summary": summary
        # })
        # logger.info(summary)

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        if not self.cfg.problem.upper() in (["CVRP"] or ["SDVRP"]):
            raise NotImplementedError

        self.cfg.eval_only = True
        self.setup()

        # run test inference
        # default to a single run if number of runs not specified
        number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
        results_all, stats_all = [], []
        logger.info(f"Run-time dependent parameters: {self.device} Device + cpu Device (threads: "
                    f"{self.cfg.lns_nb_cpus} used for local search), "
                    f"Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
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
        _, solutions_ = eval_model(
            data=self.ds.data,
            data_type=self.cfg.data_key,
            normalized=self.cfg.normalize_data,
            model_path=self.cfg.test_cfg.checkpoint_load_path,
            instance_path=self.cfg.data_file_path,
            problem=self.cfg.problem,
            int_prec=self.cfg.integer_precision,
            batch_size=self.cfg.lns_batch_size,
            time_limit=self.per_instance_time_limit,
            opts=self.cfg.copy()
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

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        self.setup()
        # self.policy.load(self.cfg.test_cfg.checkpoint_load_path)
        epoch_resume = int(os.path.splitext(os.path.split(self.cfg.test_cfg.checkpoint_load_path)[-1])[0].split("-")[1])
        logger.info(f"Resuming after {epoch_resume}")
        # self.policy.opts.epoch_start = epoch_resume + 1

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        self.train(**kwargs)

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
        if cfg.test_cfg.eval_type != "simple":
            load_bks = True
            if cfg.test_cfg.eval_type == "wrap" or "wrap" in cfg.test_cfg.eval_type:
                load_base_sol = True
            else:
                load_base_sol = False
        else:
            load_bks, load_base_sol = False, False

        if cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be 'CVRP'")

        ds = dataset_class(
            store_path=cfg.test_cfg.data_file_path if 'data_file_path' in list(cfg.test_cfg.keys()) else None,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            dataset_size=cfg.test_cfg.dataset_size,
            normalize=cfg.normalize_data,
            seed=cfg.global_seed,
            verbose=self.debug > 1,
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
                                      f"Must be 'CVRP'")
        ds = dataset_class(
            is_train=True,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            seed=cfg.global_seed,
            transform_func=prep_data_NLNS,
            transform_args=cfg.train_opts_cfg.transform_args,
            normalize=True,
            verbose=self.debug >= 1,
            device=self.device,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        ds_val = dataset_class(
            is_train=True,
            store_path=cfg.val_dataset,  # default is None --> so generate ds_val
            num_samples=cfg.val_size,
            distribution=cfg.coords_dist,
            graph_size=cfg.graph_size,
            transform_func=prep_data_NLNS,
            transform_args=cfg.train_opts_cfg.transform_args,
            normalize=True,
            seed=cfg.global_seed,
            verbose=self.debug > 1,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )

        return ds, ds_val

    def get_acronym(self, model_name: str):
        acronym = model_name
        if self.cfg.run_type in ["val", "test"]:
            # .test_cfg
            if self.cfg.mode == 'eval_single':
                acronym = model_name + '_single_search'
            elif self.cfg.mode == 'eval_batch':
                # .test_cfg
                acronym = model_name + '_batch_search'

        return acronym

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

    if cfg.test_cfg.checkpoint_load_path is not None:
        cfg.test_cfg.checkpoint_load_path = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
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
