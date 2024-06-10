import os
import logging
import shutil
import warnings
from typing import Optional, Dict, Union, List, Tuple
from omegaconf import DictConfig, OmegaConf
from warnings import warn
import json

import random
import time
import numpy as np
import hydra
import torch
from torch import Tensor, nn
from tensorboard_logger import Logger as TbLogger
from pynvml import *



from models.BQ.bq import eval_model, prep_data_BQ  # train_model
from models.BQ.bq_nco.utils.chekpointer import CheckPointer
from models.BQ.bq_nco.model.model import BQModel

from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from formats import CVRPInstance, RPSolution
from models.runners import BaseConstructionRunner

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


class Runner(BaseConstructionRunner):
    """wraps setup, training, testing of respective model
        experiments according to cfg"""

    def __init__(self, cfg: DictConfig):

        super(Runner, self).__init__(cfg)
        # fix path aliases changed by hydra
        # self.cfg = update_path(cfg)
        # OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        # option for construction models to run with local search on top
        self.acronym, self.acronym_ls = self.get_acronym(model_name="BQ")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        self.checkpointer = None

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup(compatible_problems=DATA_CLASS, data_transformation=prep_data_BQ)
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'val', 'test', 'debug']")

    def _build_model(self):
        """Infer and set the model/model arguments provided to the learning algorithm."""
        model_cfg = self.cfg.model_cfg.copy()
        # if cfg.run_type in ["train", "debug", "resume"]:
        if self.cfg.problem == "tsp":
            node_input_dim = 2
        else:
            node_input_dim = 4
        # elif self.cfg.problem == "kp":
        #     node_input_dim = 3
        # elif self.cfg.problem == "cvrp" or self.cfg.problem == "op":

        self.model = BQModel(node_input_dim,
                             model_cfg.dim_emb,
                             model_cfg.dim_ff,
                             model_cfg.activation_ff,
                             model_cfg.nb_layers_encoder,
                             model_cfg.nb_heads,
                             model_cfg.activation_attention,
                             model_cfg.dropout,
                             model_cfg.batchnorm,
                             self.cfg.problem.lower())

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # device = self.device # 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def _run_model(self) -> Tuple[Dict, List[RPSolution]]:
        """Run the model and get solutions for the RP - executed in run_inference"""
        return eval_model(net=self.model,
                          checkpointer=self.checkpointer,
                          module=self.module,
                          data_rp=self.ds.data,
                          problem=self.cfg.problem,
                          batch_size=self.cfg.test_cfg.eval_batch_size,
                          device=self.device,
                          opts=self.cfg.eval_opts_cfg)

    def train(self, **kwargs):
        """Train the specified model."""

        cfg = self.cfg.copy()

        # Optionally configure tensorboard
        tb_logger = None
        if cfg.tb_logging:
            tb_logger = TbLogger(
                os.path.join(cfg.tb_log_path, "{}_{}".format(cfg.problem, cfg.graph_size), self.run_name))

        logger.info(f"start training on {self.device}...")
        results = train_model(
            model=self.model,
            problem=cfg.problem,
            device=self.device,
            baseline_type=cfg.baseline_cfg.baseline_type,
            resume=False,
            opts=cfg.train_opts_cfg,
            train_dataset=self.ds,  # from which to sample each epoch
            val_dataset=self.val_data,  # fixed
            ckpt_save_path=cfg.checkpoint_save_path,
            tb_logger=tb_logger,
            **cfg.env_kwargs.sampling_args
        )

        logger.info(f"training finished.")
        logger.info(f"Last results: {results[-1]}")
        # logger.info(results)
        # solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        # self.save_results({
        #    "solutions": solutions,
        #    "summary": summary
        # })
        # logger.info(summary)

    def resume(self):
        """Resume training procedure of MDAM"""
        cfg = self.cfg.copy()
        if cfg.test_cfg.checkpoint_load_path is not None:
            epoch_resume = int(os.path.splitext(os.path.split(cfg.test_cfg.checkpoint_load_path)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            epoch_start = epoch_resume + 1

            logger.info(f"resuming training...")
            _ = train_model(
                model=self.model,
                val_data_rp=self.ds_val.data,
                problem=cfg.problem,
                device=self.device,
                resume=True,
                resume_pth=cfg.test_cfg.checkpoint_load_path,
                epoch_start=epoch_start,
                opts=cfg.train_opts_cfg,
            )
            logger.info(f"training finished.")

        else:
            warnings.warn("No path specified to load data for resuming training. Default to normal training?")

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        assert self.cfg.problem.upper() in ["CVRP", "TSP"], "Only TSP and CVRP are implemented currently"
        self.setup(compatible_problems=DATA_CLASS, data_transformation=prep_data_BQ)

        optimizer = None
        # print('self.cfg.checkpoint_load_path', self.cfg.checkpoint_load_path)

        # self.cfg.test_cfg.checkpoint_load_path = os.path.normpath(
        #     os.path.join(hydra.utils.get_original_cwd(), self.cfg.test_cfg.checkpoint_load_path)
        # )

        if self.cfg.test_cfg.checkpoint_load_path != "":
            path = self.cfg.test_cfg.checkpoint_load_path
            model_dir, name = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
            self.checkpointer = CheckPointer(name=name, save_dir=model_dir)
            _, other = self.checkpointer.load(self.module, optimizer, label='best', map_location=self.device)
        # else:
        #     model_dir, name = os.path.join(args.output_dir, "models"), f'{int(time.time() * 1000.0)}'
        #     checkpointer = CheckPointer(name=name, save_dir=model_dir)
        #     other = None

        results, summary = self.run_test()

    def get_acronym(self, model_name):
        acronym, acronym_ls = model_name, None
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                # acronym_ls = 'GORT_' + str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                if self.cfg.test_cfg.beam == 1:
                    acronym = model_name + '_greedy' + '_' + acronym_ls
                elif self.cfg.test_cfg.beam != 1:
                    acronym = model_name + '_beam_'+str(int(self.cfg.test_cfg.beam)) + '_' + acronym_ls
            else:
                if self.cfg.test_cfg.beam == 1:
                    acronym = model_name + '_greedy'
                elif self.cfg.test_cfg.beam != 1:
                    acronym = model_name + '_beam_'+str(int(self.cfg.test_cfg.beam))
        return acronym, acronym_ls

    def _update_path(self, cfg):
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