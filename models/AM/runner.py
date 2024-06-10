#
import os, sys
# abs_path = os.path.abspath('models/AM/AM')   # otherwise earlier trained checkpoint not loadable
# sys.path.insert(0, abs_path)

import logging
from abc import ABC
# from abc import ABC
from typing import Optional, Dict, Union, List, Tuple
from omegaconf import DictConfig, OmegaConf
from warnings import warn
import time

import random
import numpy as np
import hydra
import torch
# from tensorboard_logger import Logger as TbLogger

# from lib.routing import RPDataset, RPInstance, RPSolution, eval_rp
# from .models.runner_utils import update_path
from models.AM.AM.attention_model import AttentionModel as AttentionModel_tr
from models.AM.AM.nets.attention_model import AttentionModel
from models.AM.AM.utils.functions import load_args, load_problem, _load_model_file


from models.AM.am import make_AM_instance, train_model, eval_model
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from models.runner_utils import _adjust_time_limit, merge_sols, print_summary_stats, set_device, set_passMark, \
    eval_inference, get_time_limit
from formats import RPSolution
# from metrics.metrics import Metrics
# from models.runner_utils import NORMED_BENCHMARKS
from models.runners import BaseConstructionRunner

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


class Runner(BaseConstructionRunner, ABC):
    """wraps setup, training, testing of respective model
        experiments according to cfg"""

    def __init__(self, cfg: DictConfig):

        super(Runner, self).__init__(cfg)

        # Model acronym
        # option for construction models to run with local search on top
        self.acronym, self.acronym_ls = self.get_acronym(model_name="AM")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup(compatible_problems=DATA_CLASS, data_transformation=make_AM_instance)
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
        cfg = self.cfg.copy()
        if self.cfg.run_type in ["train", "debug"]:
            self.model = AttentionModel_tr(is_train=True,
                                           problem=cfg.problem,
                                           model_opts=cfg.train_opts_cfg,
                                           n_epochs=cfg.train_opts_cfg.n_epochs,
                                           device=self.device).to(self.device)
        else:
            args = load_args(self.cfg.trained_model_args_path)
            problem = load_problem(self.cfg.problem)
            self.model = AttentionModel(
                args['embedding_dim'],
                args['hidden_dim'],
                problem,
                n_encode_layers=args['n_encode_layers'],
                mask_inner=True,
                mask_logits=True,
                normalization=args['normalization'],
                tanh_clipping=args['tanh_clipping'],
                checkpoint_encoder=args.get('checkpoint_encoder', False),
                shrink_size=args.get('shrink_size', None)
            )
            self.model.eval()
            self.model.to(self.device)

    def _run_model(self) -> Tuple[Dict, List[RPSolution]]:
        """Run the model and get solutions for the RP - executed in run_inference"""
        return eval_model(problem=self.cfg.problem,
                          model=self.model,
                          data_rp=self.ds.data,
                          eval_batch_size=self.cfg.eval_opts_cfg.eval_batch_size,
                          eval_opts=self.cfg.eval_opts_cfg,
                          decode_strategy=self.cfg.test_cfg.decode_type,
                          device=self.device)

    def train(self, **kwargs):
        """Train the AM model."""

        # raise NotImplementedError
        cfg = self.cfg.copy()
        # Optionally configure tensorboard
        tb_logger = None
        # tb_logging currently not working
        # if cfg.tb_logging:
        #     tb_logger = TbLogger(
        #         os.path.join(cfg.tb_log_path, "{}_{}".format(cfg.problem, cfg.graph_size), self.run_name))

        logger.info(f"start training...")
        results = self.model.train_model(
            train_dataset=self.ds,  # from which to sample each epoch
            val_dataset=self.val_data,  # fixed
            ckpt_save_path=cfg.checkpoint_save_path,
            opts=cfg.train_opts_cfg,
            tb_logger=tb_logger,
            coords_distribution=cfg.env_kwargs.generator_args.coords_sampling_dist,
            sampling_args=cfg.env_kwargs.sampling_args,
            generator_args=cfg.env_kwargs.generator_args
        )
        logger.info(f"training finished.")

    def resume(self):
        pass

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""

        self.setup(compatible_problems=DATA_CLASS, data_transformation=make_AM_instance)
        # self.model.load(self.cfg.test_cfg.checkpoint_load_path) # old
        # Overwrite model parameters by parameters to load
        load_data = torch.load(self.cfg.test_cfg.checkpoint_load_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict({**self.model.state_dict(), **load_data.get('model', {})})

        model, *_ = _load_model_file(self.cfg.test_cfg.checkpoint_load_path, self.model)

        model.eval()  # Put in eval mode

        results, summary = self.run_test()

    def get_acronym(self, model_name: str):
        acronym, acronym_ls = model_name, None
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                # acronym_ls = 'GORT_' + str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                if self.cfg.test_cfg.decode_type == 'greedy':
                    acronym = model_name + '_greedy_' + acronym_ls
                elif self.cfg.test_cfg.decode_type == 'sample':
                    acronym = model_name + '_sample_' + acronym_ls
            else:
                if self.cfg.test_cfg.decode_type == 'greedy':
                    acronym = model_name + '_greedy'
                elif self.cfg.test_cfg.decode_type == 'sample':
                    acronym = model_name + '_sample'
        return acronym, acronym_ls

    def _update_path(self, cfg: DictConfig):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()

        # cfg.run_type in ["val", "test"] and
        if 'data_file_path' in list(cfg.test_cfg.keys()) and cfg.test_cfg.data_file_path is not None:
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
        if cfg.trained_model_args_path is not None:
            cfg.trained_model_args_path = os.path.normpath(
                os.path.join(cwd, cfg.trained_model_args_path)
            )
        return cfg
#
#
# def remove_dir_tree(root: str, pth: Optional[str] = None):
#     """Remove the full directory tree of the root directory if it exists."""
#     if not os.path.isdir(root) and pth is not None:
#         # select root directory from path by dir name
#         i = pth.index(root)
#         root = pth[:i + len(root)]
#     if os.path.isdir(root):
#         shutil.rmtree(root)


import io
import pickle


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == 'models.AM.nets':
            renamed_module = 'models.AM.AM.nets'

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
