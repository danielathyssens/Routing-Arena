import torch
import numpy as np
from abc import abstractmethod
from omegaconf import DictConfig
from typing import Union, Tuple, List, Callable
import warnings
import os
import pickle
import logging
from pathlib import Path

from torch.utils.data import Dataset
from formats import CVRPInstance, RPSolution
from data.generator import RPGenerator
from data.data_utils import format_ds_save_path
from metrics.metrics import Metrics
from models.runner_utils import _adjust_time_limit as adjust_TL
import matplotlib.pyplot as plt
from models.runner_utils import NORMED_BENCHMARKS

# from data.BKS import BKS_cvrp_kool, BKS_tsp_kool

logger = logging.getLogger(__name__)

DATA_KEYWORDS = {
    'uniform': 'uniform',
    'nazari': 'uniform',
    'rej': 'rejection_sampled',
    'uchoa': 'uchoa_distributed',
    'tsplib': 'tsplib_format',
    'homberger': 'homberger_200',
    'XE': 'XE',
    'S': 'S'
}

TEST_SETS_BKS = ['cvrp20_test_seed1234.pkl',
                 'cvrp50_test_seed1234.pkl',
                 'cvrp100_test_seed1234.pkl',
                 'val_seed123_size512.pt',
                 'val_seed123_size512.pkl',
                 'val_seed4321_size512.pkl',
                 'E_R_6_seed123_size512.pt',
                 'XE',
                 'X',
                 'XML100',
                 'subsampled',
                 'Golden']


class BaseDataset(Dataset):
    """
    Custom pytorch Dataset class that is inherited by all problem-specific datasets to create/sample data.

    Args:
        store_path: path to store data when downloaded or to check if data is there (mainly for test)
        needs_prep: whether imported data needs preprocessing
    """

    def __init__(self,
                 problem: str = None,
                 store_path: str = None,
                 num_samples: int = 100,
                 graph_size: int = 20,
                 distribution: str = None,
                 generator_args: dict = None,
                 sampling_args: dict = None,
                 float_prec: np.dtype = np.float32,
                 transform_func: Callable = None,
                 transform_args: DictConfig = None,
                 seed: int = None,
                 verbose: bool = False,
                 normalize: bool = True,
                 TimeLimit: Union[list, int, float] = None,
                 load_bks: bool = True,
                 load_base_sol: bool = True,
                 **kwargs):
        super(BaseDataset, self).__init__()

        if store_path is not None:
            logger.info(f"Test Data provided, No new samples are generated.")

        self.verbose = verbose
        self.problem = problem
        self.store_path = store_path
        self.num_samples = num_samples
        self.graph_size = graph_size
        self.distribution = distribution
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.normalize = normalize
        # self.passmark = pass_mark
        # self.passmark_cpu = pass_mark_cpu
        # self.single_thread = single_thread
        # self.cpu_search_on_top = add_cpu_search
        self.time_limit = TimeLimit
        self.transform_func = transform_func
        self.transform_args = transform_args
        self.save_traj_flag = True
        if self.store_path is None:
            logger.info(f"Initiating RPGenerator with {self.generator_args}")
            self.gen = RPGenerator(seed, self.verbose, float_prec, self.generator_args)
        self.size = None
        self.data = None
        self.data_transformed = None
        self.data_key = None
        self.bks_path = None
        self.base_sol_path = None
        self.bks, self.BaseSol = None, None
        if load_bks:
            self.bks = self.load_BKS_BaseSol("BKS") if self.store_path is not None else None
        if load_base_sol:
            self.BaseSol = self.load_BKS_BaseSol("BaseSol") if self.store_path is not None else None
        if self.bks is not None:
            logger.info(f'Loaded {len(self.bks)} BKS for the test (val) set.')
        elif self.bks is None and self.store_path:
            logger.info(f'No BKS loaded for dataset {self.store_path}.')
        self.metric = None  # gets initialized in runner if cfg.eval_type != "simple"
        self.adjusted_time_limit = None  # gets initialized in runner if cfg.eval_type != "simple"

    def seed(self, seed: int):
        self.gen.seed(seed)

    def sample(self, sample_size: int, graph_size: int = None, distribution=None, log_info=True):
        if distribution is None:
            distribution = self.distribution if self.distribution is not None else None
        if graph_size is None:
            graph_size = self.graph_size if self.graph_size is not None else None
        if log_info:
            logger.info(f"Sampling {sample_size} {distribution}-distributed problems with graph size {graph_size}")
        self.data = self.gen.generate(problem=self.problem,
                                      sample_size=sample_size,
                                      graph_size=graph_size,
                                      distribution=distribution,
                                      normalize=self.normalize,
                                      sampling_args=self.sampling_args,
                                      generator_args=self.generator_args)
        if self.time_limit is not None:
            self.data = [instance.update(time_limit=self.time_limit) for instance in self.data]
        # if not self.normalize:
        #     self.data = self._denormalize()
        if self.transform_func is not None:
            if self.transform_args is not None:
                self.data_transformed = self.transform_func(self.data, not self.normalize, **self.transform_args)
            else:
                self.data_transformed = self.transform_func(self.data)

        self.size = len(self.data)
        return self

    # @staticmethod
    def verify_costs_times(self, costs, times, sols, time_limit=None):
        verified_costs, verified_times, verified_sols = [], [], []
        verified_costs_full, verified_times_full, verified_sols_full = [], [], []
        prev_cost = float('inf')
        time_limit = self.adjusted_time_limit if self.adjusted_time_limit is not None else times[-1]  # if 'simple' eval
        # and time <= adjusted_time_limit
        sols = [None]*len(costs) if sols is None else sols
        for cost, time, sol in zip(costs, times, sols):
            if time <= time_limit:
                if cost < prev_cost and cost != float('inf'):
                    verified_costs.append(cost)
                    verified_times.append(time)
                    verified_sols.append(sol)
                    verified_costs_full.append(cost)
                    verified_times_full.append(time)
                    verified_sols_full.append(sol)
                    prev_cost = cost
                    # print('verified_costs', verified_costs)
                elif cost == prev_cost:
                    verified_costs_full.append(prev_cost)
                    verified_times_full.append(time)
                    verified_sols_full.append(sol)
                # print('prev_cost', prev_cost)

        return verified_costs, verified_times, verified_sols, \
            verified_costs_full, verified_times_full, verified_sols_full

    def _eval_metric(self,
                     model_name: str,
                     inst_id: str,
                     instance: CVRPInstance,
                     verified_costs: list,
                     verified_times: list,
                     run_times_orig: list,
                     eval_type: str = 'pi') -> Tuple[RPSolution, Union[float, None], Union[int, None]]:
        """(Re-)Evaluate provided solutions according to eval_type for the respective Routing Problem."""

        assert eval_type in ['pi', 'wrap'], f"Unknown Evaluation mode, must be one of 'pi', 'wrap' "
        assert self.bks is not None, f"For evaluation mode {eval_type} a Best Known Solution file is required."

        if verified_costs:
            if eval_type == "pi":
                score = self.metric.compute_pi(instance_id=inst_id,
                                               costs=verified_costs,
                                               runtimes=verified_times,
                                               normed_inst_timelimit=instance.time_limit)
            else:
                score = self.metric.compute_wrap(instance_id=inst_id, costs_=verified_costs, runtimes_=verified_times,
                                                 normed_inst_timelimit=instance.time_limit)
        elif not verified_costs and run_times_orig:
            logger.info(f"No solution found by {model_name} in time limit "
                        f"- aborting {eval_type.upper()} Evaluation for instance {inst_id}")
            logger.info(f"First run-time is {run_times_orig[0]} and adjusted time limit is {self.adjusted_time_limit}")
            score = 10 if eval_type == "pi" else 1
        else:
            logger.info(f"No feasible solution found by {model_name} "
                        f"- aborting {eval_type.upper()} evaluation for instance {inst_id}")
            score = 10 if eval_type == "pi" else 1

        return score

    def load_dataset(self, **kwargs):
        data = None
        if kwargs:
            print(f"Provided additional kwargs: {kwargs}")
        # store path given --> data import or download
        assert self.store_path is not None, f"Can only load dataset if an according Path is given"
        # if self.store_path is not None:
        filepath = os.path.normpath(os.path.expanduser(self.store_path))
        logger.info(f"Loading dataset from: {filepath}")
        # check if data directory for data exists and is not empty
        if os.path.exists(self.store_path):
            # get path and filename seperately
            dir_name = os.path.dirname(self.store_path)
            file_name = os.path.basename(self.store_path)
            # if directory is not empty & has ONE file --> load file as dataset
            if os.path.isfile(dir_name + "/" + file_name):
                assert os.path.splitext(self.store_path)[1] in ['.pkl', '.dat', '.pt', '.vrp', '.sd']
                if os.path.splitext(self.store_path)[1] == '.vrp' or \
                        os.path.splitext(self.store_path)[1] == '.sd':
                    data = self.read_vrp_instance(filepath)
                else:
                    try:
                        data = torch.load(filepath)
                    except RuntimeError:
                        # fall back to pickle loading
                        assert os.path.splitext(filepath)[1] == '.pkl', "Can only load pickled datasets."
                        with open(dir_name + "/" + file_name, 'rb') as f:
                            data = pickle.load(f)
            # if directory is not empty & has MULTIPLE files --> load files in directory as one dataset
            elif len(os.listdir(self.store_path)) > 1:
                logger.info("Loading .vrp instances...")
                data = [self.read_vrp_instance(self.store_path+"/"+file) for file in os.listdir(self.store_path)
                        if (file[:3] != 'BKS' and file[-3:] == 'vrp')]
            # download data
            else:
                print(f'{self.problem.upper()} dataset for {file_name} needs to be downloaded in the directory - '
                      f'this may take a minute')
                self._download()
                data = self.load_dataset()
            key_list = [key for key in DATA_KEYWORDS.keys() if key in self.store_path]
            self.data_key = "unknown" if len(key_list) == 0 else key_list[0]

        else:
            print(f"Directory '{self.store_path}' does not exist")
            data = None

        return data, self.data_key

    # from L2O-Meta
    @staticmethod
    def save_dataset(dataset: Union[List, np.ndarray],
                     filepath: str,
                     **kwargs):
        """Saves data set to file path"""
        filepath = format_ds_save_path(filepath, **kwargs)
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        logger.info(f"Saving dataset to:  {filepath}")
        try:
            torch.save(dataset, filepath)
        except RuntimeError:
            # fall back to pickle save
            assert os.path.splitext(filepath)[1] == '.pkl', "Can only save as pickle. Please add extension '.pkl'!"
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        return str(filepath)

    def preprocess(self):
        """Not preprocessing in the model-based sense.
        Instead, preprocessing into python-readable format"""
        pass

    def load_BKS_BaseSol(self, item_to_load: str = "BKS"):
        """get BKS from respective test file directory"""
        # check if dataset has a BKS/BaseSol store file
        if self.store_path is not None:
            _path = None
            if os.path.basename(self.store_path) in TEST_SETS_BKS \
                    or os.path.basename(self.store_path)[:2] in TEST_SETS_BKS \
                    or self.store_path.split("/")[-2] in TEST_SETS_BKS:
                logger.info(f'{item_to_load} file should exists for {self.store_path}')
                if os.path.basename(self.store_path)[:4] == "test":
                    # load BKS for original test data of size 10000
                    _path = os.path.join(os.path.dirname(self.store_path), item_to_load + ".pkl")
                elif os.path.basename(self.store_path)[:3] == "val":
                    # load BKS for val data
                    _path = os.path.join(os.path.dirname(self.store_path), item_to_load + "_val.pkl")
                elif Path(os.path.join(self.store_path, item_to_load + "_" + self.store_path.split("/")[-3] + ".pkl")).exists():
                    load_name = item_to_load + "_" + self.store_path.split("/")[-3] + ".pkl"
                    # print('os.path.join(self.store_path, load_name)', os.path.join(self.store_path, load_name))
                    _path = os.path.join(self.store_path, load_name)
                elif Path(os.path.join(os.path.dirname(self.store_path),
                                       item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl"
                    _path = os.path.join(os.path.dirname(self.store_path), load_name)
                elif Path(os.path.join(self.store_path,
                                       item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl"
                    _path = os.path.join(self.store_path, load_name)
                elif Path(os.path.join(self.store_path,
                                       item_to_load + "_" + self.store_path.split("/")[-2] + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + self.store_path.split("/")[-2] + ".pkl"
                    _path = os.path.join(self.store_path, load_name)
                elif Path(os.path.join(self.store_path,
                                       item_to_load + "_" + os.path.basename(self.store_path) + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + os.path.basename(self.store_path) + ".pkl"
                    _path = os.path.join(self.store_path, load_name)
                else:
                    logger.info(
                        f"Couldn't load {item_to_load} file - make sure it exists in directory {self.store_path} ")
                    return None
                if item_to_load == "BKS":
                    self.bks_path = _path
                    logger.info(f'Loading Best Known Solutions from {self.bks_path}')
                    return torch.load(self.bks_path)
                else:
                    self.base_sol_path = _path
                    logger.info(f'Loading Base Solver Results (for WRAP eval.) from {self.base_sol_path}')
                    return torch.load(self.base_sol_path)
            else:
                logger.info(f"No {item_to_load} stored for this Test Data - Setting {item_to_load} to None")
        else:
            warnings.warn('Attempted to load Best Known Solutions while no test data store path is given. Setting BKS '
                          'to None for training.')
            return None

    @staticmethod
    def _create_nodes(size: int,
                      graph_size: int,
                      features: List,
                      n_depots: int = 1):
        """Create node id and type vectors and concatenate with other features."""
        return np.dstack((
            np.broadcast_to(np.concatenate((  # add id and node type (depot / customer)
                np.array([1] * n_depots +
                         [0] * graph_size)[:, None],  # depot/customer type 1-hot
                np.array([0] * n_depots +
                         [1] * graph_size)[:, None],  # depot/customer type 1-hot
            ), axis=-1), (size, graph_size + n_depots, 2)),
            *features,
        ))

    # @staticmethod
    def save_trajectory(self, instance_id, costs, times, model, save_trajectory_for=None, instance=None):
        if save_trajectory_for is None and self.save_traj_flag:
            save_trajectory_for = instance_id
            self.save_traj_flag = False
        if instance_id == str(save_trajectory_for):
            logger.info(f'Saving solution trajectory for instance {instance_id}')
            if instance is not None:
                save_name = "_instance_" + instance_id + "_c_dist_" + str(instance.coords_dist) + "_d_dist_" \
                            + str(instance.demands_dist) \
                            + "_depot_type_" + str(instance.depot_type)
                self.plot_trajectory(instance_id, model, times, costs, save_name, instance.time_limit)
            else:
                self.plot_trajectory(instance_id, model, times, costs)
        elif isinstance(save_trajectory_for, List) and int(instance_id) in save_trajectory_for:
            logger.info(f'Saving solution trajectory for instance {instance_id}')
            if instance is not None:
                save_name = "_instance_" + instance_id + "_c_dist_" + str(instance.coords_dist) + "_d_dist_" \
                            + str(instance.demands_dist) \
                            + "_depot_type_" + str(instance.depot_type)
                self.plot_trajectory(instance_id, model, times, costs, save_name, instance.time_limit)
            else:
                self.plot_trajectory(instance_id, model, times, costs)
        else:
            pass

    @staticmethod
    def plot_trajectory(id_, model_name, times, costs, save_name=None, time_limit=None):
        torch.save(times, 'trajectory_times_' + str(id_) + '.pt')
        torch.save(costs, 'trajectory_costs_' + str(id_) + '.pt')
        # save plot
        plt.plot(times, costs, label=model_name)
        plt.xlabel('cumulative runtime (seconds) ')
        plt.ylabel('objective value (total cost)')
        plt.title('Trajectory for Time Limit: ' + str(time_limit))
        plt.legend()
        if save_name is not None:
            plt.savefig(save_name + '.pdf')  # will be saved in output dir
        else:
            plt.savefig('trajectory_plot_' + str(id_) + '.pdf')  # will be saved in output dir
        plt.close()  # close the figure window

    @abstractmethod
    def read_vrp_instance(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def _denormalize(self):
        raise NotImplementedError

    @abstractmethod
    def _download(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
