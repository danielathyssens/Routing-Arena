from data.base_dataset import BaseDataset
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any, Callable
from omegaconf import DictConfig, ListConfig
import warnings
import os, gzip
import requests
import torch
import numpy as np
import glob
import subprocess
import shutil
import pickle
from torch.utils.data import Dataset
from formats import TSPInstance, RPSolution
from models.runner_utils import NORMED_BENCHMARKS, get_budget_per_size, _adjust_time_limit
import logging
from copy import deepcopy
from io import BytesIO
import tarfile
from pathlib import Path
import tsplib95

import logging

logger = logging.getLogger(__name__)

EPS = np.finfo(np.float32).eps


class TSPDataset(BaseDataset):
    """Creates TSP data samples to use for training or evaluating benchmark models"""

    def __init__(self,
                 is_train: bool = False,
                 store_path: str = None,
                 dataset_size: int = None,
                 dataset_range: list = None,
                 num_samples: int = 64,
                 seed: int = None,
                 normalize: bool = True,
                 offset: int = 0,
                 distribution: Optional[str] = None,
                 generator_args: dict = None,
                 sampling_args: Optional[dict] = None,
                 graph_size: int = 20,
                 grid_size: int = 1,
                 float_prec: np.dtype = np.float32,
                 transform_func: Callable = None,
                 transform_args: DictConfig = None,
                 device: str = None,
                 verbose: bool = False,
                 TimeLimit: Union[int, float] = None,
                 machine_info: tuple = None,
                 load_bks: bool = True,
                 load_base_sol: bool = True,
                 re_evaluate: bool = False,
                 ):
        super(TSPDataset, self).__init__(problem='tsp',
                                         store_path=store_path,
                                         num_samples=num_samples,
                                         graph_size=graph_size,
                                         normalize=normalize,
                                         scale_factor=None,
                                         is_denormed=False,
                                         float_prec=float_prec,
                                         transform_func=transform_func,
                                         transform_args=transform_args,
                                         distribution=distribution,
                                         generator_args=generator_args,
                                         sampling_args=sampling_args,
                                         seed=seed,
                                         verbose=verbose,
                                         TimeLimit=TimeLimit,
                                         load_bks=load_bks,
                                         load_base_sol=load_base_sol)

        self.is_train = is_train
        self.num_samples = num_samples
        self.dataset_size = dataset_size
        self.dataset_range = dataset_range
        self.seed = seed
        self.normalize = normalize
        self.offset = offset
        self.distribution = distribution  # defaults to uniform random distribution for data generation
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.graph_size = graph_size
        self.time_limit = TimeLimit
        self.machine_info = machine_info
        if self.machine_info is not None:
            print('self.machine_info in Dataset', machine_info)
        self.re_evaluate = re_evaluate
        self.metric = None
        self.transform_func = transform_func
        self.transform_args = transform_args
        self.data_key = None
        self.scale_factor = None
        self.is_denormed = False
        self.grid_size = grid_size if self.distribution != "uchoa" else 1000
        # grid_size=grid_size, # if self.distribution != "uchoa" else 1000,

        # if is_train is False:
        if store_path is not None:
            # load or download (test) data
            self.data, self.data_key = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            if self.dataset_size is not None and self.dataset_size < len(self.data):
                self.data = self.data[:self.dataset_size]
            elif self.dataset_range is not None:
                print('self.dataset_range', self.dataset_range)
                self.data = self.data[self.dataset_range[0]:self.dataset_range[1]]
            logger.info(f"{len(self.data)} TSP Test/Validation Instances for {self.problem} with {self.graph_size} "
                        f"{self.distribution}-distributed nodes loaded.")

            # Transform loaded data to TSPInstance format
            if not isinstance(self.data[0], TSPInstance):
                self.data = self._make_TSPInstance()
            if not self.normalize and not self.is_denormed:
                self.data = self._denormalize()
            if self.bks is not None:
                # and np.any(np.array([instance.BKS for instance in self.data])):
                self.data = self._instance_bks_updates()
            if self.transform_func is not None:  # transform_func needs to return list
                self.data_transformed = self.transform_func(self.data)
            self.size = len(self.data)
        elif not is_train:
            logger.info(f"No file path for evaluation specified. Default to sampling...")
            if self.distribution is not None:
                # and self.sampling_args['sample_size'] is not None:
                logger.info(f"Sampling data according to env config file: {self.sampling_args}")
                self.sample(sample_size=self.sampling_args['sample_size'],
                            graph_size=self.graph_size,
                            distribution=self.distribution,
                            log_info=True)
            else:
                logger.info(f"Data configuration not specified in env config, "
                            f"defaulting to 100 uniformly distributed TSP20 instances")
                self.sample(sample_size=100,
                            graph_size=20,
                            distribution="uniform",
                            log_info=True)
            self.size = len(self.data)
        else:  # no data to load - but initiated CVRPDataset for sampling in training loop
            logger.info(f"No data loaded - initiated TSPDataset with env config for sampling on the fly in training...")
            self.size = None
            self.data = None

    def _download(self):
        save_path = self.get_tsplib_instances()

    def _make_TSPInstance(self):
        """Reformat (loaded) test instances as TSPInstances"""
        if isinstance(self.data[0], List) or self.data_key == 'uniform':
            logger.info("Transforming instances to TSPInstances")
            warnings.warn("This works only for Nazari et al type of data")
            # if self.normalize:
            # [0]
            # coords = np.stack([np.array(self.data[i]) for i in range(1, len(self.data))]) --> works for TSPLib only
            coords = np.stack([np.array(self.data[i]) for i in range(len(self.data))])

            # update self.graph_size - if data not sampled
            self.graph_size = coords.shape[1]

            node_features = self._create_nodes(len(self.data), self.graph_size - 1, n_depots=1, features=[coords])

            return [
                TSPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=self.graph_size,
                    time_limit=self.time_limit,
                    instance_id=i

                )
                for i in range(len(self.data))
            ]
        else:
            warnings.warn("Seems not to be the correct test data format for TSP - expected a List of List of "
                          "coordinate instances")
            raise RuntimeError("Unexpected format for TSP Test Instances - make sure that a TSP test set is loaded")

        # return TSPInstance()

    def _instance_bks_updates(self):
        # always update benchmark data instances with newest BKS registry if registry given for loaded data
        return [
            TSPInstance(
                coords=instance.coords,
                node_features=instance.node_features,
                graph_size=instance.graph_size,
                time_limit=self.time_limit if instance.time_limit is None else instance.time_limit,
                BKS=self.bks[str(instance.instance_id if instance.instance_id is not None else i)][0]
                if self.bks is not None else None,
                instance_id=instance.instance_id if instance.instance_id is not None else i,
                coords_dist=instance.coords_dist,
                depot_type=instance.depot_type,
                original_locations=instance.original_locations if instance.original_locations is not None else None,
                type=instance.type if instance.type is not None else None,
            )
            for i, instance in enumerate(self.data)
        ]

    # def _get_costs(self, sol: RPSolution) -> Tuple[float, bool, Union[list, None]]:
    #    # perform problem-specific feasibility check while getting routing costs
    #     cost, solution_upd = self.feasibility_check(sol.instance, sol)
    #     is_feasible = True if cost != float("inf") else False
    #     return cost, is_feasible, solution_upd



    @staticmethod
    def feasibility_check(instance: TSPInstance, rp_solution: Union[RPSolution, List[list]], is_running: bool = False):
        # Check that sol is valid for TSP
        sol = rp_solution.solution.copy() if isinstance(rp_solution, RPSolution) else rp_solution.copy()
        if sol is not None:
            sol.sort()
            is_feasible = np.arange(len(sol)).tolist() == sol
            # print('is_feasible', is_feasible)
            # check feasibility
            if is_feasible:
                data = instance
                coords = data.coords
                tour = rp_solution.solution.copy() if isinstance(rp_solution, RPSolution) else rp_solution.copy()
                # calculate cost
                cost = 0.0
                transit = 0
                source = tour[0]
                for target in tour[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    source = target
                # add transit from last to first
                transit += np.linalg.norm(coords[tour[0]] - coords[source], ord=2)
                cost += transit
                return cost, 1, tour
            else:
                logger.info(f"TSP solution not feasible - not all nodes covered in tour.")
                return float("inf"), None, None
        else:
            return float("inf"), None, None

#    @staticmethod
    def get_tsplib_instances(self):
        """
        download tsplib from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
        return: directory for the saved instances
        """

        print("Start downloading tsp instances from tsplib...")

        prefix = "./../../../../../" # simple
        save_path = "data/test_data/tsp/tsplib/raw_data/"
        url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
        filename = url.split("/")[-1]
        r = requests.get(url, stream=True)
        with tarfile.open(fileobj=BytesIO(r.content), mode="r:gz") as tar_file:
            tar_file.extractall(prefix + save_path)
            tar_file.close()

        print(f'Downloading tsplib to {save_path}')
        for file in os.listdir(prefix + save_path):
            if file.endswith('.gz'):
                fullpath = os.path.join(prefix + save_path, file)
                with gzip.open(fullpath, 'rb') as f_in:
                    filename = Path(os.path.basename(file))
                    filename_wo_ext = filename.with_suffix('')
                    full_dest_fpath = (prefix + save_path) / filename_wo_ext
                    with open(full_dest_fpath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        # removing .gz file from the directory
        for item in os.listdir(prefix + save_path):
            if item.endswith(".gz"):
                os.remove(os.path.join(prefix + save_path, item))

        # remove files that do not have EUC_2D and NODE_COORD_SECTION
        for file in os.listdir(prefix + save_path):
            if (file[-3:] == 'tsp'):
                problem = tsplib95.load(os.path.join(prefix, save_path, file))
                if problem.edge_weight_type != "EUC_2D":
                    os.remove(os.path.join(prefix, save_path, file))

        return prefix + save_path

    def read_tsp_instance(self, filepath: str):
        problem = tsplib95.load(filepath)
        instances = problem.node_coords
        coords = np.stack([value for key,value in instances.items()])
        graph_size = len(instances.items())
        
        return TSPInstance(
            coords=coords,
            node_features=None,
            graph_size=graph_size,
            instance_id=None,
        )

# ============= #
# ### TEST #### #
# ============= #
def _test(
        size: int = 10,
        n: int = 20,
        seed: int = 1,
):
    # problems = ['tsp', 'cvrp']
    # coord_samp = ['uniform', 'gm']
    # weight_samp = ['random_int', 'uniform', 'gamma']
    coord_samp = ['nazari']
    # for test get verbose information
    verb = True
    k = 4
    cap = 9
    max_cap_factor = 1.1
    for csmp in coord_samp:
        # for wsmp in weight_samp:
        ds = TSPDataset(num_samples=size,
                        distribution=csmp,
                        graph_size=n,
                        seed=seed,
                        normalize=True,
                        verbose=verb)
        # ds.sample(sample_size=size, graph_size=n) -> sampling in init class
        print('ds.data[0]', ds.data[0])
        print('ds.size', ds.size)
