from data.base_dataset import BaseDataset
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any
import warnings
from abc import ABC
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
# , make_instance_vrptw
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
                 num_samples: int = 1000,
                 seed: int = None,
                 normalize: bool = True,
                 offset: int = 0,
                 distribution: Optional[str] = None,
                 graph_size: int = 20,
                 device: str = None,
                 verbose: bool = False):
        super(TSPDataset, self).__init__(problem='tsp',
                                         store_path=store_path,
                                         num_samples=num_samples,
                                         seed=seed,
                                         verbose=verbose)

        self.is_train = is_train
        self.num_samples = num_samples
        self.seed = seed
        self.normalize = normalize
        self.offset = offset
        self.distribution = distribution  # defaults to uniform random distribution for data generation
        self.graph_size = graph_size
        self.device = device
        self.data_key = None
        self.bks = None

        # if is_train is False:
        if store_path is not None:
            # load OR DOWNLOAD (test) data
            self.data = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            # For test instances in store path - load the BKS registry
            self.bks = self.load_BKS(self.problem, self.distribution, self.graph_size)
            # Transform loaded data to TSPInstance format
            self.data = self._make_TSPInstance()
        else:
            # sample data
            self.sample(sample_size=self.num_samples,
                        graph_size=self.graph_size,
                        distribution=self.distribution)
        self.size = len(self.data)

    def _download(self):
        save_path = self.get_tsplib_instances()
        self.save_tsplib_instances(save_path)

    def _make_TSPInstance(self):
        """Reformat (loaded) test instances as TSPInstances"""
        if isinstance(self.data[0], List) or self.data_key == 'uniform':    # for kool test data
            coords = np.stack([np.array(self.data[0][i]) for i in range(len(self.data))])

            # update self.graph_size - if data not sampled
            self.graph_size = coords.shape[1]

            node_features = self._create_nodes(len(self.data), self.graph_size - 1, n_depots=1, features=[coords])

            bks = self.load_BKS(self.problem, self.distribution, self.graph_size)

            return [
                TSPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=self.graph_size,
                    time_limit=10,  # arbitrarily give 10 seconds for now -> # TODO: schematic approach to time limit
                    BKS=bks[str(i)][0] if bks is not None else None,
                    instance_id=i if self.bks is not None else None

                )
                for i in range(len(self.data))
            ]
        else:
            warnings.warn("Seems not to be the correct test data format for TSP - expected a List of List of "
                          "coordinate instances")
            raise RuntimeError("Unexpected format for TSP Test Instances - make sure that a TSP test set is loaded")

        # return TSPInstance()

    def _is_feasible(self, solution: RPSolution):
        # Check that sol is valid for TSP
        sol = solution.solution.copy()
        sol.sort()
        return np.arange(len(sol)).tolist() == sol

    def _eval_metric(self,
                     solution: RPSolution,
                     PI_Evaluation: bool = False,
                     passMark: int = None) -> Tuple[RPSolution, Union[float, None], Union[int, None]]:
        """(Re-)Evaluate provided solutions for the TSP."""

        new_best = None

        # check feasibility
        if self._is_feasible(solution):
            data = solution.instance
            # depot = data.depot_idx[0] # for TSP no depot
            coords = data.coords
            tour = solution.solution  # only one single tour for TSP
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

            if cost < data.BKS:
                # update global BKS
                self.bks[str(data.instance_id)] = cost
                logger.info(f"New BKS found for instance {data.instance_id} of the {self.distribution}-"
                            f"distributed {self.problem} Test Set")
                logger.info(f"New BKS with cost {cost} is {data.BKS - cost} better than old BKS with cost {data.BKS}")
                new_best = str(data.instance_id)
                # store here new BKS? # TODO is there a more efficient way to update the global registry?

            if PI_Evaluation:
                # calculate Primal Integral as in DIMACS challenge (see BaseDataset for func. compute_pi)
                # --> possible only if some Best Known Solution (BKS) is available
                pi_score, prev_cost, prev_time = self.compute_pi(cost,
                                                                 solution.run_time,
                                                                 data.BKS,
                                                                 data.time_limit,
                                                                 passMark,
                                                                 solution.last_cost,
                                                                 solution.last_runtime)
                if self.verbose is True:
                    logger.info(
                        f"Recorded PI Score={pi_score} for {self.distribution}-distributed {self.problem} instance.")
                return solution.update(cost=cost,
                                       pi_score=pi_score,
                                       last_cost=prev_cost,
                                       last_runtime=prev_time), pi_score, new_best

        else:
            warnings.warn(f"TSP solution infeasible. setting cost to 'inf'")
            cost = float("inf")

        return solution.update(cost=cost), None, new_best

    @staticmethod
    def get_tsplib_instances():
        """
        download tsplib from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
        return: directory for the saved instances
        """

        print("Start downloading tsp instances from tsplib...")

        save_path = "./data/test_data/tsp/tsplib/raw_data"
        url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
        filename = url.split("/")[-1]
        r = requests.get(url, stream=True)
        with tarfile.open(fileobj=BytesIO(r.content), mode="r:gz") as tar_file:
            tar_file.extractall(save_path)
            tar_file.close()

        print(f'Downloading tsplib to {save_path}')
        for file in os.listdir(save_path):
            if file.endswith('.gz'):
                fullpath = os.path.join(save_path, file)
                with gzip.open(fullpath, 'rb') as f_in:
                    filename = Path(os.path.basename(file))
                    filename_wo_ext = filename.with_suffix('')
                    full_dest_fpath = save_path / filename_wo_ext
                    with open(full_dest_fpath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        # removing .gz file from the directory
        for item in os.listdir(save_path):
            if item.endswith(".gz"):
                os.remove(os.path.join(save_path, item))

        return save_path

    @staticmethod
    def read_tsplib(file_path: str, DATA_SPATH: str):
        """
        read tsplib file with tsplib95 library
        """
        problem = tsplib95.load(file_path)
        instances = problem.node_coords
        data = []
        for key, value in instances.items():
            data.append(value)
        tsp_data = torch.FloatTensor(data)

        return tsp_data

    def save_tsplib_instances(self, raw_data_path: str):
        """
        process .pkl file for tsp instances
        optimal tour files are excluded
        """
        directory = "pkl/"
        save_path = './data/test_data/tsp/tsplib/' + directory
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
            print(f"Directory {save_path} created")
        print(f'Processing pkl file...')
        for member in os.listdir(raw_data_path):
            file_path = os.path.join(raw_data_path, member)
            fname = os.path.basename(file_path).split('.')[0]
            DATA_SPATH = save_path + fname + '.pkl'
            tsp_data = self.read_tsplib(file_path, DATA_SPATH)
            os.makedirs(os.path.dirname(DATA_SPATH), exist_ok=True)
            with open(DATA_SPATH, 'wb') as f:
                pickle.dump(tsp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'.pkl files are available at {save_path}')


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
