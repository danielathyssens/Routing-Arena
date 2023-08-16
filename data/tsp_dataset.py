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

        # if is_train is False:
        if store_path is not None:
            # load or download (test) data
            self.data, self.data_key = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            if self.dataset_size is not None and self.dataset_size < len(self.data):
                self.data = self.data[:self.dataset_size]
            logger.info(f"{len(self.data)} TSP Test/Validation Instances for {self.problem} with {self.graph_size} "
                        f"{self.distribution}-distributed nodes loaded.")
            # Transform loaded data to TSPInstance format
            if not isinstance(self.data[0], TSPInstance):
                self.data = self._make_TSPInstance()
            if not self.normalize and not self.is_denormed:
                self.data = self._denormalize()
            if self.bks is not None and np.any(np.array([instance.bks for instance in self.data])):
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
        self.save_tsplib_instances(save_path)

    def _make_TSPInstance(self):
        """Reformat (loaded) test instances as TSPInstances"""
        if isinstance(self.data[0], List) or self.data_key == 'uniform':
            logger.info("Transforming instances to CVRPInstances")
            warnings.warn("This works only for Nazari et al type of data")
            # if self.normalize:
            # [0]
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
        print('self.data[0].instance_id, self.data[0].time_limit', self.data[0].instance_id, self.data[0].time_limit)
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

    def _get_costs(self, sol: RPSolution) -> Tuple[float, bool, Union[list, None]]:
        # perform problem-specific feasibility check while getting routing costs
        cost, solution_upd = self.feasibility_check(sol.instance, sol)
        is_feasible = True if cost != float("inf") else False
        return cost, is_feasible, solution_upd

    def eval_solution(self,
                      model_name: str,
                      solution: RPSolution,
                      eval_mode: Union[str, list] = 'simple',
                      save_trajectory: bool = False,
                      save_trajectory_for: Union[int, List] = None,
                      place_holder_final_sol: bool = False):

        # init scores
        pi_score, wrap_score = None, None
        # get instance
        instance = solution.instance
        # get cost + feasibility check
        cost, is_feasible, solution_updated = self._get_costs(solution)
        # directly return infeasible solution
        if not is_feasible:
            self.return_infeasible_sol(eval_mode, instance, solution, cost)
        # print('solution_updated', solution_updated)
        solution = solution.update(solution=solution_updated)
        print('self.scale_factor', self.scale_factor)
        if self.scale_factor is not None:
            cost = cost * self.scale_factor
        if self.is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
            print('IS_DENORMED AND STOREPATH IN NORMED_BENCHMARKS --> RENORMALIZE COSTS FOR EVAL')
            # (re-)normalize costs for evaluation for dataset that is originally normalized
            cost = cost / self.grid_size

        # default to simple evaluation if no BKS loaded or incorrect ID order
        eval_mode = self.check_eval_mode(eval_mode, solution)

        # get and verify running values
        if self.verbose:
            if solution.running_sols is not None:
                print('Len(running_sols) before VERIFY', len(solution.running_sols))
            if solution.running_costs is not None:
                print('Len(running_costs) before VERIFY', len(solution.running_costs))
                print('Len(running_times) before VERIFY', len(solution.running_times))

        verified_values, running_times = self.get_running_values(
            instance,
            solution.running_sols,
            solution.running_costs,
            solution.running_times,
            solution.run_time,
            cost,
            self.scale_factor,
            self.grid_size,
            self.is_denormed,
            place_holder_final_sol
        )

        v_costs, v_times, v_sols, v_costs_full, v_times_full, v_sols_full = verified_values

        if isinstance(eval_mode, ListConfig) or isinstance(eval_mode, list):
            for mode in eval_mode:
                if mode == "pi":
                    pi_score = self.eval_costs("pi", instance, v_costs, v_times, running_times,
                                               model_name)
                elif mode == "wrap":
                    if self.metric.base_sol_results is None:
                        warnings.warn(f"Defaulting to simple evaluation - "
                                      f"no base solver results loaded for WRAP Evaluation.")
                    else:
                        wrap_score = self.eval_costs("wrap", instance, v_costs, v_times,
                                                     running_times, model_name)
                else:
                    assert mode == "simple", f"Unknown eval type in list eval_mode. Must be in ['simple', 'pi', 'wrap']"
                    # simple eval already done in self._get_costs()
        elif eval_mode == "pi":
            pi_score = self.eval_costs("pi", instance, v_costs, v_times, running_times, model_name)
        elif eval_mode == "wrap":
            if self.metric.base_sol_results is None:
                warnings.warn(f"Defaulting to simple evaluation - "
                              f"no base solver results loaded for WRAP Evaluation.")
            else:
                wrap_score = self.eval_costs("wrap", instance, v_costs, v_times, running_times,
                                             model_name)
        else:
            assert eval_mode == "simple", f"Unknown eval type. Must be in ['simple', 'pi', 'wrap']"
            # simple eval already done in self._get_costs()

        # update global BKS
        new_best = self.update_BKS(instance, cost) if instance.BKS is not None and cost < instance.BKS else None

        # save sol-trajectories
        if save_trajectory and v_costs:
            self.save_trajectory(str(instance.instance_id), v_costs_full, v_times_full, model_name,
                                 save_trajectory_for, instance)

        return solution.update(cost=cost,
                               running_costs=v_costs if v_costs and v_costs is not None else None,
                               running_times=v_times if v_times and v_times is not None else None,
                               running_sols=v_sols if v_sols and any(v_sols) else None,
                               run_time=solution.run_time,  # self.adjusted_time_limit),
                               pi_score=pi_score,
                               wrap_score=wrap_score), None, new_best

    @staticmethod
    def feasibility_check(instance: TSPInstance, solution: Union[RPSolution, List[list]], is_running: bool = False):
        # Check that sol is valid for TSP
        sol = solution.solution.copy() if isinstance(solution, RPSolution) else solution.copy()
        if sol is not None:
            sol.sort()
            is_feasible = np.arange(len(sol)).tolist() == sol
            # check feasibility
            if is_feasible:
                data = instance
                # depot = data.depot_idx[0] # for TSP no depot
                coords = data.coords
                tour = solution.solution if isinstance(solution, RPSolution) else solution
                print('tour', tour)
                # calculate cost
                cost = 0.0
                transit = 0
                source = tour[0]
                for target in tour[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    print('transit', transit)
                    source = target
                # add transit from last to first
                transit += np.linalg.norm(coords[tour[0]] - coords[source], ord=2)
                cost += transit
                return cost, tour
            else:
                return float("inf"), None
        else:
            return float("inf"), None

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
