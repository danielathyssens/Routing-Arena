from data.base_dataset import BaseDataset
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any, Callable
from omegaconf import DictConfig, ListConfig
import os, gzip
import requests
import torch
import glob
import subprocess
import shutil
import pickle
import warnings
import itertools
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from formats import CVRPTWInstance, RPSolution, RPInstance
from models.runner_utils import NORMED_BENCHMARKS, get_budget_per_size, _adjust_time_limit
import logging
from urllib.request import urlopen, Request, urlparse
from zipfile import ZipFile
from io import BytesIO
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

EPS = np.finfo(np.float32).eps

Ccvrptw_DEFAULTS = {  # num vehicles and integer capacity per problem size
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}


class CVRPTWDataset(BaseDataset):
    """Creates CVRPTW data samples to use for training or evaluating benchmark models"""

    def __init__(self,
                 is_train: bool = False,
                 store_path: str = None,
                 dataset_size: int = None,
                 dataset_range: list = None,
                 seed: int = None,
                 num_samples: int = 100000,
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
        super(CVRPTWDataset, self).__init__(problem='cvrptw',
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
        self.dataset_range = dataset_range
        self.seed = seed
        self.normalize = normalize
        self.offset = offset
        self.distribution = distribution
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.graph_size = graph_size
        self.time_limit = TimeLimit
        self.machine_info = machine_info
        if self.machine_info is not None:
            print('self.machine_info in Dataset', machine_info)
        # self.num_vehicles = num_vehicles  --> in generator args
        # self.capacity = capacity   --> in generator args
        # self.max_cap_factor = max_cap_factor   --> in generator args
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
            print('self.dataset_size in else', self.dataset_size)
            if self.dataset_size is not None:
                if not isinstance(self.data, dict) and self.dataset_size < len(self.data):
                    self.data = self.data[:self.dataset_size]
                else:
                    self.data["loc"] = self.data["loc"][:self.dataset_size]
                    self.data["demand"] = self.data["demand"][:self.dataset_size]
                    self.data["start"] = self.data["start"][:self.dataset_size]
                    self.data["end"] = self.data["end"][:self.dataset_size]
            elif self.dataset_range is not None:
                print('self.dataset_range', self.dataset_range)
                if not isinstance(self.data, dict) and self.dataset_size < len(self.data):
                    self.data = self.data[self.dataset_range[0]:self.dataset_range[1]]
                else:
                    print('self.dataset_range[0]', self.dataset_range[0])
                    print('self.dataset_range[0]', self.dataset_range[1])
                    print('self.data["loc"][self.dataset_range[0]:self.dataset_range[1], :, :]',
                          self.data["loc"][self.dataset_range[0]:self.dataset_range[1], :, :].shape)
                    self.data["loc"] = self.data["loc"][self.dataset_range[0]:self.dataset_range[1]]
                    self.data["demand"] = self.data["demand"][self.dataset_range[0]:self.dataset_range[1]]
                    self.data["start"] = self.data["start"][self.dataset_range[0]:self.dataset_range[1]]
                    self.data["end"] = self.data["end"][self.dataset_range[0]:self.dataset_range[1]]
            self.size = len(self.data) if isinstance(self.data, List) else len(self.data["loc"])
            logger.info(f"{self.size} CVRPTW Test/Validation Instances for {self.problem} with {self.graph_size} "
                        f"{self.distribution}-distributed nodes loaded.")
            # Transform loaded data to TSPInstance format
            if isinstance(self.data, dict):
                self.data = self._make_CVRPTWInstance()
            elif not isinstance(self.data[0], CVRPTWInstance):
                self.data = self._make_CVRPTWInstance()
            if not self.normalize and not self.is_denormed:
                self.data = self._denormalize()
            if self.bks is not None:   #  and np.any(np.array([instance.bks for instance in self.data])):
                self.data = self._instance_bks_updates()
            if self.transform_func is not None:  # transform_func needs to return list
                self.data_transformed = self.transform_func(self.data)
            self.size = len(self.data) if not isinstance(self.data, dict) else len(self.data["loc"])
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

        else:  # no data to load - but initiated CVRPDataset for sampling in training loop
            logger.info(f"No data loaded - initiated TSPDataset with env config for sampling on the fly in training...")
            self.size = None
            self.data = None

    def _download(self):
        save_paths, folder_names = self.get_cvrptw_instances()
        self.data = self.prepare_solomon_instances(save_paths, folder_names)
        return self.data

    def _make_CVRPTWInstance(self):
        """Reformat (loaded) test instances as VRPTWInstances"""
        if isinstance(self.data, dict):  # NeuroLKH type test data

            coords = np.stack([loc for loc in self.data['loc']])
            demands = np.stack([np.concatenate([np.zeros(1), de]).reshape(
                de.shape[0] + 1, 1) for de in self.data['demand']])
            twindows = np.stack([np.stack([np.concatenate([np.zeros(1),
                                                           self.data["start"][i]]),  # /10
                                           np.concatenate([np.ones(1), self.data["end"][i]])],  # /10
                                          axis=1) for i in range(len(self.data["start"]))])

            node_features = self._create_nodes(len(coords), coords.shape[1] - 1, n_depots=1,
                                               features=[coords, demands, twindows])
            # print('node_features.shape', node_features.shape)
            # print("demands[0]/self.data['capacity']", demands[0]/self.data['capacity'])
            return [
                CVRPTWInstance(
                    graph_size=len(self.data['loc'][i]),
                    coords=coords[i],
                    demands=demands[i] / self.data['capacity'],  # /self.data['capacity']
                    node_features=node_features[i],
                    depot_tw=np.array([0, 1]),
                    tw=twindows[i],
                    org_service_horizon=10,
                    service_horizon=10,
                    original_capacity=self.data['capacity'],
                    vehicle_capacity=1.0,
                    service_time=self.data['service_time'],
                    coords_dist="uniform",
                    depot_type="uniform",
                    demands_dist="rnd_normal",
                    time_limit=self.time_limit,
                    BKS=self.bks[str(i)][0] if self.bks is not None else None,
                    instance_id=i,
                )
                for i in range(len(self.data["loc"]))]

        elif isinstance(self.data[0], List) or self.data_key == 'uniform':
            logger.info("Transforming instances to CVRPTWInstances")
            coords, demands, twindows = [], [], []
            for i in range(len(self.data)):
                if self.normalize:
                    coords_i = np.vstack((self.data[i][0], self.data[i][1])) / self.grid_size
                    demands_i = np.array(self.data[i][2])
                    twindows_i = np.array(self.data[i][3]) / self.grid_size
                    demands.append(np.insert(demands_i, 0, 0) / self.data[i][3])
                    twindows.append(twindows_i)
                    coords.append(coords_i)
                else:
                    coords_i = np.vstack((self.data[i][0], self.data[i][1]))
                    demands_i = np.array(self.data[i][2])
                    twindows_i = np.array(self.data[i][3])
                    demands.append(np.insert(demands_i, 0, 0))
                    coords.append(coords_i)
                    twindows.append(twindows_i)
                    self.is_denormed = True
            coords = np.stack(coords)
            demands = np.stack(demands)
            twindows = np.stack(twindows)
            self.graph_size = coords.shape[1]

            node_features = self._create_nodes(len(self.data), self.graph_size - 1, n_depots=1,
                                               features=[coords, demands, twindows])
            return [
                CVRPTWInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=self.graph_size,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=1.0,  # demands are normalized
                    original_capacity=self.data[i][3],
                    time_limit=self.time_limit,
                    BKS=self.bks[str(i)][0] if self.bks is not None else None,
                    instance_id=i,
                    # data_key=self.data_key,
                )
                for i in range(len(self.data))
            ]
        elif isinstance(self.data[0], RPInstance):
            return [
                RPInstance(
                    coords=instance.coords,
                    demands=instance.demands,
                    tw=instance.tw,
                    service_time=instance.service_time,
                    org_service_horizon=instance.org_service_horizon,
                    max_num_vehicles=instance.max_num_vehicles,
                    graph_size=instance.graph_size,
                    vehicle_capacity=instance.vehicle_capacity,
                    depot_idx=instance.depot_idx,
                    type=instance.type,
                    tw_frac=instance.tw_frac
                    # instance_id=i,
                )
                for i, instance in enumerate(self.data)
            ]
        else:
            warnings.warn("Seems not to be the correct test data format for CVRPTW - expected a List of List of "
                          "coordinate instances")
            raise RuntimeError(
                "Unexpected format for CVRPTW Test Instances - make sure that a CVRPTW test set is loaded")

    def _instance_bks_updates(self):
        # always update benchmark data instances with newest BKS registry if registry given for loaded data
        return [
            instance.update(
                original_capacity=instance.original_capacity if instance.original_capacity is not None else
                CVRPTW_DEFAULTS[instance.graph_size - 1][1],
                time_limit=self.time_limit if instance.time_limit is None else instance.time_limit,
                BKS=self.bks[str(instance.instance_id if instance.instance_id is not None else i)][0]
                if self.bks is not None else None,
                instance_id=instance.instance_id if instance.instance_id is not None else i,
            )
            for i, instance in enumerate(self.data)
        ]

    def _denormalize(self):
        # default is normalized demands and 0-1-normed coordinates for generated data
        # --> denormalize for self.normalize = False and update bks registry in meantime (if given)
        logger.info(f'DE-NORMALIZING data ...')
        demands = []
        coords = []
        for i, instance in enumerate(self.data):
            orig_capa = instance.original_capacity if instance.original_capacity is not None \
                else CVRPTW_DEFAULTS[instance.graph_size - 1][1]
            demand_denorm = np.round(instance.node_features[:, -1] * orig_capa)
            coords_denorm = instance.coords * self.grid_size
            demands.append(demand_denorm)
            coords.append(coords_denorm)
        coords = np.stack(coords)
        demands = np.stack(demands)
        self.graph_size = coords.shape[1]  # make sure for loaded data that graph_size matches coords shape
        node_features_denormed = self._create_nodes(len(self.data), self.graph_size - 1, n_depots=1,
                                                    features=[coords, demands])
        self.is_denormed = True
        return [
            instance.update(
                coords=coords[i],
                node_features=node_features_denormed[i],
                original_capacity=instance.original_capacity if instance.original_capacity is not None else
                CVRPTW_DEFAULTS[instance.graph_size - 1][1],
                original_locations=instance.original_locations if instance.original_locations is not None else None,
                instance_id=instance.instance_id if instance.instance_id is not None else i,
                type=instance.type if instance.type is not None else None,
            )
            for i, instance in enumerate(self.data)
        ]

    def _is_feasible(self, sol: RPSolution) -> bool:
        _, _ = self._eval_metric(sol)
        return True

    # @staticmethod
    def feasibility_check(self, instance: CVRPTWInstance,
                          rp_solution: Union[RPSolution, List[list]],
                          costs_with_wait_times: bool = False,
                          is_running: bool = False):
        # Check that sol is valid for CVRPTW
        depot = instance.depot_idx[0]
        coords = instance.coords
        # print('coords[:5]', coords[:5])
        demands = instance.demands  # instance.node_features[:, instance.constraint_idx[0]]
        # print('demands', demands)
        capa = instance.original_capacity if self.is_denormed else instance.vehicle_capacity
        # print('capa', capa)
        # print('instance.original_capacity', instance.original_capacity)
        # print('instance.vehicle_capacity', instance.vehicle_capacity)
        tw = instance.tw
        service_time = instance.service_time
        routes = rp_solution.solution if isinstance(rp_solution, RPSolution) else rp_solution
        # print('routes', routes)
        service_time = np.ones(coords.shape[0]) * instance.service_time if isinstance(service_time, float) \
            else service_time
        # change service time depot to be service horizon --> 10
        # print('instance.service_horizon', instance.service_horizon)
        tw[0] = np.array([0.0, instance.service_horizon])
        # print('tw', tw)

        # check feasibility of routes and calculate cost
        routes_ = []
        if routes is not None:  # or len(solution) == 0:
            k, cost = 0, 0.0
            for r in routes:
                if r:
                    if r[0] != depot:
                        r = [depot] + r
                    if r[-1] != depot:
                        r.append(depot)
                    transit = 0
                    wait_time = 0
                    source = r[0]
                    cum_d = 0
                    t = 0
                    # print('r', r)
                    for target in r[1:]:
                        transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                        arrive_t = t + transit
                        # print('source', source)
                        # print('transit', transit)
                        cum_d += demands[target]
                        # print('cum_d', cum_d)
                        # print(f"time window[ target ] {tw[target]}")
                        # print(f"time window start {tw[target, 0]}")
                        # print(f"time window end {tw[target, 1]}")
                        # print('arrive_t', arrive_t)
                        start_t = max(tw[target, 0], arrive_t)
                        # print('start_t', start_t)
                        wait_time = max(0, tw[target, 0] - arrive_t)
                        # print('wait_time: ', wait_time)
                        wait_time += wait_time
                        if costs_with_wait_times:
                            transit += wait_time
                        due_t = tw[target, 1]
                        serve_t = service_time[target]
                        # end_t = start_t + serve_t
                        # t = end_t
                        transit += serve_t
                        source = target
                        if due_t + 0.001 < start_t:
                            print('due_t', due_t)
                            print('due_t + EPS', due_t + EPS)
                            print('start_t', start_t)
                            print('start_t - EPS', start_t - EPS)
                            warnings.warn(f"CVRPTW solution infeasible. "
                                          f"Time Window constraint violated. Setting cost and k to 'inf'")
                            cost = float("inf")
                            k = float("inf")
                            break
                        if cum_d > capa + EPS:
                            print('cum_d', cum_d)
                            warnings.warn(f"CVRPTW solution infeasible. "
                                          f"Capacity constraint violated. Setting cost and k to 'inf'")
                            cost = float("inf")
                            k = float("inf")
                            break
                    cost += transit
                    k += 1
                    routes_.append(r)
        else:
            warnings.warn(f"No CVRPTW solution specified (None). setting cost and k to 'inf'")
            cost = float("inf")
            k = float("inf")
            routes_ = None
        return cost, k, routes_

    # function for  downloading dataset  for VRP-TW
    def read_solomon(self, store_path, normalize=False):
        instances = {}
        features = []
        cpt = 0
        with open(store_path, 'r') as f:
            lines = f.readlines()
            name = lines[0].split()
            max_vehicle_number, vehicle_capacity = list(map(int, lines[4].split()))
            instances['max_vehicle_number'] = max_vehicle_number
            instances['vehicle_capacity'] = vehicle_capacity
            for line in lines[9:]:
                line = line.strip().split()
                if len(line) == 7:
                    instance = {}
                    instance['index'] = cpt
                    instance['cust_number'] = line[0]
                    instance['x_coord'] = float(line[1])
                    instance['y_coord'] = float(line[2])
                    instance['demand'] = float(line[3])
                    instance['tw_start'] = float(line[4])
                    instance['tw_end'] = float(line[5])
                    instance['service_time'] = float(line[6])
                    instance['is depot'] = True if cpt == 0 else False
                    features.append(instance)
                    cpt += 1
                else:
                    continue

        instances['depot_idx'] = 0
        df = pd.DataFrame(data=features)
        df.set_index('index')
        df.drop(labels='index', axis=1, inplace=True)
        df['tw_len'] = df.tw_end - df.tw_start
        instances['features'] = df

        return instances

    def normalize_instances(self, instances):
        df = deepcopy(instances['features'])
        service_horizon = df['tw_end'][0]
        # To do: Check normalization in Homberger Dataset

        df['x_coord'] /= 100
        df['y_coord'] /= 100
        df['demand'] /= instances['vehicle_capacity']
        df['tw_start'] /= service_horizon
        df['tw_end'] /= service_horizon
        df['service_time'] /= service_horizon
        df['tw_len'] /= service_horizon

        # to calculate the correct distance matrix of normalized coordinates interpreted as times
        # for constraint checking, one needs to correct for the normalization
        # instance['dist_to_time_factor'] = 100 / service_horizon
        instances['org_service_horizon'] = service_horizon
        instances['norm_features'] = df

        return instances

    def to_cvrptw_instance(self, instances):
        df = instances['norm_features']
        dloc = df.loc[[0], ('x_coord', 'y_coord')].to_numpy()
        depot_tw = df.loc[[0], ('tw_start', 'tw_end')].to_numpy()
        nloc = df.loc[:, ('x_coord', 'y_coord')].to_numpy()
        demand = df.loc[:, 'demand'].to_numpy()
        tw_len = df.loc[:, ('tw_start', 'tw_end')].to_numpy()
        service_time = df.loc[:, 'service_time'].to_numpy()
        node_tw = [df.loc[1:, ('tw_start', 'tw_end')].to_numpy()]

        # service time in Solomon data is constant for each instance, so mean == exact value
        # self.service_time = self.norm_summary.loc['mean', 'service_time']

        if instances['vehicle_capacity'] < 500:
            type = "1"
        else:
            type = "2"

        # infer TW fraction
        org_df = instances['features'].loc[1:, :]  # without depot!
        has_tw = (org_df.tw_start != 0)
        tw_frac = has_tw.sum() / org_df.shape[0]

        node_features = self._create_nodes(1, nloc.shape[0], n_depots=1, features=[np.stack([dloc, nloc]), demand,
                                                                                   np.stack([depot_tw, node_tw])])

        return CVRPTWInstance(
            depot_idx=[0],
            coords=torch.tensor(np.stack([dloc, nloc]), dtype=torch.float),
            demands=torch.tensor(demand, dtype=torch.float),
            node_features=node_features,
            graph_size=nloc.shape[0],
            org_service_horizon=instances['org_service_horizon'],  # torch.tensor(, dtype=torch.float),
            max_vehicle_number=instances['max_vehicle_number'],  # torch.tensor(, dtype=torch.float),
            vehicle_capacity=1.0,
            depot_tw=torch.tensor(depot_tw, dtype=torch.float),
            node_tw=torch.tensor(node_tw, dtype=torch.float),
            service_time=service_time,
            service_horizon=1.0,
            tw_frac=tw_frac,  # torch.tensor(tw_frac, dtype=torch.float)
        )
        # data = {"depot_idx": torch.tensor(dloc, dtype=torch.float),
        #         "coords": torch.tensor(nloc, dtype=torch.float),
        #         'demands': torch.tensor(demand, dtype=torch.float),
        #         'graph_size': nloc.shape[0],
        #         'org_service_horizon': torch.tensor(instances['org_service_horizon'], dtype=torch.float),
        #         'max_vehicle_number': torch.tensor(instances['max_vehicle_number'], dtype=torch.float),
        #         'capacity': 1.0,
        #         'depot_tw': torch.tensor(depot_tw, dtype=torch.float),
        #         'node_tw': torch.tensor(node_tw, dtype=torch.float),
        #         'service_horizon': 1.0,
        #         'tw_frac': torch.tensor(tw_frac, dtype=torch.float)}

        return data

    def get_vrptw_instances(self):
        """
        This CVRPTW benchmark dataset consiss of instances from Solomon and Gehring & Homberger.
        Source:  "https://www.sintef.no/globalassets/project/top/cvrptw/..."
        Solomon : 100 instances
        Gehring & Homberger  : 200,  400, 600, 800, 1000 instances
        """

        print('Downloading started...')

        base_url = "https://www.sintef.no"
        homberger_index_path = "/globalassets/project/top/vrptw/homberger/%s/homberger_%s_customer_instances.zip"
        solomon_index_path = "/globalassets/project/top/vrptw/solomon/solomon-100.zip"
        num_customers_list = ["100", "200", "400", "600", "800", "1000"]
        # pattern = re.compile('"(/contentassets/.*?/((c|r|rc)\d+_\d+_\d+).*?\.txt)"')
        # headers = {"User-Agent": ""}
        save_paths = []
        folder_names = []
        for num_customers in num_customers_list:
            if num_customers == "100":
                url_list = [base_url + solomon_index_path]
            else:
                url_list = [base_url + homberger_index_path % (num_customers, num_customers)]
            # Downloading the file by sending the request to the URL
            for url in url_list:
                a = urlparse(url)
                # printa(a.path)
                # print(os.path.basename(a.path))
                folder_name = os.path.basename(a.path).split(".")[0]
                print("Downloading... ", folder_name)
                cur_dir = os.path.dirname(os.path.abspath('__file__'))
                save_path = cur_dir + '/data/test_data/cvrptw/' + folder_name
                folder_names.append(folder_name)
                save_paths.append(save_path)
                req = requests.get(url)

                # extracting the zip file contents
                zipfile = ZipFile(BytesIO(req.content))
                zipfile.extractall(save_path)

        print('Downloading Completed! \n raw data is available at', cur_dir + '/test_data/cvrptw/')
        return save_paths, folder_names

    def prepare_solomon_instances(self, save_paths, folder_names):

        # group the same file type together for preparing .pkl file.
        GROUPS = ["c1", "c2", "r1", "r2", "rc1", "rc2"]
        print(f'Start preparing files...')

        for path in save_paths:
            if os.path.basename(path).split(".")[0] == folder_names[0]:
                LPATH = "./data/test_data/cvrptw/solomon-100/In"
                DATA_SPATH = "./data/test_data/solomon_prep.pkl"
            elif os.path.basename(path).split(".")[0] == folder_names[1]:
                LPATH = "./data/test_data/cvrptw/" + folder_names[1]
                DATA_SPATH = "./data/test_data/" + folder_names[1] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[2]:
                LPATH = "./data/test_data/cvrptw/" + folder_names[2]
                DATA_SPATH = "./data/test_data/" + folder_names[2] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[3]:
                LPATH = "./data/test_data/cvrptw/" + folder_names[3]
                DATA_SPATH = "./data/test_data/" + folder_names[3] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[4]:
                LPATH = "./data/test_data/cvrptw/" + folder_names[4]
                DATA_SPATH = "./data/test_data/" + folder_names[4] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[5]:
                LPATH = "./data/test_data/cvrptw/" + folder_names[5]
                DATA_SPATH = "./data/test_data/" + folder_names[5] + '.pkl'

            test = os.listdir(LPATH)
            instances = {}
            for g in sorted(GROUPS):
                groups = [list(c) for _, c in itertools.groupby(sorted(test), (lambda x: x[:2]))]
                groups[4] = [list(c) for _, c in itertools.groupby(sorted(groups[4]), (lambda x: x[:3]))]
                groups = groups[0], groups[1], groups[2], groups[3], groups[4][0], groups[4][1]

            for p in range(len(GROUPS)):
                print(f"Data Type:{GROUPS[p]}")

                data = []
                for file in groups[p]:
                    filename = os.path.join(LPATH, file)
                    pth = os.path.join(LPATH, file)
                    print(f"preparing file '{file}' from {pth}")
                    instance = self.read_solomon(pth)
                    instance = self.normalize_instances(instance)
                    data.append(instance)

                buffer = {'tw_frac=0.25': [], 'tw_frac=0.5': [], 'tw_frac=0.75': [], 'tw_frac=1.0': []}
                # infer tw frac of instance
                for instance in data:
                    org_df = instance['features'].loc[1:, :]  # without depot!
                    has_tw = (org_df.tw_start != 0)
                    tw_frac = has_tw.sum() / org_df.shape[0]
                    for instance in data:
                        buffer[f"tw_frac={tw_frac}"].append(self.to_vrptw_instance(instance))

                instances[p] = buffer

                with open(DATA_SPATH, 'wb') as f:
                    pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)
                path_name = os.path.dirname(DATA_SPATH)
                file_name = os.path.basename(DATA_SPATH)

            print(f"{file_name} is saved in directory {path_name}")

        return instances

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
