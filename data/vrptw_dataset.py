from data.base_dataset import BaseDataset
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any
from abc import ABC
import os
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
from formats import RPInstance, RPSolution
import logging
from urllib.request import urlopen, Request, urlparse
from zipfile import ZipFile
from io import BytesIO
from copy import deepcopy

EPS = np.finfo(np.float32).eps


class VRPTWDataset(BaseDataset, ABC):
    """Creates VRPTW data samples to use for training or evaluating benchmark models"""

    def __init__(self,
                 is_train: bool = False,
                 store_path: str = None,
                 seed: int = None,
                 num_samples: int = 100000,
                 normalize: bool = True,
                 offset: int = 0,
                 distribution: Optional[str] = None,
                 graph_size: int = 20,
                 num_vehicles: int = 5,
                 capacity: int = 30,
                 max_cap_factor: float = 1.1,
                 device: str = None,
                 **kwargs):
        super(VRPTWDataset, self).__init__(problem='vrptw',
                                           store_path=store_path,
                                           num_samples=num_samples,
                                           seed=seed)

        self.is_train = is_train
        self.num_samples = num_samples
        self.normalize = normalize
        self.offset = offset
        self.distribution = distribution
        self.graph_size = graph_size
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.max_cap_factor = max_cap_factor
        self.device = device
        self.BKS = self.get_curr_BKS(problem='vrptw')

        # if is_train is False:
        if store_path is not None:
            # load OR DOWNLOAD (test) data
            self.data = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            # Transform loaded data to CVRPInstance format
            self.data = self._make_VRPTWInstance()
        else:
            # sample data
            self.sample(sample_size=self.num_samples,
                        graph_size=self.graph_size,
                        distribution=self.distribution,
                        k=self.num_vehicles,
                        cap=self.capacity,
                        max_cap_factor=self.max_cap_factor)
        self.size = len(self.data)

    def _download(self):
        save_paths, folder_names = self.get_vrptw_instances()
        self.data = self.prepare_solomon_instances(save_paths, folder_names)
        return self.data

    def _is_feasible(self, sol: RPSolution) -> bool:
        _, _ = self._eval_metric(sol)
        return True

    def _eval_metric(self, solution: RPSolution,
                     PI_Evaluation: bool = False,
                     passMark: int = None) -> Tuple[RPSolution, Union[float, None]]:
        """(Re-)Evaluate provided solutions for the CVRP."""
        data = solution.instance
        depot = data.depot_idx[0]
        coords = data.coords
        demands = data.node_features[:, data.constraint_idx[0]]
        tw = data.tw
        service_time = data.service_time
        routes = solution.solution

        # check feasibility of routes and calculate cost
        k = 0
        cost = 0.0
        for r in routes:
            if r:
                if r[0] != depot:
                    r = [depot] + r
                if r[-1] != depot:
                    r.append(depot)
                transit = 0
                source = r[0]
                cum_d = 0
                t = 0
                for target in r[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    arrive_t = t + transit
                    cum_d += demands[target]
                    print(f"time window[ source ] {tw[target]}")
                    print(f"time window start {tw[target, 0]}")
                    print(f"time window end {tw[target, 1]}")
                    start_t = max(tw[target, 0], arrive_t)
                    wait_time = max(0, tw[target, 0] - arrive_t)
                    print('wait_time: ', wait_time)
                    transit += wait_time
                    due_t = tw[target, 1]
                    serve_t = service_time[target]
                    end_t = start_t + serve_t
                    t = end_t
                    source = target
                    if due_t < start_t:
                        warnings.warn(f"VRPTW solution infeasible. "
                                      f"Time Window constraint violated. Setting cost and k to 'inf'")
                        cost = float("inf")
                        k = float("inf")
                if cum_d > 1.0 + EPS:
                    warnings.warn(f"VRPTW solution infeasible. "
                                  f"Capacity constraint violated. Setting cost and k to 'inf'")
                    cost = float("inf")
                    k = float("inf")
                    break
                cost += transit
                k += 1

        if PI_Evaluation:
            # calculate Primal Integral as in DIMACS challenge (see BaseDataset for func. compute_pi)
            # --> possible only if some Best Known Solution (BKS) is available
            pi_score, prev_cost, prev_time = self.compute_pi(cost,
                                                             solution.run_time,
                                                             self.BKS,
                                                             data.time_limit,
                                                             passMark,
                                                             solution.last_cost,
                                                             solution.last_runtime)

            return solution.update(cost=cost, num_vehicles=k, last_cost=prev_cost, last_runtime=prev_time), pi_score
        else:
            return solution.update(cost=cost, num_vehicles=k), None

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

    def to_rp_instance(self, instances):
        df = instances['norm_features']
        dloc = df.loc[[0], ('x_coord', 'y_coord')].to_numpy()
        depot_tw = df.loc[[0], ('tw_start', 'tw_end')].to_numpy()
        nloc = df.loc[:, ('x_coord', 'y_coord')].to_numpy()
        demand = df.loc[:, 'demand'].to_numpy()
        tw_len = df.loc[:, ('tw_start', 'tw_end')].to_numpy()
        service_time = df.loc[:, 'service_time'].to_numpy()
        node_tw = [df.loc[1:, ('tw_start', 'tw_end')].to_numpy()]

        if instances['vehicle_capacity'] < 500:
            type = "1"
        else:
            type = "2"

        # infer TW fraction
        org_df = instances['features'].loc[1:, :]  # without depot!
        has_tw = (org_df.tw_start != 0)
        tw_frac = has_tw.sum() / org_df.shape[0]

        data = {"depot_idx": torch.tensor(dloc, dtype=torch.float),
                "coords": torch.tensor(nloc, dtype=torch.float),
                'demands': torch.tensor(demand, dtype=torch.float),
                'graph_size': nloc.shape[0],
                'org_service_horizon': torch.tensor(instances['org_service_horizon'], dtype=torch.float),
                'max_vehicle_number': torch.tensor(instances['max_vehicle_number'], dtype=torch.float),
                'capacity': 1.0,
                'depot_tw': torch.tensor(depot_tw, dtype=torch.float),
                'node_tw': torch.tensor(node_tw, dtype=torch.float),
                'service_horizon': 1.0,
                'tw_frac': torch.tensor(tw_frac, dtype=torch.float)}

        return data

    def get_vrptw_instances(self):
        """
        This VRPTW benchmark dataset consiss of instances from Solomon and Gehring & Homberger.
        Source:  "https://www.sintef.no/globalassets/project/top/vrptw/..."
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
                save_path = cur_dir + '/data/test_data/vrptw/' + folder_name
                folder_names.append(folder_name)
                save_paths.append(save_path)
                req = requests.get(url)

                # extracting the zip file contents
                zipfile = ZipFile(BytesIO(req.content))
                zipfile.extractall(save_path)

        print('Downloading Completed! \n raw data is available at', cur_dir + '/test_data/vrptw/')
        return save_paths, folder_names

    def prepare_solomon_instances(self, save_paths, folder_names):

        # group the same file type together for preparing .pkl file.
        GROUPS = ["c1", "c2", "r1", "r2", "rc1", "rc2"]
        print(f'Start preparing files...')

        for path in save_paths:
            if os.path.basename(path).split(".")[0] == folder_names[0]:
                LPATH = "./data/test_data/vrptw/solomon-100/In"
                DATA_SPATH = "./data/test_data/solomon_prep.pkl"
            elif os.path.basename(path).split(".")[0] == folder_names[1]:
                LPATH = "./data/test_data/vrptw/" + folder_names[1]
                DATA_SPATH = "./data/test_data/" + folder_names[1] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[2]:
                LPATH = "./data/test_data/vrptw/" + folder_names[2]
                DATA_SPATH = "./data/test_data/" + folder_names[2] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[3]:
                LPATH = "./data/test_data/vrptw/" + folder_names[3]
                DATA_SPATH = "./data/test_data/" + folder_names[3] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[4]:
                LPATH = "./data/test_data/vrptw/" + folder_names[4]
                DATA_SPATH = "./data/test_data/" + folder_names[4] + '.pkl'
            elif os.path.basename(path).split(".")[0] == folder_names[5]:
                LPATH = "./data/test_data/vrptw/" + folder_names[5]
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
                        buffer[f"tw_frac={tw_frac}"].append(self.to_rp_instance(instance))

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
