from data.base_dataset import BaseDataset
from data.dataset_utils import CVRP_DEFAULTS, CVRPLIB_LINKS, SCALE_FACTORS_CVRP, XE_UCHOA_TYPES, EPS
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any, Callable
from omegaconf import DictConfig, ListConfig
import warnings
import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import shutil
import torch
import numpy as np
from formats import CVRPInstance, RPSolution
from models.runner_utils import NORMED_BENCHMARKS, get_budget_per_size, _adjust_time_limit
# from data.data_utils import prepare_sol_instances
import logging

logger = logging.getLogger(__name__)


class CVRPDataset(BaseDataset):
    """Creates VRP data samples to use for training or evaluating benchmark models"""

    def __init__(self,
                 is_train: bool = False,
                 store_path: str = None,
                 dataset_size: int = None,
                 dataset_range: list = None,
                 seed: int = None,
                 num_samples: int = 100,
                 normalize: bool = True,
                 offset: int = 0,
                 distribution: Optional[str] = None,
                 generator_args: dict = None,
                 sampling_args: Optional[dict] = None,
                 graph_size: int = 20,
                 grid_size: int = 1,
                 num_vehicles: int = None,
                 capacity: int = 30,
                 max_cap_factor: float = 1.1,
                 float_prec: np.dtype = np.float32,
                 transform_func: Callable = None,
                 transform_args: DictConfig = None,
                 verbose: bool = False,
                 TimeLimit: Union[int, float] = None,
                 machine_info: tuple = None,
                 load_bks: bool = True,
                 load_base_sol: bool = True,
                 re_evaluate: bool = False,
                 ):
        super(CVRPDataset, self).__init__(problem='cvrp',
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
        self.normalize = normalize
        self.offset = offset
        self.dataset_size = dataset_size
        self.dataset_range = dataset_range
        self.distribution = distribution
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.graph_size = graph_size
        print('self.distribution', self.distribution)
        self.grid_size = grid_size if self.distribution in ["uchoa", "XML"] else 1000
        print('self.grid_size', self.grid_size)
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.time_limit = TimeLimit
        self.machine_info = machine_info
        if self.machine_info is not None:
            print('self.machine_info in Dataset', machine_info)
        self.re_evaluate = re_evaluate
        self.metric = None
        self.max_cap_factor = max_cap_factor
        self.transform_func = transform_func
        self.transform_args = transform_args
        self.data_key = None
        self.scale_factor = None
        self.is_denormed = False

        if store_path is not None:
            # load or download (test) data
            self.data, self.data_key = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            if self.dataset_size is not None and self.dataset_size < len(self.data):
                self.data = self.data[:self.dataset_size]
            elif self.dataset_range is not None:
                self.data = self.data[self.dataset_range[0]:self.dataset_range[1]]
            logger.info(f"{len(self.data)} CVRP Test/Validation Instances for {self.problem} with {self.graph_size} "
                        f"{self.distribution}-distributed customers loaded.")
            # Transform loaded data to CVRPInstance format IF NOT already is in format
            # print('self.normalize', self.normalize)
            # print('self.generator_args.normalize_demands', self.generator_args.normalize_demands)
            if not isinstance(self.data[0], CVRPInstance):
                self.data = self._make_CVRPInstance()
            if not self.normalize and not self.is_denormed:
                self.data = self._denormalize()
            elif self.normalize and self.generator_args is not None:
                if not self.generator_args.normalize_demands:
                    self.data = self._denormalize(grid_size=1.0)
            if self.bks is not None:
                self.data = self._instance_bks_updates()
            if self.time_limit is not None:
                self.data = self._instance_bks_updates()
            if self.transform_func is not None:  # transform_func needs to return list
                self.data_transformed = self.transform_func(self.data)
            # if not self.data_key.startswith('X') or self.data_key.startswith('S'):
            self.size = len(self.data)
        elif not is_train:
            logger.info(f"No file path for evaluation specified.")
            if self.distribution is not None:
                # and self.sampling_args['sample_size'] is not None:
                logger.info(f"Sampling data according to env config file: {self.sampling_args}")
                self.sample(sample_size=self.sampling_args['sample_size'],
                            graph_size=self.graph_size,
                            distribution=self.distribution,
                            log_info=True)
                if not self.normalize and not self.is_denormed:
                    self.is_denormed = True
            else:
                logger.info(f"Data configuration not specified in env config, "
                            f"defaulting to 100 uniformly distributed VRP20 instances")
                self.sample(sample_size=100,
                            graph_size=20,
                            distribution="uniform",
                            log_info=True)
                if not self.normalize and not self.is_denormed:
                    self.is_denormed = True
            self.size = len(self.data)
        else:  # no data to load - but initiated CVRPDataset for sampling in training loop
            logger.info(f"No data loaded - initiated CVRPDataset with env config for sampling in training...")
            self.size = None
            self.data = None

    def _download(self, extract_to='.', from_platform='CVRPLIB'):
        """Download CVRPLIB Datasets"""
        url = None
        # default: extract to existing self.store_path
        extract_to = os.path.dirname(self.store_path)
        if from_platform == 'CVRPLIB':
            url = CVRPLIB_LINKS[os.path.basename(self.store_path)][0]
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)
        # remove empty placeholder directory
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        # move data from double folders directly to main data directory, e.g. dimacs/D (instead of dimacs/Vrp-Set-D...)
        original = extract_to + "/Vrp-Set-" + os.path.basename(self.store_path) + \
                   "/Vrp-Set-" + os.path.basename(self.store_path) + "/" + os.path.basename(self.store_path)
        # move downloaded files to self.stor_path
        try:
            shutil.move(original, extract_to)
        except FileNotFoundError:
            original = extract_to + "/Vrp-Set-" + os.path.basename(self.store_path) + "/" \
                       + os.path.basename(self.store_path)
            # os.makedirs(extract_to)
            shutil.move(original, extract_to)
        # remove empty Vrp-Set-" " directories
        shutil.rmtree(extract_to + "/Vrp-Set-" + os.path.basename(self.store_path))

    def _make_CVRPInstance(self):
        """Reformat (loaded) test instances as CVRPInstances"""
        if isinstance(self.data[0][0], List) or self.data_key == 'uniform':
            logger.info("Transforming instances to CVRPInstances")
            warnings.warn("This works only for Nazari et al type of data")
            coords, demands = [], []
            for i in range(len(self.data)):
                if self.normalize:
                    coords_i = np.vstack((self.data[i][0], self.data[i][1])) / self.grid_size
                    demands_i = np.array(self.data[i][2])
                    demands.append(np.insert(demands_i, 0, 0) / self.data[i][3])
                    coords.append(coords_i)
                else:
                    coords_i = np.vstack((self.data[i][0], self.data[i][1]))
                    demands_i = np.array(self.data[i][2])
                    demands.append(np.insert(demands_i, 0, 0))
                    coords.append(coords_i)
                    self.is_denormed = True
            coords = np.stack(coords)
            demands = np.stack(demands)

            self.graph_size = coords.shape[1]

            node_features = self._create_nodes(len(self.data), self.graph_size - 1, n_depots=1,
                                               features=[coords, demands])
            return [
                CVRPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=self.graph_size,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=1.0,  # demands are normalized
                    original_capacity=self.data[i][3],  # original_capacity for NLNS, DPDP
                    time_limit=self.time_limit,  # TODO: schematic approach to time limit
                    BKS=self.bks[str(i)][0] if self.bks is not None else None,
                    instance_id=i,
                    # data_key=self.data_key,
                )
                for i in range(len(self.data))
            ]

    def _denormalize(self, grid_size=None):
        # default is normalized demands and 0-1-normed coordinates for generated data
        # --> denormalize for self.normalize = False and update bks registry in meantime (if given)
        self.grid_size = self.grid_size if grid_size is None else grid_size
        logger.info(f'DE-NORMALIZING data with grid-size {self.grid_size} '
                    f'and capacity {self.data[0].original_capacity}...')
        demands = []
        coords = []
        for i, instance in enumerate(self.data):
            orig_capa = instance.original_capacity if instance.original_capacity is not None \
                else CVRP_DEFAULTS[instance.graph_size - 1][1]
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
            CVRPInstance(
                coords=coords[i],
                node_features=node_features_denormed[i],
                graph_size=instance.graph_size,
                constraint_idx=instance.constraint_idx,  # demand is at last col position of node features
                vehicle_capacity=instance.vehicle_capacity,  # demands are normalized by default
                original_capacity=instance.original_capacity if instance.original_capacity is not None else
                CVRP_DEFAULTS[instance.graph_size - 1][1],
                time_limit=self.time_limit,
                BKS=self.bks[str(instance.instance_id if instance.instance_id is not None else i)][0]
                if self.bks is not None else None,
                instance_id=instance.instance_id if instance.instance_id is not None else i,
                coords_dist=instance.coords_dist,
                depot_type=instance.depot_type,
                demands_dist=instance.demands_dist,
                original_locations=instance.original_locations if instance.original_locations is not None else None,
                type=instance.type if instance.type is not None else None,
            )
            for i, instance in enumerate(self.data)
        ]

    def _instance_bks_updates(self):
        # always update benchmark data instances with newest BKS registry if registry given for loaded data
        return [
            CVRPInstance(
                coords=instance.coords,
                node_features=instance.node_features,
                graph_size=instance.graph_size,
                constraint_idx=instance.constraint_idx,  # demand is at last position of node features
                vehicle_capacity=instance.vehicle_capacity,  # demands are normalized
                original_capacity=instance.original_capacity if instance.original_capacity is not None else
                CVRP_DEFAULTS[instance.graph_size - 1][1],
                time_limit=self.time_limit if instance.time_limit is None else instance.time_limit,
                BKS=self.bks[str(instance.instance_id if instance.instance_id is not None else i)][0]
                if self.bks is not None else None,
                instance_id=instance.instance_id if instance.instance_id is not None else i,
                coords_dist=instance.coords_dist,
                depot_type=instance.depot_type,
                demands_dist=instance.demands_dist,
                original_locations=instance.original_locations if instance.original_locations is not None else None,
                type=instance.type if instance.type is not None else None,
            )
            for i, instance in enumerate(self.data)
        ]

    # def _get_costs(self, sol: RPSolution) -> Tuple[float, int, bool, List[list]]:
    #     # perform problem-specific feasibility check while getting routing costs
    #     cost, k, solution_upd = self.feasibility_check(sol.instance, sol.solution)
    #     is_feasible = True if cost != float("inf") else False
    #     return cost, k, is_feasible, solution_upd

    # @staticmethod
    # def return_infeasible_sol(mode, instance, solution, cost, nr_vs):
    #     if mode in ['wrap', 'pi'] or 'wrap' in mode or 'pi' in mode:
    #         logger.info(f"Metric Analysis for instance {instance.instance_id} cannot be performed. No feasible "
    #                     f"solution provided in Time Limit. Setting PI score to 10 and WRAP score to 1.")
    #         pi_ = 10 if mode == 'pi' or 'pi' in mode else None
    #         wrap_ = 1 if mode == 'wrap' or 'wrap' in mode else None
    #         return solution.update(cost=cost, pi_score=pi_, wrap_score=wrap_, num_vehicles=nr_vs), None, None
    #     else:
    #         return solution.update(cost=cost, num_vehicles=nr_vs), None, None

    # def eval_costs(self, mode: str, instance: CVRPInstance, v_costs: list, v_times: list, orig_r_times: list,
    #                model_name: str):
    #     return self._eval_metric(model_name=model_name,
    #                              inst_id=str(instance.instance_id),
    #                              instance=instance,
    #                              verified_costs=v_costs,
    #                              verified_times=v_times,
    #                              run_times_orig=orig_r_times,
    #                              eval_type=mode)

    def feasibility_check(self, instance: CVRPInstance, rp_solution: RPSolution, is_running: bool = False):
        solution = rp_solution.solution if isinstance(rp_solution, RPSolution) else rp_solution
        # print('solution in cvrp_dataset-feasibility_check: ', solution)
        depot = instance.depot_idx[0]
        coords = instance.coords.astype(int) if self.is_denormed and isinstance(instance.coords[0][0], np.int64) \
            else instance.coords
        # print('coords', coords)
        # if self.scale_factor is None else (instance.coords * self.scale_factor).astype(int)
        demands = instance.node_features[:, instance.constraint_idx[0]] if self.is_denormed \
            else instance.node_features[:, instance.constraint_idx[0]]
        # print('self.is_denormed', self.is_denormed)
        # print('demands in cvrp_dataset-feasibility_check: ', demands)
        # demands = np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity)
        # print('demands[:10]', demands[:10])
        # * instance.original_capacity).astype(int)
        # np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity, 3).astype(int)
        routes = solution if solution else None  # check if solution list is not empty - if empty set to None
        # capacity = instance.original_capacity
        capacity = instance.original_capacity if self.is_denormed else instance.vehicle_capacity
        # print('capacity in cvrp_dataset-feasibility_check: ', capacity)
        routes_ = []
        visited_nodes = [0]
        # print('routes in cvrp_dataset', routes)
        if routes is not None:  # or len(solution) == 0:
            k, cost = 0, 0  # .0
            for r in routes:
                if r:
                    if r[0] != depot:
                        r = [depot] + r
                    if r[-1] != depot:
                        r.append(depot)
                    transit = 0
                    source = r[0]
                    cum_d = 0
                    for target in r[1:]:
                        transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                        cum_d += demands[target]
                        source = target
                    # print('cum_d', cum_d)
                    # print('transit', transit)
                    if cum_d > capacity + EPS:
                        if is_running:
                            warnings.warn(f"One of the solutions in the trajectory for instance {instance.instance_id} "
                                          f"is infeasible: {cum_d}>{capacity + EPS}. Setting cost and k to 'inf'.")

                        else:
                            warnings.warn(f"cumulative demand {cum_d} surpasses (normalized) capacity "
                                          f"{capacity} for instance with ID {instance.instance_id}.")
                            warnings.warn(f"Final CVRP solution {solution} is infeasible for instance "
                                          f"with ID {instance.instance_id}. Setting cost and k to 'inf'.")
                        cost = float("inf")
                        k = float("inf")
                        break
                    cost += transit
                    k += 1
                    routes_.append(r)
                # print('r', r)
                visited_nodes.extend(r[1:-1])
            # print('visited_nodes', visited_nodes)
            visited_nodes.sort()
            # print('visited_nodes (sorted)', visited_nodes)
            if visited_nodes != list(np.arange(len(demands))):
                warnings.warn(f"Not all nodes covered:  "
                              f"\n list of node IDs={list(np.arange(len(demands)))} - "
                              f"\n visit_nodes.sort={visited_nodes} for instance with ID {instance.instance_id}."
                              f"\n Missing nodes are: {list(set(list(np.arange(len(demands)))) - set(visited_nodes))}")
                warnings.warn(f"Final CVRP solution {solution} is infeasible for instance "
                              f"with ID {instance.instance_id}. Setting cost and k to 'inf'.")
                cost = float("inf")
                k = float("inf")
                routes_ = None
        else:
            warnings.warn(f"No CVRP solution specified (None). setting cost and k to 'inf'")
            cost = float("inf")
            k = float("inf")
            routes_ = None
        return cost, k, routes_

    def read_vrp_instance(self, filepath: str):
        """
        taken from l2o meta
        For loading and parsing benchmark instances in CVRPLIB format esp for NLNS.
        """
        file = open(filepath, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        cap = 1.0
        dimension, locations, demand, node_features, capacity, K = None, None, None, None, None, None
        overall_inst_type, X_inst_type = "unknown", None  # for Uchoa (XE) type data store depot, coord distrib. type
        inst_id, int_loc = None, True
        overall_inst_type = self.store_path.split(os.sep)[-2] if self.store_path.split(os.sep)[-2] != "cvrp" \
            else self.store_path.split(os.sep)[-1]
        # print('overall_inst_type', overall_inst_type)
        self.scale_factor = SCALE_FACTORS_CVRP[overall_inst_type] if overall_inst_type in SCALE_FACTORS_CVRP.keys() else 1
        # print('self.scale_factor', self.scale_factor)
        # print('os.path.dirname(filepath).split(os.sep)', os.path.dirname(filepath).split(os.sep))
        while i < len(lines):
            line = lines[i]
            if line.startswith("NAME"):
                name = line.split(':')[1].strip()
                if os.path.dirname(filepath).split(os.sep)[-2] == "XE":
                    X_inst_type = os.path.dirname(filepath).split(os.sep)[-1]
                    inst_id = name.split('_')[-1]
                elif os.path.dirname(filepath).split(os.sep)[-1] == "X":
                    X_inst_type = 'X'
                    inst_id = name
                else:
                    inst_id = name
                if "k" in name.split("-")[-1]:
                    K = name.split("-")[-1][1:]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
                node_features = np.zeros((dimension, 3), dtype=np.single)
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                try:
                    locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                except ValueError:
                    locations = np.loadtxt(lines[i + 1:i + 1 + dimension])
                    int_loc = False
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension

            i += 1
        # print('X_inst_type', X_inst_type)
        original_locations = locations[:, 1:]
        # print('self.normalize', self.normalize)
        # normalize coords and demands
        locations = original_locations / int(self.scale_factor) if self.normalize else original_locations
        demand = demand[:, 1:].squeeze() / capacity if self.normalize else demand[:, 1:].squeeze()
        self.is_denormed = True if not self.normalize else False  # flag for denormalized input
        self.scale_factor = None if not self.normalize else self.scale_factor  # reset scale factor if locs are unscaled
        # print('self.is_denormed', self.is_denormed)
        # print('self.scale_factor', self.scale_factor)

        assert locations.max() <= 1000
        assert demand.min() >= 0
        node_features[:, :2] = locations
        node_features[:, -1] = demand / 1.0
        # print('node_features[:, -1]', node_features[:, -1])
        # add additional indicators
        depot_1_hot = np.zeros(dimension, dtype=np.single)
        depot_1_hot[0] = 1
        customer_1_hot = np.ones(dimension, dtype=np.single)
        customer_1_hot[0] = 0

        # set per instance time limit
        adj_per_inst_tl = None
        if self.time_limit is None and self.bks is not None and not self.re_evaluate:
            # print('dimension:', dimension)
            per_inst_tl = get_budget_per_size(problem_size=dimension)
            pass_mark, pass_mark_cpu, device, nr_threads, ls_on_top = self.machine_info
            if not ls_on_top:
                adj_per_inst_tl = _adjust_time_limit(per_inst_tl, pass_mark, device, nr_threads)
            else:
                adj_per_inst_tl = _adjust_time_limit(per_inst_tl, pass_mark_cpu, device, nr_threads)
        # print('adj_per_inst_tl', adj_per_inst_tl)
        # adj_per_inst_tl = adj_per_inst_tl*(1/100) if adj_per_inst_tl is not None else None
        # print('re_adjust for testing: ', adj_per_inst_tl)

        return CVRPInstance(
            coords=locations,
            node_features=np.concatenate((
                depot_1_hot[:, None],
                customer_1_hot[:, None],
                node_features
            ), axis=-1),
            graph_size=dimension,
            constraint_idx=[-1],  # demand is at last position of node features
            vehicle_capacity=1.0,  # demands are normalized
            time_limit=self.time_limit if adj_per_inst_tl is None else adj_per_inst_tl,
            original_capacity=capacity,
            original_locations=original_locations,
            coords_dist=XE_UCHOA_TYPES[X_inst_type][1] if X_inst_type not in [None, "X"] else None,
            depot_type=XE_UCHOA_TYPES[X_inst_type][0] if X_inst_type not in [None, "X"] else None,
            demands_dist=XE_UCHOA_TYPES[X_inst_type][2] if X_inst_type not in [None, "X"] else None,
            instance_id=inst_id,  # int(inst_id)
            type=X_inst_type if X_inst_type is not None else overall_inst_type,
            # demands_dist=None,
            max_num_vehicles=int(K) if K is not None else None,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


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
    coord_samp = ['nazari', 'uchoa']
    k = 4
    cap = 9
    max_cap_factor = 1.1
    verb = True

    for csmp in coord_samp:
        # for wsmp in weight_samp:
        ds = CVRPDataset(num_samples=size,
                         distribution=csmp,
                         graph_size=n,
                         num_vehicles=k,
                         capacity=9,
                         max_cap_factor=max_cap_factor,
                         seed=seed,
                         normalize=True,
                         verbose=verb)
        # ds.sample() --> already in CVRPDataset initialisation
        print('ds.data[0]', ds.data[0])
        print('ds.size', ds.size)
