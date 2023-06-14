from data.base_dataset import BaseDataset
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any, Callable
from omegaconf import DictConfig, ListConfig
import warnings
from abc import ABC
import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import shutil
import torch
import numpy as np
from formats import CVRPInstance, RPSolution
from models.runner_utils import NORMED_BENCHMARKS, get_budget_per_size, _adjust_time_limit
# , make_instance_vrptw
# from data.data_utils import prepare_sol_instances
import logging

logger = logging.getLogger(__name__)

EPS = 0.01  # 0.002 changed in cluster b/c of NLNS # np.finfo(np.float32).eps

CVRPLIB_LINKS = {
    "D": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-D.zip", "D"],
    "X": ["vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-X.zip", "X"],
    "Li": ["vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Li.zip", "Li"],
    "Golden": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Golden.zip", "Golden"],
    "XML100": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-XML100.zip", "XML100"]
}

SCALE_FACTORS = {
    "uchoa": 1000,
    "XE": 1000,
    "XML100": 1000,
    "subsampled": 1000,
    "dimacs": 1000,
    "Li": 1000,
    "Golden": 1000
}

CVRP_DEFAULTS = {  # num vehicles and integer capacity per problem size
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}

XE_UCHOA_TYPES = {  # depot type and customer distribution type
    'XE_1': ['R', 'RC', "1-100"],
    'XE_2': ['R', 'C', "Q"],
    'XE_3': ['E', 'RC', "1-10"],
    'XE_4': ['C', 'RC', '50-100'],
    'XE_5': ['R', 'C', 'U'],
    'XE_6': ['R', 'R', '50-100'],
    'XE_7': ['R', 'C', 'Q'],
    'XE_8': ['C', 'RC', '50-100'],
    'XE_9': ['C', 'C', '1-100'],
    'XE_10': ['E', 'R', 'U'],
    'XE_11': ['E', 'R', 'U'],
    'XE_12': ['E', 'R', '1-10'],
    'XE_13': ['C', 'RC', '50-100'],
    'XE_14': ['R', 'C', 'U'],
    'XE_15': ['E', 'R', 'SL'],
    'XE_16': ['C', 'R', '1-100'],
    'XE_17': ['R', 'R', '1-100'],
}


# XE 10    218 E R     U       3
# XE 11    236 E R     U       18
# XE 12    241 E R     1-10    28
# XE 13    269 C RC(5) 50-100  585
# XE 14    274 R C(3)  U       10
# XE 15    279 E R     SL      192
# XE 16    293 C R     1-100   285
# XE 17    297 R R     1-100   55

class CVRPDataset(BaseDataset):
    """Creates VRP data samples to use for training or evaluating benchmark models"""

    def __init__(self,
                 is_train: bool = False,
                 store_path: str = None,
                 dataset_size: int = None,
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
        self.distribution = distribution
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.graph_size = graph_size
        self.grid_size = grid_size if self.distribution != "uchoa" else 1000
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.time_limit = TimeLimit
        self.machine_info = machine_info
        self.re_evaluate = re_evaluate
        print('self.machine_info', machine_info)
        self.metric = None
        self.max_cap_factor = max_cap_factor
        self.transform_func = transform_func
        self.transform_args = transform_args
        self.data_key = None
        self.scale_factor = None
        self.is_denormed = False

        if store_path is not None:
            # load OR download (test) data
            self.data, self.data_key = self.load_dataset()
            assert self.data is not None, f"No data loaded! Please initiate class with valid data path"
            if self.dataset_size is not None and self.dataset_size < len(self.data):
                self.data = self.data[:self.dataset_size]
            logger.info(f"{len(self.data)} Test/Validation Instances for {self.problem} with {self.graph_size} "
                        f"{self.distribution}-distributed customers loaded.")
            # Transform loaded data to CVRPInstance format IF NOT already is in format
            if not isinstance(self.data[0], CVRPInstance):
                self.data = self._make_CVRPInstance()
            if not self.normalize and not self.is_denormed:
                self.data = self._denormalize()
            if self.bks is not None:
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
            else:
                logger.info(f"Data configuration not specified in env config, "
                            f"defaulting to 100 uniformly distributed VRP20 instances")
                self.sample(sample_size=100,
                            graph_size=20,
                            distribution="uniform",
                            log_info=True)

        else:  # no data to load - but initiated CVRPDataset for sampling in training loop
            logger.info(f"No data loaded - initiated CVRPDataset with env config for sampling in training...")
            self.size = None
            self.data = None

    def _download(self, extract_to='.', from_platform='CVRPLIB'):
        """Download CVRPLIB Datasets"""
        url = None
        # default: extract to existing self.store_path
        extract_to = os.path.dirname(self.store_path)
        print('extract_to', extract_to)
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
            print('original BEF', original)
            original = extract_to + "/Vrp-Set-" + os.path.basename(self.store_path) + "/" \
                       + os.path.basename(self.store_path)
            print('original AFT', original)
            #os.makedirs(extract_to)
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

    def _denormalize(self):
        # default is normalized demands and 0-1-normed coordinates for generated data
        # --> denormalize for self.normalize = False and update bks registry in meantime (if given)
        logger.info(f'DE-NORMALIZING data ...')
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
                BKS=self.bks[str(i)][0] if self.bks is not None else None,
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
        print('self.data[0].instance_id, self.data[0].time_limit', self.data[0].instance_id, self.data[0].time_limit)
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

    def _get_costs(self, sol: RPSolution) -> Tuple[float, int, bool, List[list]]:
        # perform problem-specific feasibility check while getting routing costs
        cost, k, solution_upd = self.feasibility_check(sol.instance, sol.solution)
        print('k', k)
        is_feasible = True if cost != float("inf") else False
        return cost, k, is_feasible, solution_upd

    def return_infeasible_sol(self, mode, instance, solution, cost, nr_vs):
        if mode in ['wrap', 'pi'] or 'wrap' in mode or 'pi' in mode:
            logger.info(f"Metric Analysis for instance {instance.instance_id} cannot be performed. No feasible "
                        f"solution provided in Time Limit. Setting PI score to 10 and WRAP score to 1.")
            pi_ = 10 if mode == 'pi' or 'pi' in mode else None
            wrap_ = 1 if mode == 'wrap' or 'wrap' in mode else None
            return solution.update(cost=cost, pi_score=pi_, wrap_score=wrap_, num_vehicles=nr_vs), None, None
        else:
            return solution.update(cost=cost, num_vehicles=nr_vs), None, None

    def eval_costs(self, mode: str, instance: CVRPInstance, v_costs: list, v_times: list, orig_r_times: list,
                   model_name: str):
        return self._eval_metric(model_name=model_name,
                                 inst_id=str(instance.instance_id),
                                 instance=instance,
                                 verified_costs=v_costs,
                                 verified_times=v_times,
                                 run_times_orig=orig_r_times,
                                 eval_type=mode)

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
        cost, nr_v, is_feasible, solution_updated = self._get_costs(solution)
        # print('solution_updated', solution_updated)
        solution = solution.update(solution=solution_updated)
        print('self.scale_factor', self.scale_factor)
        if self.scale_factor is not None:
            cost = cost * self.scale_factor
        if self.is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
            print('IS_DENORMED AND STOREPATH IN NORMED_BENCHMARKS --> RENORMALIZE COSTS FOR EVAL')
            # (re-)normalize costs for evaluation for dataset that is originally normalized
            cost = cost / self.grid_size
        # directly return infeasible solution
        if not is_feasible:
            self.return_infeasible_sol(eval_mode, instance, solution, cost, nr_v)

        # default to simple evaluation if no BKSs are loaded and ensure correct ID order if eval_mode = ['wrap','pi']
        if eval_mode != "simple" and eval_mode != ["simple"]:
            eval_mode = eval_mode if self.bks else "simple"
            if eval_mode == "simple":
                warnings.warn(f"Defaulting to simple evaluation - no BKS loaded for PI or WRAP Evaluation.")
            else:
                assert self.bks[str(solution.instance.instance_id)][0] == \
                       solution.instance.BKS, f"ID mismatch: Instance Tuple BKS does not match " \
                                              f"loaded global BKS repository"

        # get and verify running values
        if solution.running_sols is not None:
            print('Len(running_sols) before VERIFY', len(solution.running_sols))
        if solution.running_costs is not None:
            print('Len(running_costs) before VERIFY', len(solution.running_costs))
        print('Len(running_times) before VERIFY', len(solution.running_times))
        print(f'getting running values for MODEL {model_name}')
        running_costs, running_times, running_sols = self.get_running_values(instance,
                                                                             solution.running_sols,
                                                                             solution.running_costs,
                                                                             solution.running_times,
                                                                             solution.run_time,
                                                                             cost,
                                                                             place_holder_final_sol)

        # if running_sols is not None:
        #     if not running_sols[-1] == solution.solution:
        #         if running_sols[-1] is not None and solution.solution is not None:
        #             if round(cost, 2) < round(running_costs[-1], 2):
        #                 warnings.warn(f"missmatch final solution (with cost: {cost}) and "
        #                               f"last running solution (with cost: {running_costs[-1]})")
        #                 print(f"Updating running solution with better final solution")
                        # print('final runtime', solution.run_time)
                        # print('running_costs', running_costs)
                        # print('running_times', running_times)
        #                 running_costs.append(cost)
        #                 running_times.append(solution.run_time)  # we don't have running time here...
        #                 running_sols.append(solution.solution)

        v_costs, v_times, v_sols, v_costs_full, v_times_full, v_sols_full = self.verify_costs_times(running_costs,
                                                                                                    running_times,
                                                                                                    running_sols,
                                                                                                    instance.time_limit)

        # if running_sols is not None:
        #     if not v_sols[-1] == solution.solution:
        #         if v_sols[-1] is not None and solution.solution is not None:
        #             if round(cost, 2) > round(v_costs[-1], 2):
        #                 warnings.warn(f"missmatch final solution (with cost: {cost}) and "
        #                               f"last running solution (with cost: {running_costs[-1]})")
        #                 print(f"Updating final solution with better running cost solution")
        #                 solution = solution.update(solution=running_sols[-1])
        #                 cost = v_costs[-1]

        print('Len(verified_costs) AFTER VERIFY', len(v_costs))
        print('Len(verified_times) AFTER VERIFY', len(v_times))
        print('Len(verified_sols) AFTER VERIFY', len(v_sols))
        # print('verified_times in cvrp_dataset.eval_solution', verified_times)
        # print('len(verified_costs)', len(verified_costs))
        # print('len(v_costs_full)', len(v_costs_full))
        # print('len(v_times_full)', len(v_times_full))
        # print('verified_costs', verified_costs)
        # print('verified_times', verified_times)

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
                               num_vehicles=nr_v,
                               running_costs=v_costs if v_costs and v_costs is not None else None,
                               running_times=v_times if v_times and v_times is not None else None,
                               running_sols=v_sols if v_sols and any(v_sols) else None,
                               run_time=solution.run_time,  # self.adjusted_time_limit),
                               pi_score=pi_score,
                               wrap_score=wrap_score), None, new_best

    def update_BKS(self, instance, cost):
        # self.bks[str(instance.instance_id)] = cost
        logger.info(f"New BKS found for instance {instance.instance_id} of the {self.distribution}-"
                    f"distributed {self.problem} Test Set")
        logger.info(f"New BKS with cost {cost} is {instance.BKS - cost} better than old BKS with cost {instance.BKS}")

        return str(instance.instance_id)

    def get_running_values(self,
                           instance: CVRPInstance,
                           running_sol: List[List[List]],
                           running_costs: List[float],
                           running_t: List[float],
                           final_runtime: float,
                           final_cost: float,
                           place_holder_final_sol: bool = False,
                           update_runn_sols: bool = True):
        runn_costs_upd, runn_sols = None, None
        if running_sol is not None and running_t is not None:
            runn_costs = [self.feasibility_check(instance, sol, is_running=True)[0] for sol in running_sol]
            if update_runn_sols and (len(runn_costs) != len(running_t)):
                assert len(runn_costs) == len(running_sol), f"Cannot update running sols - not same length with costs"
                prev_cost = float('inf')
                runn_sols, runn_costs_upd = [], []
                for cost, sol in zip(runn_costs, running_sol):
                    if cost < prev_cost and cost != float('inf'):
                        runn_costs_upd.append(cost)
                        runn_sols.append(sol)
                        prev_cost = cost
                print(f"len runn_cost_upd {len(runn_costs_upd)}, len runn_sols {len(runn_sols)}, len running_t {len(running_t)}")
                runn_costs = runn_costs_upd
                # if len(runn_costs) > len(running_t):
                #     runn_costs.pop()
                #     runn_sols.pop()
                # assert len(runn_costs) == len(runn_sols) == len(running_t)
                # runn_costs = runn_costs_upd
            if self.scale_factor is not None:
                # print('scaling running COSTS with self.scale_factor', self.scale_factor)
                runn_costs = [c * self.scale_factor for c in runn_costs]
            elif self.is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
                runn_costs = [c / self.grid_size for c in runn_costs]
            else:
                runn_costs = runn_costs
            runn_times = running_t
        elif running_costs is not None and running_t is not None:
            warnings.warn(f"Getting objective costs directly from solver - feasibility not checked by BaseDataset")
            if self.scale_factor is not None:
                runn_costs = [c * self.scale_factor for c in running_costs]
            elif self.is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
                runn_costs = [c / self.grid_size for c in running_costs]
            else:
                runn_costs = running_costs
            runn_times = running_t
            print('FINAL COST:', final_cost)
            print("final_cost == float('inf')", final_cost == float('inf'))
            if final_cost is not None and runn_costs:
                print('np.round(final_cost, 1)', np.round(final_cost, 1))
                print('np.round(runn_costs[-1], 1)', np.round(runn_costs[-1], 1))
                if np.round(final_cost, 2) != np.round(runn_costs[-1], 2):
                    if final_cost != float('inf') and final_cost > runn_costs[-1]:
                        # np.round(
                        warnings.warn(f"Last running cost {runn_costs[-1]} is smaller than calculated final"
                                      f" cost {final_cost}. Removing running costs < final costs, because don't have"
                                      f" solution for this cost.")
                        runn_costs, runn_times = [], []
                        for r_cost, r_time in zip(running_costs, running_t):
                            if r_cost > final_cost:
                                runn_costs.append(r_cost)
                                runn_times.append(r_time)
                            elif r_cost < final_cost:
                                print('np.round(r_cost)', np.round(r_cost))
                                print('np.round(final_cost)', np.round(final_cost))
                                runn_costs.append(final_cost)
                                runn_times.append(r_time)
                                break
                        print('runn_costs', runn_costs)
                        print('runn_times', runn_times)
                        print('final_cost', final_cost)
                        print('final_runtime', final_runtime)
                    elif final_cost == float('inf'):
                        if place_holder_final_sol:
                            print(f"Is placeholder final solution in Re-evaluation...")
                    else:
                        print('np.round(final_cost, 1)', np.round(final_cost, 1))
                        print('np.round(runn_costs[-1], 1)', np.round(runn_costs[-1], 1))
                        if np.round(final_cost, 1) < np.round(runn_costs[-1], 1):
                            warnings.warn(f"Last running cost {runn_costs[-1]} is larger than calculated final"
                                          f" cost {final_cost}. Adding final costs to running costs.")
                            runn_costs.append(final_cost)
                            runn_times.append(final_runtime)
                        else:
                            # is rounding precision error --> replace better final cost in runn_costs
                            runn_costs[-1] = final_cost

        else:
            if self.scale_factor is not None:
                runn_costs = [final_cost * self.scale_factor]
            # elif self.is_denormed: --> FINAL COST ALREADY NORMALIZED
            #     runn_costs = [final_cost / self.grid_size]
            else:
                runn_costs = [final_cost]
            runn_times = [final_runtime]
        print('runn_costs[:3]', runn_costs[:3])
        # runn_costs = runn_costs_upd if runn_costs_upd is not None else runn_costs
        runn_sols = runn_sols if runn_sols is not None else running_sol
        return runn_costs, runn_times, runn_sols

    def feasibility_check(self, instance: CVRPInstance, solution: List[List], is_running: bool = False):
        depot = instance.depot_idx[0]
        coords = instance.coords.astype(int) if self.is_denormed and isinstance(instance.coords[0][0], np.int64) \
            else instance.coords
        # if self.scale_factor is None else (instance.coords * self.scale_factor).astype(int)
        demands = instance.node_features[:, instance.constraint_idx[0]] if self.is_denormed \
            else instance.node_features[:, instance.constraint_idx[0]]
        # demands = np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity)
        # print('demands[:10]', demands[:10])
        # * instance.original_capacity).astype(int)
        # np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity, 3).astype(int)
        routes = solution if solution else None  # check if solution list is not empty - if empty set to None
        # capacity = instance.original_capacity
        capacity = instance.original_capacity if self.is_denormed else instance.vehicle_capacity
        # print('capacity', capacity)
        routes_ = []
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
                    if cum_d > capacity + EPS:
                        if is_running:
                            warnings.warn(f"One of the solutions in the trajectory for instance {instance.instance_id} "
                                          f"is infeasible: {cum_d}>{capacity + EPS}. Setting cost and k to 'inf'.")

                        else:
                            warnings.warn(f"Final CVRP solution {solution} is infeasible for instance "
                                          f"with ID {instance.instance_id}. Setting cost and k to 'inf'.")
                            warnings.warn(f"cumulative demand {cum_d} surpasses (normalized) capacity "
                                          f"{capacity} for instance with ID {instance.instance_id}.")
                        cost = float("inf")
                        k = float("inf")
                        break
                    cost += transit
                    k += 1
                    routes_.append(r)
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
        self.scale_factor = SCALE_FACTORS[overall_inst_type] if overall_inst_type in SCALE_FACTORS.keys() else 1
        # print('self.scale_factor', self.scale_factor)
        while i < len(lines):
            line = lines[i]
            if line.startswith("NAME"):
                name = line.split(':')[1].strip()
                if os.path.dirname(filepath).split(os.sep)[-2] == "XE":
                    X_inst_type = os.path.dirname(filepath).split(os.sep)[-1]
                    inst_id = name.split('_')[-1]
                # elif os.path.dirname(filepath).split(os.sep)[-2] == "X":
                #     X_inst_type = 'X'
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
            coords_dist=XE_UCHOA_TYPES[X_inst_type][1] if X_inst_type is not None else None,
            depot_type=XE_UCHOA_TYPES[X_inst_type][0] if X_inst_type is not None else None,
            demands_dist=XE_UCHOA_TYPES[X_inst_type][2] if X_inst_type is not None else None,
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
