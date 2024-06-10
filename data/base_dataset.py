import torch
import numpy as np
from abc import abstractmethod
from omegaconf import DictConfig, ListConfig
from typing import Union, Tuple, List, Callable
import warnings
import os
import pickle
import logging
from pathlib import Path

from torch.utils.data import Dataset
from formats import TSPInstance, CVRPInstance, CVRPTWInstance, RPSolution
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
    'uniform_fu': 'uniform_fu',
    'nazari': 'uniform',
    'rej': 'rejection_sampled',
    'uchoa': 'uchoa_distributed',
    'tsplib': 'tsplib_format',
    'homberger': 'homberger_200',
    'XE': 'XE',
    'S': 'S'
}

TEST_SETS_BKS = ['tsp100_fu.pt',
                 'tsp200_fu.pt',
                 'tsp500_fu.pt',
                 'tsp1000_fu.pt',
                 'tsp10000_fu.pt',
                 'cvrp20_test_seed1234.pkl',
                 'cvrp50_test_seed1234.pkl',
                 'cvrp100_test_seed1234.pkl',
                 'val_seed123_size512.pt',
                 'val_seed123_size512.pkl',
                 'val_seed4321_size512.pkl',
                 'val_seed1234_size128.pkl',
                 'val_seed1234_size128.pt',
                 'E_R_6_seed123_size512.pt',
                 'val_seed4321_size128.pt',
                 'val_cvrptw_40.pkl',
                 'val_cvrptw_200.pkl',
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
                 generator_args: Union[dict, DictConfig] = None,
                 sampling_args: Union[dict, DictConfig] = None,
                 float_prec: np.dtype = np.float32,
                 transform_func: Callable = None,
                 transform_args: DictConfig = None,
                 seed: int = None,
                 verbose: bool = False,
                 normalize: bool = True,
                 grid_size: int = None,
                 scale_factor=None,
                 is_denormed=False,
                 TimeLimit: Union[list, int, float] = None,
                 load_bks: bool = True,
                 load_base_sol: bool = True,
                 **kwargs):
        super(BaseDataset, self).__init__()

        if store_path is not None:
            logger.info(f"Test Data provided: {store_path}, No new samples are generated.")

        self.verbose = verbose
        self.problem = problem
        self.store_path = store_path
        self.num_samples = num_samples
        self.graph_size = graph_size
        self.distribution = distribution
        self.generator_args = generator_args
        self.sampling_args = sampling_args
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.grid_size = grid_size
        self.is_denormed = is_denormed
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
            print('seed', seed)
            if seed is None:
                seed = 1234
                logger.info(f"Set default seed for RPGenerator with {seed}")
            self.gen = RPGenerator(seed, self.verbose, float_prec, self.generator_args)
        self.size = None
        self.data = None
        self.data_transformed = None
        self.data_key = None
        self.bks_path = None
        self.base_sol_path = None
        self.bks, self.BaseSol = None, None
        self.load_bks = load_bks
        self.load_base_sol = load_base_sol
        if self.load_bks:
            self.bks = self.load_BKS_BaseSol("BKS") if self.store_path is not None else None
        if self.load_base_sol:
            self.BaseSol = self.load_BKS_BaseSol("BaseSol") if self.store_path is not None else None
        if self.bks is not None:
            logger.info(f'Loaded {len(self.bks)} BKS for the test (val) set.')
        elif self.bks is None and self.store_path:
            logger.info(f'No BKS loaded for dataset {self.store_path}.')
        self.metric = None  # gets initialized in runner if cfg.eval_type != "simple"
        self.adjusted_time_limit = None  # gets initialized in runner if cfg.eval_type != "simple"

    def seed(self, seed: int):
        self.gen.seed(seed)

    def sample(self, sample_size: int, graph_size: int = None, distribution=None, log_info=True, sub_samples=False):
        if distribution is None:
            distribution = self.distribution if self.distribution is not None else None
        if graph_size is None:
            graph_size = self.graph_size if self.graph_size is not None else None
        if log_info:
            logger.info(f"Sampling {sample_size} {distribution}-distributed problems with graph size {graph_size}")
        # print('sub_samples', sub_samples)
        sub_samples = self.sampling_args.subsample
        # print('sub_samples', sub_samples)
        if not sub_samples:
            self.data, demands_normalized = self.gen.generate(problem=self.problem,
                                                              sample_size=sample_size,
                                                              graph_size=graph_size,
                                                              # distribution=distribution,
                                                              normalize=self.normalize,
                                                              # generator_args=self.generator_args
                                                              sampling_args=self.sampling_args)
            if demands_normalized is not None:
                self.is_denormed = not demands_normalized
        else:
            self.data = self.gen.generate_subsamples(problem=self.problem,
                                                     sample_size=sample_size,
                                                     graph_size=graph_size,
                                                     distribution=distribution,
                                                     normalize=self.normalize,
                                                     # generator_args=self.generator_args
                                                     sampling_args=self.sampling_args)
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

    def _get_costs(self, sol: RPSolution,
                   is_runn: bool = False) -> Tuple[float, Union[int, None], bool, Union[list, None]]:
        # perform problem-specific feasibility check while getting routing costs
        cost, k, solution_list_upd = self.feasibility_check(instance=sol.instance, rp_solution=sol, is_running=is_runn)
        is_feasible = True if cost != float("inf") else False
        return cost, k, is_feasible, solution_list_upd

    def eval_costs(self, mode: str, instance: Union[TSPInstance, CVRPInstance], v_costs: list, v_times: list,
                   orig_r_times: list, model_name: str):
        return self._eval_metric(model_name=model_name,
                                 inst_id=str(instance.instance_id),
                                 instance=instance,
                                 verified_costs=v_costs,
                                 verified_times=v_times,
                                 run_times_orig=orig_r_times,
                                 eval_type=mode)

    # @staticmethod
    def verify_costs_times(self, costs, times, sols, time_limit=None):
        verified_costs, verified_times, verified_sols = [], [], []
        verified_costs_full, verified_times_full, verified_sols_full = [], [], []
        prev_cost = float('inf')
        time_limit = self.adjusted_time_limit if self.adjusted_time_limit is not None else times[-1]  # if 'simple' eval
        # and time <= adjusted_time_limit
        sols = [None] * len(costs) if sols is None else sols
        if times != [None]:
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
        # directly return infeasible solution
        if not is_feasible:
            self.return_infeasible_sol(eval_mode, instance, solution, cost, nr_v)
        # print('solution_updated', solution_updated)
        solution = solution.update(solution=solution_updated)
        print('self.scale_factor', self.scale_factor)
        if self.scale_factor is not None:
            cost = cost * self.scale_factor
        if self.is_denormed and self.store_path is None:
            pass
        elif self.is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
            if self.verbose:
                logger.info(f'Dataset is de-normalized for run '
                            f'--> Re-Normalize costs for benchmark evaluation with grid-size {self.grid_size}')
            # (re-)normalize costs for evaluation for dataset that is originally normalized
            cost = cost / self.grid_size

        # default to simple evaluation if no BKS loaded or incorrect ID order
        eval_mode = self.check_eval_mode(eval_mode, solution)

        # get and verify running values
        # if self.verbose:
        #     if solution.running_sols is not None:
        #         print('Len(running_sols) before VERIFY', len(solution.running_sols))
        #     if solution.running_costs is not None:
        #         print('Len(running_costs) before VERIFY', len(solution.running_costs))
        #         print('Len(running_times) before VERIFY', len(solution.running_times))
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
                               num_vehicles=nr_v,
                               running_costs=v_costs if v_costs and v_costs is not None else None,
                               running_times=v_times if v_times and v_times is not None else None,
                               running_sols=v_sols if v_sols and any(v_sols) else None,
                               run_time=solution.run_time,  # self.adjusted_time_limit),
                               pi_score=pi_score,
                               wrap_score=wrap_score), None, new_best

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
                logger.info("Loading instance files...")
                if self.problem.lower() == "cvrp":
                    data = [self.read_vrp_instance(self.store_path + "/" + file) for file in os.listdir(self.store_path)
                            if (file[:3] != 'BKS' and file[-3:] == 'vrp')]
                else:
                    data = [self.read_tsp_instance(self.store_path + "/" + file) for file in os.listdir(self.store_path)
                            if (file[:3] != 'BKS' and file[-3:] == 'tsp')]
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

    def get_running_values(self,
                           instance: Union[TSPInstance, CVRPInstance],
                           running_sol: List[List[List]],
                           running_costs: List[float],
                           running_t: List[float],
                           final_runtime: float,
                           final_cost: float,
                           scale_factor: int,
                           grid_size: int,
                           is_denormed: bool,
                           place_holder_final_sol: bool = False,
                           update_runn_sols: bool = True):
        runn_costs_upd, runn_sols = None, None
        # print('running_sol', running_sol)
        # print('running_costs', running_costs)
        print('scale_factor', scale_factor)
        print('grid_size', grid_size)
        if running_sol is not None and running_t is not None:
            runn_costs = [self.feasibility_check(instance=instance, rp_solution=sol, is_running=True)[0]
                          for sol in running_sol]
            if update_runn_sols and (len(runn_costs) != len(running_t)):
                assert len(runn_costs) == len(running_sol), f"Cannot update running sols - not same length with costs"
                prev_cost = float('inf')
                runn_sols, runn_costs_upd = [], []
                for cost, sol in zip(runn_costs, running_sol):
                    if cost < prev_cost and cost != float('inf'):
                        runn_costs_upd.append(cost)
                        runn_sols.append(sol)
                        prev_cost = cost
                print(
                    f"len runn_cost_upd {len(runn_costs_upd)}, len runn_sols {len(runn_sols)}, len running_t {len(running_t)}")
                runn_costs = runn_costs_upd
                # if len(runn_costs) > len(running_t):
                #     runn_costs.pop()
                #     runn_sols.pop()
                # assert len(runn_costs) == len(runn_sols) == len(running_t)
                # runn_costs = runn_costs_upd
            if scale_factor is not None:
                # print('scaling running COSTS with self.scale_factor', self.scale_factor)
                runn_costs = [c * scale_factor for c in runn_costs]
            elif is_denormed and self.store_path is not None and \
                    os.path.basename(self.store_path) in NORMED_BENCHMARKS:
                runn_costs = [c / grid_size for c in runn_costs]
            else:
                runn_costs = runn_costs
            runn_times = running_t
        elif running_costs is not None and running_t is not None:
            warnings.warn(f"Getting objective costs directly from solver - feasibility not checked by BaseDataset")
            if scale_factor is not None:
                runn_costs = [c * scale_factor for c in running_costs]
            elif is_denormed and os.path.basename(self.store_path) in NORMED_BENCHMARKS:
                runn_costs = [c / grid_size for c in running_costs]
            else:
                runn_costs = running_costs
            runn_times = running_t
            # print('FINAL COST:', final_cost)
            # print("final_cost == float('inf')", final_cost == float('inf'))
            if final_cost is not None and runn_costs:
                # print('np.round(final_cost, 1)', np.round(final_cost, 1))
                # print('np.round(runn_costs[-1], 1)', np.round(runn_costs[-1], 1))
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
                                # print('np.round(r_cost)', np.round(r_cost))
                                # print('np.round(final_cost)', np.round(final_cost))
                                runn_costs.append(final_cost)
                                runn_times.append(r_time)
                                break
                        # print('runn_costs', runn_costs)
                        # print('runn_times', runn_times)
                        # print('final_cost', final_cost)
                        # print('final_runtime', final_runtime)
                    elif final_cost == float('inf'):
                        if place_holder_final_sol:
                            print(f"Is placeholder final solution in Re-evaluation...")
                    else:
                        # print('np.round(final_cost, 1)', np.round(final_cost, 1))
                        # print('np.round(runn_costs[-1], 1)', np.round(runn_costs[-1], 1))
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
        # print('runn_costs[:3]', runn_costs[:3])
        # runn_costs = runn_costs_upd if runn_costs_upd is not None else runn_costs
        runn_sols = runn_sols if runn_sols is not None else running_sol

        return self.verify_costs_times(runn_costs, runn_times, runn_sols, instance.time_limit), runn_times

    def update_BKS(self, instance, cost):
        # self.bks[str(instance.instance_id)] = cost
        logger.info(f"New BKS found for instance {instance.instance_id} of the {self.distribution}-"
                    f"distributed {self.problem} Test Set")
        logger.info(f"New BKS with cost {cost} is {instance.BKS - cost} better than old BKS with cost {instance.BKS}")

        return str(instance.instance_id)

    @staticmethod
    def return_infeasible_sol(mode, instance, solution, cost, nr_vs=None):
        if mode in ['wrap', 'pi'] or 'wrap' in mode or 'pi' in mode:
            logger.info(f"Metric Analysis for instance {instance.instance_id} cannot be performed. No feasible "
                        f"solution provided in Time Limit. Setting PI score to 10 and WRAP score to 1.")
            pi_ = 10 if mode == 'pi' or 'pi' in mode else None
            wrap_ = 1 if mode == 'wrap' or 'wrap' in mode else None
            return solution.update(cost=cost, pi_score=pi_, wrap_score=wrap_, num_vehicles=nr_vs), None, None
        else:
            return solution.update(cost=cost, num_vehicles=nr_vs), None, None

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
            print('os.path.basename(self.store_path)', os.path.basename(self.store_path))
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
                elif Path(os.path.join(self.store_path,
                                       item_to_load + "_" + self.store_path.split("/")[-3] + ".pkl")).exists():
                    load_name = item_to_load + "_" + self.store_path.split("/")[-3] + ".pkl"
                    # print('os.path.join(self.store_path, load_name)', os.path.join(self.store_path, load_name))
                    _path = os.path.join(self.store_path, load_name)
                elif Path(os.path.join(os.path.dirname(self.store_path),
                                       item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + os.path.basename(self.store_path)[:5] + ".pkl"
                    _path = os.path.join(os.path.dirname(self.store_path), load_name)
                elif Path(os.path.join(os.path.dirname(self.store_path),
                                       item_to_load + "_" + os.path.basename(self.store_path)[:-3] + ".pkl")).exists():
                    # load BKS for val data
                    load_name = item_to_load + "_" + os.path.basename(self.store_path)[:-3] + ".pkl"
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
                elif Path(os.path.join(os.path.dirname(self.store_path),
                                       item_to_load + "_" + os.path.basename(self.store_path))):
                    load_name = item_to_load + "_" + os.path.basename(self.store_path)
                    _path = os.path.join(os.path.dirname(self.store_path), load_name)
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
                    try:
                        base_sols = torch.load(self.base_sol_path)
                    except FileNotFoundError:
                        base_sols = None
                    if base_sols is None:
                        logger.info(f'Tried to load Base Solutions, but do not exists... No WRAP eval. possible.')
                        self.load_base_sol = False
                    return base_sols
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

    def check_eval_mode(self, eval_mode: str, solution: RPSolution):
        if eval_mode != "simple" and eval_mode != ["simple"]:
            eval_mode = eval_mode if self.bks else "simple"
            if eval_mode == "simple":
                warnings.warn(f"Defaulting to simple evaluation - no BKS loaded for PI or WRAP Evaluation.")
            elif not self.bks[str(solution.instance.instance_id)][0] == solution.instance.BKS:
                warnings.warn(f"ID mismatch: Instance Tuple BKS does not match loaded global BKS repository."
                              f" Defaulting to simple evaluation")
                eval_mode = "simple"
            return eval_mode
        else:
            return eval_mode

    # @staticmethod
    def save_trajectory(self, instance_id, costs, times, model, save_trajectory_for=None, instance=None,
                        save_name=None):
        if save_trajectory_for is None and self.save_traj_flag:
            save_trajectory_for = instance_id
            self.save_traj_flag = False
        if instance is not None:
            if self.problem == "cvrp":
                save_name = "_instance_" + instance_id + "_c_dist_" + str(instance.coords_dist) + "_d_dist_" \
                            + str(instance.demands_dist) \
                            + "_depot_type_" + str(instance.depot_type)
            # elif self.problem == "cvrptw":
            else:
                save_name = "_instance_" + instance_id + "_c_dist_" + str(instance.coords_dist) + "_depot_type_" \
                            + str(instance.depot_type)
        if instance_id == str(save_trajectory_for):
            logger.info(f'Saving solution trajectory for instance {instance_id}')
            if instance is not None:
                self.plot_trajectory(instance_id, model, times, costs, save_name, instance.time_limit)
            else:
                self.plot_trajectory(instance_id, model, times, costs)
        elif isinstance(save_trajectory_for, List) and int(instance_id) in save_trajectory_for:
            logger.info(f'Saving solution trajectory for instance {instance_id}')
            if instance is not None:
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
    def read_tsp_instance(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def _denormalize(self):
        raise NotImplementedError

    @abstractmethod
    def _download(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def feasibility_check(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
