#
import numpy
import os
import sys

import logging
import time
import itertools as it
from typing import Union, Optional, Dict, List, Tuple, Any, Type
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from formats import TSPInstance, CVRPInstance, RPSolution

from .EAS.source.active_search import run_active_search
from .EAS.source.cvrp.grouped_actors import ACTOR as CVRP_ACTOR
from .EAS.source.cvrp.read_data import read_instance_pkl as CVRP_read_instance_pkl
from .EAS.source.cvrp.read_data import read_instance_vrp
from .EAS.source.cvrp.utilities import augment_and_repeat_episode_data as CVRP__augment_and_repeat_episode_data
from .EAS.source.cvrp.utilities import get_episode_data as CVRP_get_episode_data
from .EAS.source.eas_emb import run_eas_emb
from .EAS.source.eas_lay import run_eas_lay
from .EAS.source.eas_tab import run_eas_tab
from .EAS.source.sampling import run_sampling
from .EAS.source.tsp.grouped_actors import ACTOR as TSP_ACTOR
from .EAS.source.tsp.read_data import read_instance_pkl as TSP_read_instance_pkl
from .EAS.source.tsp.utilities import augment_and_repeat_episode_data as TSP_augment_and_repeat_episode_data
from .EAS.source.tsp.utilities import get_episode_data as TSP_get_episode_data

logger = logging.getLogger(__name__)


def eval_model(method: str,
               grouped_actor: Union[TSP_ACTOR, CVRP_ACTOR],
               data: List[CVRPInstance],
               config: Union[Dict, DictConfig],
               problem: str,
               device=torch.device("cpu"),
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    eas_method = "eas-" + method if method not in ["sampling", "as"] else method
    print('eas_method', eas_method)
    print('config["max_runtime"]', config["max_runtime"])
    print('len(set([inst.graph_size for inst in data]))', len(set([inst.graph_size for inst in data])))
    if len(set([inst.graph_size for inst in data])) != 1:
        logger.error("Enforcing batch_size=1 .. different sized problems in set...")
        config["batch_size"] = 1
    # method prep
    if eas_method == "sampling":
        start_search_fn = run_sampling
    elif eas_method.startswith("as"):
        start_search_fn = run_active_search
    elif eas_method.startswith("eas-emb"):
        start_search_fn = run_eas_emb
    elif eas_method.startswith("eas-lay"):
        start_search_fn = run_eas_lay
    elif eas_method.startswith("eas-tab"):
        start_search_fn = run_eas_tab
    else:
        raise NotImplementedError("Unknown EAS-search method")

    if config.batch_size == 1:
        logging.info("Starting single instance search. 1 instance is solved per episode.")
    else:
        assert config.p_runs == 1
        logging.info(f"Starting batch search. {config.batch_size} instances are solved per episode.")

    if problem == "TSP":
        config["problem"] = "TSP"
        get_episode_data_fn = TSP_get_episode_data
        augment_and_repeat_episode_data_fn = TSP_augment_and_repeat_episode_data
    elif problem == "CVRP":
        config["problem"] = "CVRP"
        get_episode_data_fn = CVRP_get_episode_data
        augment_and_repeat_episode_data_fn = CVRP__augment_and_repeat_episode_data
    else:
        get_episode_data_fn = None
        augment_and_repeat_episode_data_fn = None

    sols, runtimes, costs, costs_aug = [], [], [], []
    # Run the actual search
    if len(set([inst.graph_size for inst in data])) == 1:
        # data prep
        instance_data_scaled, problem_size = prep_data(problem, dat=data, offset=config.instances_offset)
        # scale down time-limit factor - because over-budgeting
        config['max_runtime'] = config.max_runtime  # - 1
        problem_size_list = [problem_size]*len(data)
        start_t = time.time()
        perf, best_solutions, running_sols, running_times = start_search_fn(grouped_actor, instance_data_scaled,
                                                                            problem_size, config, get_episode_data_fn,
                                                                            augment_and_repeat_episode_data_fn)
        total_runtime = time.time() - start_t
        total_runtimes = [total_runtime / len(data)] * len(data)
    else:
        perf, best_solutions, running_sols, running_times, problem_size_list = [], [], [], [], []
        total_runtime = 0
        for i in range(len(data)):
            # data prep
            instance_data_scaled, problem_size = prep_data(problem, dat=[data[i]], offset=config.instances_offset)
            print('data[i].time_limit', data[i].time_limit)
            config['max_runtime'] = int(data[i].time_limit)
            print(f"i {i}: config['max_runtime']: {config['max_runtime']}")
            start_t_i = time.time()
            # try:
            perf_, best_solutions_, running_sols_, running_times_ = start_search_fn(grouped_actor,
                                                                                    instance_data_scaled,
                                                                                    problem_size, config,
                                                                                    get_episode_data_fn,
                                                                                    augment_and_repeat_episode_data_fn)
            end_t_i = time.time()
            total_runtime += (end_t_i - start_t_i)
            perf.append(perf_)
            best_solutions.append(best_solutions_)
            running_sols.append(running_sols_)
            running_times.append(running_times_)
            problem_size_list.append(problem_size)
            torch.cuda.empty_cache()
        best_solutions = torch.stack(best_solutions)
        total_runtimes = [total_runtime / len(data)] * len(data)
    logger.info(f"Mean costs: {np.mean(perf)}")
    logger.info(f"Runtime: {total_runtimes}")
    logger.info(f"Nb. instances: {len(perf)}")
    # print('len(running_times)', len(running_times))
    # print('running_times', running_times)
    # print('len(running_sols[0])', len(running_sols[0]))
    # print('best_solutions[0]', best_solutions[0])
    s_parsed_final = _get_sep_tours(problem, best_solutions, problem_sizes=problem_size_list)
    # print('s_parsed_final', s_parsed_final)
    s_parsed_running = [_get_sep_tours(problem, sol, problem_sizes=problem_size_list) for sol in running_sols]
    # print('len(s_parsed_running)', len(s_parsed_running))
    # print('s_parsed_running[0]', s_parsed_running[0])
    solutions = make_RPSolution(problem, perf, s_parsed_final, s_parsed_running, running_times, total_runtimes, data)
    res = {}
    #     "reward_mean": np.mean(rews),
    #     "reward_std": np.std(rews),
    # }

    return res, solutions


def train_model(Trainer,
                env,
                model,
                optimizer: Optimizer,
                scheduler: Scheduler,
                device: torch.device = torch.device("cpu"),
                train_cfg: Optional[Dict] = None):
    # note: Train data from TSPDataset/CVRPDataset class generated on the fly in POMO Env

    # USE_CUDA = True if device != torch.device("cpu") and not DEBUG_MODE else False
    # print("use_cuda", USE_CUDA)
    trainer = Trainer(env=env,
                      generate_problems=True,
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      trainer_params=train_cfg,
                      USE_CUDA=True)

    # if DEBUG_MODE:
    #     _set_debug_mode()

    # copy_all_src(trainer.result_folder)

    trainer.run()

    return None


def _get_sep_tours(problem: str, sols: torch.Tensor, problem_sizes: list) -> List[List]:
    """get solution (res) as List[List]"""
    # parse solution
    # print('problem', problem)
    if problem.lower() == 'tsp':
        # if problem is tsp - only have single tour
        # for sol_ in sols:
        # sol tensors are of size (len(dat), problem_size*2)  - still have 0s
        return [sol_.tolist()[:problem_size] for sol_, problem_size in zip(sols, problem_sizes)]

    elif problem.lower() == 'cvrp':
        sols_ = []
        # for sol_batch in sols:
        #     print('sol_batch', sol_batch)
        # print('sols', sols)
        for sol in sols:
            sol_lst = sol_to_list(sol, depot_idx=0)
            sols_.append(sol_lst)
        # print(f'sols_: {sols_}')
        return sols_


# Transform solution returned from POMO to List[List]
def sol_to_list(sol: Union[torch.tensor, np.ndarray], depot_idx: int = 0) -> List[List]:
    sol_lst, lst = [], [0]
    for e in sol:
        if e == depot_idx:
            if len(lst) > 1:
                lst.append(0)
                sol_lst.append(lst)
                lst = [0]
        else:
            if isinstance(e, Tensor):
                lst.append(e.item())
            else:
                lst.append(e)
    # print(f'sol_lst: {sol_lst}')
    return sol_lst


def make_RPSolution(problem, rews, s_parsed, s_running, t_running, total_t, data) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    # sol_list = [_get_sep_tours(problem, sol_) for sol_ in sols]

    return [
        RPSolution(
            solution=sol,
            cost=r,
            num_vehicles=len(sol) if problem.upper() == 'CVRP' else len([sol]),
            run_time=t,  # float(t[:-1]),
            running_sols=runn_sol,
            running_times=runn_t,
            problem=problem,
            instance=inst,
            method_internal_cost=r
        )
        for sol, runn_sol, r, runn_t, t, inst in zip(s_parsed, s_running, rews, t_running, total_t, data)
    ]


def make_cvrp_instance(instance: CVRPInstance):
    coords = instance.coords
    demand = instance.node_features[1:, instance.constraint_idx[0]]
    return coords, demand


def prep_data(problem: str, dat: Union[List[TSPInstance], List[CVRPInstance]], offset=0):
    """preprocesses data format for EAS (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    if problem.lower() == "tsp" and isinstance(dat[0], TSPInstance):
        instance_data = np.array([row.coords for row in (dat[offset:offset + len(dat)])])
        problem_size = instance_data.shape[1]
        instance_data_scaled = (instance_data, None)
    elif problem.lower() == "cvrp" and isinstance(dat[0], CVRPInstance):
        # print('dat[0].type.upper()', dat[0].type.upper())
        if dat[0].type.upper() not in ["X", "XE", "DIMACS", "GOLDEN", "XML100"]:
            coords = [instance.coords for instance in dat[offset:offset + len(dat)]]
            demands = [instance.node_features[1:, instance.constraint_idx[0]] for instance in
                       dat[offset:offset + len(dat)]]
            instance_data = np.stack(coords), np.stack(demands)
            problem_size = instance_data[0].shape[1] - 1
            demand_scaler = 1.0  # dat[0].original_capacity  --> already normalized
            instance_data_scaled = instance_data[0], instance_data[1] / demand_scaler
        else:
            # print("dat[0].original_locations[:5]", dat[0].original_locations[:5])
            # print("dat[0].coords[:5]", dat[0].coords[:5])
            # print("dat[0].node_features[1:, instance.constraint_idx[0]][:5]",
            #       dat[0].node_features[1:, dat[0].constraint_idx[0]][:5])

            # Prepare empty numpy array to store instance data
            instance_data_scaled = (np.zeros((len(dat), dat[0].coords.shape[0], 2)),
                                    np.zeros((len(dat), dat[0].coords.shape[0] - 1)))

            problem_size = dat[0].coords.shape[0] - 1

            # Read in all instances
            for i, instance in enumerate(dat):
                # logging.info(f'Instance: {os.path.split(file)[-1]}')
                # original_locations, locations, demand, capacity = read_instance_vrp(file)
                # instance_data_scaled[0][i], instance_data_scaled[1][i] = locations, demand / capacity
                instance_data_scaled[0][i], instance_data_scaled[1][i] = instance.coords, instance.node_features[1:,
                                                                                          dat[0].constraint_idx[0]]
    else:
        raise NotImplementedError

    # print('instance_data_scaled[0].shape', instance_data_scaled[0].shape)
    # print('instance_data_scaled[1].shape', instance_data_scaled[1].shape)
    return instance_data_scaled, problem_size
