import sys
import warnings

sys.path.insert(0, '/home/thyssens/git/Research/L2O/routing-arena/models/NLNS/NLNS')

import logging
import time
# import itertools as it
from typing import Union, NamedTuple, Dict, List, Tuple, Any
from omegaconf import DictConfig
# import datetime
import numpy as np
import multiprocessing as mp
import torch
import os
import math
# from torch import Tensor
# from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, Manager
from copy import deepcopy
from typing import Optional, Dict, Union, NamedTuple, List, Tuple

from formats import CVRPInstance, RPSolution
from data import CVRPDataset
from models.NLNS.NLNS.vrp.data_utils import *
from models.NLNS.NLNS.vrp.data_utils import read_instances_pkl, read_instance
from models.NLNS.NLNS import search, search_batch, repair
from models.NLNS.NLNS.vrp.vrp_problem import VRPInstance

# train imports
from models.NLNS.NLNS import train
from models.NLNS.NLNS.actor import VrpActorModel
from models.NLNS.NLNS.critic import VrpCriticModel

logger = logging.getLogger(__name__)

EMA_ALPHA = 0.2


def prep_data_NLNS(dat: List[CVRPInstance],
                   is_denormed=True,
                   mode: str = 'eval_batch',
                   data_type: str = 'uniform',
                   offset: int = 0,
                   int_prec: int = 10000,
                   num_samples=None) -> List[VRPInstance]:
    """
    preprocesses data format for NLNS
    """
    num_samples = len(dat) if num_samples is None else num_samples
    if data_type == 'uniform':
        if mode == 'eval_single':
            return [
                VRPInstance(nb_customers=len(instance.coords) - 1,
                            locations=instance.coords * int_prec,
                            original_locations=instance.coords * int_prec,
                            demand=(instance.node_features[:, 4] * int_prec).astype(int) if not is_denormed
                            else instance.node_features[:, 4].astype(int),
                            capacity=int(instance.vehicle_capacity * int_prec) if not is_denormed
                            else instance.original_capacity,
                            id_=instance.instance_id)
                for i, instance in enumerate(dat[offset:offset + num_samples])
            ]
        elif mode == 'train':
            train_instances = []
            for i, instance in enumerate(dat[offset:offset + num_samples]):
                NLNS_instance = VRPInstance(nb_customers=len(instance.coords) - 1,
                                            locations=instance.coords,
                                            original_locations=instance.coords,
                                            demand=(instance.node_features[:, 4] * int_prec).astype(int) if not is_denormed
                                            else instance.node_features[:, 4].astype(int),
                                            capacity=int(instance.vehicle_capacity * int_prec) if not is_denormed
                                            else instance.original_capacity,
                                            id_=instance.instance_id)
                train_instances.append(NLNS_instance)
                NLNS_instance.create_initial_solution()
            return train_instances

        else:
            return [
                VRPInstance(nb_customers=len(instance.coords) - 1,
                            locations=instance.coords,
                            original_locations=instance.coords,
                            demand=(instance.node_features[:, 4] * int_prec).astype(int) if not is_denormed
                            else instance.node_features[:, 4].astype(int),
                            capacity=int(instance.vehicle_capacity * int_prec) if not is_denormed
                            else instance.original_capacity,
                            id_=instance.instance_id)
                for i, instance in enumerate(dat[offset:offset + num_samples])
            ]

    elif data_type in ["X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7", "X_8", "X_9", "X_10", "X_11", "X_12", "X_13",
                       "X_14", "X_15", "X_16", "X_17", "X_1", "X_1", "XE", "uchoa", "S"]:
        if mode == 'train':
            # for training all capacities need to be the same --> so by default select the int_prec version of demand
            assert is_denormed is False
            train_instances = []
            for i, instance in enumerate(dat[offset:offset + num_samples]):
                NLNS_instance = VRPInstance(nb_customers=len(instance.coords) - 1,
                                            locations=instance.coords,
                                            original_locations=instance.original_locations,
                                            demand=(instance.node_features[:, 4] * int_prec).astype(int) if not is_denormed
                                            else instance.node_features[:, 4].astype(int),
                                            capacity=int(instance.vehicle_capacity * int_prec) if not is_denormed
                                            else instance.original_capacity,
                                            id_=instance.instance_id)
                train_instances.append(NLNS_instance)
                NLNS_instance.create_initial_solution()
            return train_instances
        else:
            if mode == 'eval_single':
                # only scale demands with original capacity if single_eval,
                # batch_eval cannot deal with different capacities in dataset
                return [
                    VRPInstance(nb_customers=len(instance.coords) - 1,
                                locations=instance.coords,
                                original_locations=instance.original_locations,
                                demand=(np.round(instance.node_features[:, 4] * instance.original_capacity)).astype(int)
                                if not is_denormed else instance.node_features[:, -1].astype(int),
                                capacity=instance.original_capacity,
                                id_=instance.instance_id)
                    for i, instance in enumerate(dat[offset:offset + num_samples])
                ]
            else:
                return [
                    VRPInstance(nb_customers=len(instance.coords) - 1,
                                locations=instance.coords,
                                original_locations=instance.original_locations,
                                demand=(instance.node_features[:, -1] * int_prec).astype(int),
                                capacity=int(instance.vehicle_capacity * int_prec),
                                id_=instance.instance_id)
                    for i, instance in enumerate(dat[offset:offset + num_samples])
                ]


def train_model(train_dataclass: CVRPDataset,
                val_dataclass: CVRPDataset,
                actor: VrpActorModel,
                critic: VrpCriticModel,
                train_cfg: DictConfig = None,
                run_id: Union[int, str] = None):
    # prelims
    run_id = np.random.randint(10000, 99999) if run_id is None or isinstance(run_id, str) else run_id

    # train data sampling (not on the fly for NLNS...) --> make it on the fly - memory issues
    # train_data = train_dataclass.sample(sample_size=train_cfg.batch_size * train_cfg.nb_batches_training_set)
    # val data can be sampled upfront
    logging.info("Generating validation data...")
    val_data = val_dataclass.sample(sample_size=train_cfg.valid_size)

    # call training procedure
    model_path = train.train_nlns(train_dataclass, val_data.data_transformed,
                                  actor, critic, run_id, train_cfg)
    search.evaluate_batch_search(train_cfg, model_path)


def eval_model(data: List[CVRPInstance],
               data_type: str,
               normalized: bool,
               model_path: str,
               instance_path: str,
               problem: str,
               batch_size: int,
               opts: Union[DictConfig, NamedTuple],
               time_limit: Union[int, float] = None,
               int_prec: int = 10000,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    assert model_path is not None, 'No model path given'

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    logger.info(f"eval_mode: {opts.mode}, data_key: {data_type}")

    eval_data = prep_data_NLNS(data,
                               is_denormed=not normalized,
                               mode=opts.mode,
                               data_type=data_type,
                               num_samples=len(data),
                               int_prec=int_prec,
                               offset=0)
    # set init values
    costs, durations, sol_parse, final_running_sols, final_running_times = float('inf'), 0.0, None, None, None
    running_sols_all, running_times_all = [], []
    sols_all, times_all = [], []
    # run evaluation
    if opts.mode == 'eval_batch':
        # if data_type == 'uniform':
        logger.info('### Batch Search ###')
        start_time = time.time()
        assert time_limit is not None, f"Can only evaluate Datasets which are of the same size in 'batch_eval'. " \
                                       f"Set eval_mode to single search."
        # for inst in eval_data:
        results = lns_batch_search_mp(opts,
                                      model_path,
                                      eval_data,
                                      time_limit,
                                      start_time)

        runtime = (time.time() - start_time)
        instance_ids, costs, iterations, solution_insts, running_sols, running_times = [], [], [], [], [], []
        for r in results:
            instance_ids.extend(list(range(len(r[1]) * r[0], len(r[1]) * (r[0] + 1))))
            costs.extend(r[1])
            iterations.append(r[2])
            running_sols.append(r[3])
            running_times.append(r[4])

        # flatten dictionaries of running values (for number of processes)
        running_ts_flat = {k: v for d in running_times for k, v in d.items()}
        running_sols_flat = {k: v for d in running_sols for k, v in d.items()}
        logger.info(f"Weird NLNS internal Test set rewards: {np.mean(costs)/int_prec:.3f} Total Runtime (s): {runtime:.1f} "
                    f"Iterations: {np.mean(iterations):.1f}")

        # parse solutions
        # for p in range(len(results)):
        inst_ids_from_data = [dat.instance_id for dat in data]
        # print('inst_ids_from_data', inst_ids_from_data)
        # for inst_id in range(len(data)):
        for inst_id, instance in zip(inst_ids_from_data, data):
            # print('inst_id', inst_id)
            running_sols_parsed = []
            running_times_parsed = []
            # print('running_sols_flat', running_sols_flat)
            # print('running_ts_flat', running_ts_flat)
            for running_sol, running_time in zip(running_sols_flat[inst_id], running_ts_flat[inst_id]):
                if running_time <= time_limit:
                    running_sols_parsed.append(
                        sol_postprocess(sol=running_sol, depot_idx=instance.depot_idx[0]))
                    running_times_parsed.append(running_time)
            # assert for instance have same amount of sols and runtimes
            assert len(running_sols_parsed) == len(running_times_parsed)
            running_sols_all.append(running_sols_parsed)
            running_times_all.append(running_times_parsed)
            sols_all.append(running_sols_parsed[-1])
            times_all.append(running_times_parsed[-1])
        assert len(running_sols_all) == len(running_times_all) == len(instance_ids)

        durations = [running_times_all[inst][-1] for inst in range(len(data))]

    elif opts.mode == 'eval_single':
        assert model_path is not None, 'No model path given'

        instance_names, costs, durations, tours = [], [], [], []
        logger.info("### Single instance search ###")
        if data_type == 'XE' or data_type == 'S':
            logger.info("Starting solving a single instance X or S")
        elif data_type == 'uniform':
            logger.info("Starting solving a uniform instance set")
        else:
            raise Exception("Unknown instance file format.")

        for i, instance_ in enumerate(eval_data):
            if time_limit is not None:
                cost, sol, run_sol, run_t, rt_total = lns_single_search_mp(instance_, time_limit, opts, model_path, i)
            else:
                cost, sol, run_sol, run_t, rt_total = lns_single_search_mp(instance_, data[i].time_limit,
                                                                           opts, model_path, i)
            instance_names.append(instance_path)
            costs.append(cost / int_prec)
            durations.append(rt_total)
            sols_all.append(sol_to_list(sol, data[i].depot_idx[0]))
            running_sols_all.append([sol_to_list(s, data[i].depot_idx[0]) for s in run_sol])
            running_times_all.append(run_t)

    return {}, make_RPSolution(problem, costs, durations, sols_all, data, running_sols_all, running_times_all)


# Get separate tours 2
def sol_postprocess(sol: np.ndarray,
                    depot_idx: int = 0) -> List[List]:
    sol_lst, lst = [], [0]
    for sol_ in sol:
        for e in sol_:
            if e[0] == 0:
                if len(lst) > depot_idx:
                    lst.append(e[0])
                    sol_lst.append(lst)
                    lst = []
                else:
                    lst.append(e[0])
            else:
                lst.append(e[0])
        if len(lst) > 0:
            sol_lst.append(lst)

    return sol_lst[1:]


# Get separate tours
def sol_to_list(sol: np.ndarray,
                depot_idx: int = 0) -> List[List]:
    sol_lst, lst = [], []
    for sol_ in sol:
        for e in sol_:
            if e[0] == 0:
                if len(lst) > depot_idx:
                    sol_lst.append(lst)
                lst = []
            else:
                lst.append(e[0])
        if len(lst) > 0:
            sol_lst.append(lst)

    return sol_lst


def make_RPSolution(problem, rews, durations, s_parsed, data,
                    running_sols=None, running_times=None) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    return [
        RPSolution(
            solution=sol if sol is not None else None,
            cost=r,
            num_vehicles=len(sol),
            run_time=rt,
            problem=problem,
            instance=inst,
            running_sols=sols_r if sols_r is not None else None,
            running_times=times_r if times_r is not None else None,
        )
        for sol, r, rt, sols_r, times_r, inst in zip(s_parsed, rews, durations, running_sols, running_times, data)
    ]


######################################################################################################
" Modified from NLNS repo by André Hottung and  Kevin Tierney to fit with the config file "


##################
#  batch search  #
##################

def lns_batch_search(instances, running_sols, running_ts, max_iterations, timelimit, operator_pairs,
                     lns_batch_size, config):
    if len(instances) % lns_batch_size != 0:
        raise Exception("Instance set size must be multiple of lns_batch_size for batch search.")

    instance_ids = [instance.instance_id for instance in instances]
    costs = [instance.get_costs_memory(config.round_distances) for instance in instances]  # Costs for each instance
    performance_EMA = [np.inf] * len(operator_pairs)  # Exponential moving average of avg. imprvt in last iterations
    start_time = time.time()
    for iteration_id in range(max_iterations):
        if time.time() - start_time > timelimit:
            break

        mean_cost_before_iteration = np.mean(costs)
        solution_copies = [instance.get_solution_copy() for instance in instances]

        # Select an LNS operator pair (destroy + repair operator)
        if config.lns_adaptive_search:
            selected_operator_pair_id = np.argmax(performance_EMA)  # select operator pair with the best EMA
        else:
            selected_operator_pair_id = np.random.randint(0, len(operator_pairs))  # select operator pair at random
        actor = operator_pairs[selected_operator_pair_id].model
        destroy_procedure = operator_pairs[selected_operator_pair_id].destroy_procedure
        p_destruction = operator_pairs[selected_operator_pair_id].p_destruction

        start_time_destroy = time.time()

        # Destroy instances
        search.destroy_instances(instances, destroy_procedure, p_destruction)

        # Repair instances
        for i in range(int(len(instances) / lns_batch_size)):
            with torch.no_grad():
                repair.repair(instances[i * lns_batch_size: (i + 1) * lns_batch_size], actor,
                              config)

        destroy_repair_duration = time.time() - start_time_destroy

        # print('before appending running costs / running times:')
        # print('costs: ', costs)
        # print('current time: ', time.time() - start_time)
        # print('destroy_repair_duration: ', destroy_repair_duration)
        # for i in range(len(instances)):
        for i, id in enumerate(instance_ids):
            # print(f'i, id in cost check {i, id}')
            cost = instances[i].get_costs_memory(config.round_distances)
            # print('costs[i]: ', costs[i])
            # print('cost', cost)
            # Only "accept" improving solutions (original code)
            if costs[i] < cost:
                instances[i].solution = solution_copies[i]  # reset solution
            else:
                costs[i] = cost
                # added code:
                # print(f"i {i}, instance.instance_id {instances[i].instance_id}")
                running_sols[id].append(instances[i].solution)
                running_ts[id].append(running_ts[id][0] + time.time() - start_time)  # adding init sol time
            # else:
            #     costs[i] = cost
            # print(f"running values for {i}: running_times={running_ts[i]} costs={costs[i]} "
            #       f"len(sols[i])= {len(running_sols[i])}")

        # If adaptive search is used, update performance scores
        if config.lns_adaptive_search:
            delta = (mean_cost_before_iteration - np.mean(costs)) / destroy_repair_duration
            if performance_EMA[selected_operator_pair_id] == np.inf:
                performance_EMA[selected_operator_pair_id] = delta
            performance_EMA[selected_operator_pair_id] = performance_EMA[selected_operator_pair_id] * (
                    1 - EMA_ALPHA) + delta * EMA_ALPHA
        # print(performance_EMA)

    # Verify solutions
    for id, instance in zip(instance_ids, instances):
        if instance.solution not in running_sols[id]:
            running_sols[id].append(instance.solution)
            running_ts[id].append(running_ts[id][0] + time.time() - start_time)  # adding init sol time
            # print('running_sols[i][-2:]', running_sols[i][-2:])
            # print('running_ts[i][-2:]', running_ts[i][-2:])
        instance.verify_solution(config)
    return costs, iteration_id, running_sols, running_ts


def _lns_batch_search_job(args):
    (i, init_start_time, test_size, config, instances_all, model_path, timelimit) = args
    lns_batch_size = config.lns_batch_size
    lns_operations = search.load_operator_pairs(model_path, config)
    instances_i = instances_all[i*test_size: (i+1)*test_size]
    running_sols, running_ts = {}, {}
    for instance in instances_i:
        instance.create_initial_solution()
        # get initial solution
        running_sols[instance.instance_id] = [instance.solution]
        running_ts[instance.instance_id] = [time.time() - init_start_time]
    # print('running_sols after initial solution', running_sols)
    # print('running_ts after initial solution', running_ts)
    costs, nb_iterations, run_sols, run_ts = None, None, None, None
    costs_all, nb_iterations_all = [], []
    if lns_batch_size > len(instances_i):
        warnings.warn('config.lns_batch_size > instance portion per CPU, '
                      'Defaulting to Batch size = nb_instances // config.lns_nb_cpus')
        lns_batch_size = len(instances_i)
    for j in range(int(len(instances_i) / lns_batch_size)):
        costs, nb_iterations, run_sols, run_ts = lns_batch_search(
            instances_i[j * lns_batch_size: (j + 1) * lns_batch_size],
            running_sols,
            running_ts,
            config.lns_max_iterations,
            timelimit,
            lns_operations,
            lns_batch_size,
            config)
        costs_all.extend(costs)
        nb_iterations_all.extend([nb_iterations] * lns_batch_size)
    return i, costs_all, nb_iterations_all, run_sols, run_ts


def lns_batch_search_mp(config, model_path, instances, timelimit, st_time):
    nb_instances = len(instances)
    assert nb_instances % config.lns_nb_cpus == 0
    test_size_per_cpu = nb_instances // config.lns_nb_cpus
    logger.info(f"nb_ins {nb_instances}, batch_size {config.lns_batch_size}, test_size_per_cpu {test_size_per_cpu}")

    if config.lns_nb_cpus > 1:
        with mp.Pool(config.lns_nb_cpus) as pool:
            results = pool.map(
                _lns_batch_search_job,
                [(i, st_time, test_size_per_cpu, config, instances, model_path, timelimit) for i in
                 range(config.lns_nb_cpus)]
            )
    else:
        results = _lns_batch_search_job((0, st_time, test_size_per_cpu, config, instances, model_path, timelimit))
        results = [results]
    return results


#################
# single search #
#################

def lns_single_seach_job(args):
    try:
        id, config, instance, model_path, queue_jobs, queue_results, pkl_instance_id, timelimit = args

        operator_pairs = search.load_operator_pairs(model_path, config)
        # instance = read_instance(instance_path, pkl_instance_id)

        T_min = config.lns_t_min

        # Repeat until the process is terminated
        while True:
            solution, incumbent_cost = queue_jobs.get()
            incumbent_solution = deepcopy(solution)
            cur_cost = np.inf
            instance.solution = solution
            start_time_reheating = time.time()

            # Create a batch of copies of the same instances that can be repaired in parallel
            instance_copies = [deepcopy(instance) for _ in range(config.lns_batch_size)]

            iter = -1
            # Repeat until the time limit of one reheating iteration is reached
            while time.time() - start_time_reheating < timelimit / config.lns_reheating_nb:
                iter += 1

                # Set the first config.lns_Z_param percent of the instances/solutions in the batch
                # to the last accepted solution
                for i in range(int(config.lns_Z_param * config.lns_batch_size)):
                    instance_copies[i] = deepcopy(instance)

                # Select an LNS operator pair (destroy + repair operator)
                selected_operator_pair_id = np.random.randint(0, len(operator_pairs))
                actor = operator_pairs[selected_operator_pair_id].model
                destroy_procedure = operator_pairs[selected_operator_pair_id].destroy_procedure
                p_destruction = operator_pairs[selected_operator_pair_id].p_destruction

                # Destroy instances
                search.destroy_instances(instance_copies, destroy_procedure, p_destruction)

                # Repair instances
                for i in range(int(len(instance_copies) / config.lns_batch_size)):
                    with torch.no_grad():
                        repair.repair(
                            instance_copies[i * config.lns_batch_size: (i + 1) * config.lns_batch_size], actor, config)

                costs = [instance.get_costs_memory(config.round_distances) for instance in instance_copies]

                # Calculate the T_max and T_factor values for simulated annealing in the first iteration
                if iter == 0:
                    q75, q25 = np.percentile(costs, [75, 25])
                    T_max = q75 - q25
                    T_factor = -math.log(T_max / T_min)

                min_costs = min(costs)

                # Update incumbent if a new best solution is found
                if min_costs <= incumbent_cost:
                    incumbent_solution = deepcopy(instance_copies[np.argmin(costs)].solution)
                    incumbent_cost = min_costs

                # Calculate simulated annealing temperature
                T = T_max * math.exp(
                    T_factor * (time.time() - start_time_reheating) / (timelimit / config.lns_reheating_nb))

                # Accept a solution if the acceptance criteria is fulfilled
                if min_costs <= cur_cost or np.random.rand() < math.exp(-(min(costs) - cur_cost) / T):
                    instance.solution = instance_copies[np.argmin(costs)].solution
                    cur_cost = min_costs

            queue_results.put([incumbent_solution, incumbent_cost])

    except Exception as e:
        print("Exception in lns_single_search job: {0}".format(e))


def lns_single_search_mp(instance, timelimit, config, model_path, instance_id=None):
    start_time = time.time()
    instance.create_initial_solution()
    running_ts = [time.time() - start_time]
    incumbent_cost = instance.get_costs(config.round_distances)
    instance.verify_solution(config)
    running_sols = [instance.solution]
    running_costs = [incumbent_cost]

    m = Manager()
    queue_jobs = m.Queue()
    queue_results = m.Queue()
    pool = Pool(processes=config.lns_nb_cpus)
    pool.map_async(lns_single_seach_job,
                   [(i, config, instance, model_path, queue_jobs, queue_results, instance_id, timelimit) for i in
                    range(config.lns_nb_cpus)])
    # Distribute starting solution to search processes
    for i in range(config.lns_nb_cpus):
        queue_jobs.put([instance.solution, incumbent_cost])

    prep_time = time.time() - start_time
    logger.info(f'Preperation and initial solution for NLNS single search took {prep_time} seconds')
    final_cost_sol_time = None
    start_search = time.time()
    # start_search_plus_init_sol = start_search + running_ts[0]
    # (time.time() - start_time) + prep_time
    while time.time() - start_search < timelimit:
        # Receive the incumbent solution from a finished search process (reheating iteration finished)
        result = queue_results.get()
        if result != 0:
            if result[1] < incumbent_cost:
                incumbent_cost = result[1]
                instance.solution = result[0]
                if time.time() - start_search < timelimit:
                    print(f'appending sol with cost {instance.get_costs(config.round_distances)} at '
                          f'{time.time() - start_search} seconds')
                    running_ts.append(time.time() - start_search)
                    running_sols.append(instance.solution)
                    running_costs.append(instance.get_costs(config.round_distances))
        final_cost_sol_time = (instance.get_costs(config.round_distances), instance.solution,
                               time.time() - start_search)
        # Distribute incumbent solution to search processes
        queue_jobs.put([instance.solution, incumbent_cost])

    pool.terminate()
    duration = (time.time() - start_search) + running_ts[0]  # time.time() - start_time
    print('duration', duration)
    if running_sols[-1] != final_cost_sol_time[1]:
        print(f'final_cost_sol_time[0] cost {final_cost_sol_time[0]} vs final running costs {running_costs[-1]}')
        print(f'final_cost_sol_time[2] time {final_cost_sol_time[2]} vs final running time {running_ts[-1]}')
        if final_cost_sol_time[2] <= timelimit:
            running_sols.append(final_cost_sol_time[1])
            running_ts.append(final_cost_sol_time[2])
        else:
            final_cost_sol_time = (running_costs[-1], running_sols[-1], running_ts[-1])
        assert running_sols[-1] == final_cost_sol_time[1]
    instance.verify_solution(config)
    return final_cost_sol_time[0], final_cost_sol_time[1], running_sols, running_ts, duration
