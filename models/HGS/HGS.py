####
# This code is partly based on and adapted from the code provided in
# https://github.com/jokofa/NeuroLS (Falkner, Jonas K., et al. "Learning to Control Local Search for Combinatorial
# Optimization." Joint European Conference on Machine Learning and
# Knowledge Discovery in Databases. Cham: Springer Nature Switzerland, 2022.) and
# https://github.com/liangxinedu/NeuroLKH (Xin, Liang, et al. "NeuroLKH:
# Combining deep learning model with Lin-Kernighan-Helsgaun heuristic for
# solving the traveling salesman problem."
# Advances in Neural Information Processing Systems 34 (2021): 7472-7483.)
import warnings
import os
import logging
from typing import Optional, Dict, Union, NamedTuple, List, Callable, Type
from abc import abstractmethod
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

from formats import CVRPInstance, TSPInstance, RPSolution
import tqdm
from subprocess import check_call
# from hygese import Solver

import torch
from torch.autograd import Variable
from visualization import Viewer

logger = logging.getLogger(__name__)

CVRP_DEFAULTS = {  # max num vehicles and integer capacity per problem size (not for Uchoa-type data)
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}


def read_results(log_filename, sol_filename):
    sol_summary = sol_filename + ".PG.csv"  # to get results for running values of search progress file
    running_objs_log = []
    running_times_log = []
    num_vehicles = 0
    line_count = 1
    prep_time = 0
    prev_best_obj = float("inf")
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read log file to get running objective
            if line[:2] == "It" and float(line.strip().split()[9]) < prev_best_obj:
                line = line.strip().split()
                running_objs_log.append(float(line[9]))  # add better obj than prev_best_obj
                running_times_log.append(float(line[5]))
            elif line[:13] == "----- GENETIC":
                final_runtime = float(line.strip().split()[-1])  # seconds
            line_count += 1
            if running_objs_log:
                prev_best_obj = running_objs_log[-1]
    running_objs, running_times = [], []
    # get results for running values of search progress file
    with open(sol_summary, "r") as f:
        lines = f.readlines()
        for line in lines:
            # print('line.strip().split(";")[2]', line.strip().split(";")[2])
            running_objs.append(int(line.strip().split(";")[2]))  # add recorded sequence of better objs
            running_times.append(float(line.strip().split(";")[3]))  # in seconds
    final_obj = running_objs[-1]
    tours = []
    dim, total_length = 0, 0
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):  # read out solution tours
            if line[:5] == 'Route':
                l = line.strip().split()
                tours.append([int(idx) for idx in l[2:]])
            else:
                assert float(line.strip().split()[1]) == float(final_obj)

    # assert len(tours) == num_vehicles
    num_vehicles = len(tours)

    # return objs, runtimes, ...
    return {
        "final_obj": final_obj,
        "runtime": final_runtime,
        "num_vehicles": num_vehicles,
        "solution": tours,
        "running_costs": running_objs,
        "running_times": running_times,
    }


# code write_instance from NeuroLKH / NeuroLS
def write_instance(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        n_nodes = len(instance[0]) - 1
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : " + str(instance[2]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        # s = 10000
        #  * s
        for i in range(n_nodes + 1):
            f.write(
                " " + str(i + 1) + " " + str(instance[0][i][0])[:15] + " " + str(instance[0][i][1])[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2) + " " + str(instance[1][i]) + "\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def method_wrapper(args):
    return solve_HGS(*args)


def solve_HGS(dataset_name,
              instance,
              cvrpinstance,
              instance_name,
              rerun,
              time_limit,
              exe_path,
              max_k=None):
    log_filename = "result/" + dataset_name + "/HGS_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".vrp"
    out_path = "result/" + dataset_name + "/HGS_log/"
    solution_filename = "result/" + dataset_name + "/HGS_log/" + instance_name + ".sol"
    # print('cvrpinstance.graph_size', cvrpinstance.graph_size)
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        if max_k is not None:
            with open(log_filename, "w") as f:
                check_call([str(exe_path), instance_filename, solution_filename, '-t', str(time_limit), '-veh',
                            str(max_k), '-round', str(0)], stdout=f)
        else:
            with open(log_filename, "w") as f:
                check_call([str(exe_path), instance_filename, solution_filename, '-t', str(time_limit)], stdout=f)
    try:
        res = read_results(log_filename, solution_filename)
    except FileNotFoundError:
        warnings.warn(f"HGS could not find solution for instance {instance_name}. See dir <outputs/visualisations/> "
                      "for further information on this instance.")

       # if instance is not solved, set res to None and save instance gif to outputs/visualisations/
        res = None
        # plot instances which failed -> saved in visualisations dir
        SAVE_PATH = os.path.join(os.getcwd(), 'visualisations/failed_')
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        save_name = "c_dist_" + str(cvrpinstance.coords_dist) + "_d_dist_" + str(cvrpinstance.demands_dist) \
                    + "_depot_type_" + str(cvrpinstance.depot_type) + "_instance_" + instance_name
        view = Viewer(locs=cvrpinstance.coords, save_dir=SAVE_PATH, gif_naming=save_name)

    return res


def eval_HGS(
        data: List[CVRPInstance],
        hgs_exe_path: str,
        num_workers: int = 1,
        is_normalized: bool = True,
        int_prec: int = 10000,
        TIME_LIMIT: int = None
):
    # re-adjust int_prec if data is original (unnormalized scale) to avoid wrongly scaled outputs
    int_prec = 1 if not is_normalized else int_prec
    dataset_name = "CVRP"
    rerun = True
    # print("NUM_WORKERS", num_workers)
    if num_workers > os.cpu_count():
        warnings.warn(f"num_workers > num logical cores! This can lead to "
                      f"decrease in performance if env is not IO bound.")

    # set up directories
    os.makedirs("result/" + dataset_name + "/HGS_log", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)

    if is_normalized:
        dataset = [
            [
                (d.coords*int_prec).astype(int).tolist(),
                np.ceil(d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist(),
                int(d.vehicle_capacity * int_prec)
            ] for d in data
        ]
    else:
        # print('data[0]', data[0])
        if not data[0].type == "Golden":
            assert isinstance(data[0].coords[0][0], np.int64)  # if input is not normalized coords need to be int
            assert int_prec == 1  # make sure re-adjusted int_prec to 1, so to have correct output costs
            dataset = [
                [
                    d.coords.astype(int).tolist(),
                    np.ceil(d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist(),
                    int(d.original_capacity * int_prec)
                ] for d in data
            ]
        else:
            assert int_prec == 1  # make sure re-adjusted int_prec to 1, so to have correct output costs
            dataset = [
                [
                    d.coords.tolist(),
                    np.ceil(d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist(),
                    int(d.original_capacity * int_prec)
                ] for d in data
            ]
            # print('dataset[0]', dataset[0])
    # run solver
    if num_workers <= 1:
        if TIME_LIMIT is not None:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                                rerun, TIME_LIMIT, hgs_exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
        else:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                                rerun, data[i].time_limit, hgs_exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
    else:
        if TIME_LIMIT is not None:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, TIME_LIMIT, hgs_exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))
        else:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, data[i].time_limit, hgs_exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))

    # s = 10000  # precision hardcoded in write_instance()
    objs = np.asarray([np.asarray(r['running_costs']) for r in results if r is not None], dtype=object)
    objs = objs / int_prec
    final_objs = [r['final_obj'] / int_prec for r in results if r is not None]
    final_rts = [r['runtime'] for r in results if r is not None]
    # runtimes = np.array([r['runtimes'] for r in results if r is not None])
    # running_costs = np.array([r['running_costs'] for r in results if r is not None])
    # + prep_rt + init_sol_rt
    solutions = [
        RPSolution(
            solution=r['solution'] if r is not None else None,
            run_time=r['runtime'] if r is not None else None,
            running_costs=list(np.array(r['running_costs']) / int_prec) if r is not None else None,
            running_times=[r['running_times'][t] for t in range(len(r['running_times']))] if r is not None else None,
            problem=dataset_name.upper(),
            instance=d
        ) for r, d in zip(results, data)
    ]

    results_ = {
        "objs": objs,
        "final_objs": final_objs,
        "runtime": final_rts,
    }
    return results_, solutions


# running pip-installable python wrapper for HGS
def eval_hygese(solver,  # : Solver
                data: Union[List[TSPInstance], List[CVRPInstance]],
                problem: str,
                precision: Union[int, float] = 10000):
    solutions = []
    own_run_times = []
    for i, transformed_instance in enumerate(prep_data_hygese(problem, data)):
        start = timer()
        # with open(log_filename, "w") as f:
        #     # check_call(["./LKH", para_filename], stdout=f)
        #    check_call([str(exe_path), instance_filename, '--outpath', out_path, '--time', str(time_limit)], stdout=f)
        result = solver.solve_cvrp(data=transformed_instance)
        t = timer() - start
        own_run_times.append(t)
        print('result.cost', result.cost)
        print('result.routes', result.routes)
        print('result.time', result.time)
        print('type(result)', type(result))
        solutions.append(make_RPSolution(problem, data[i], result, prec=precision))
    return own_run_times, solutions


def eval_rp(
        data: Union[List[TSPInstance], List[CVRPInstance]],
        problem: str,
        is_normalized: bool = True,
        policy = None,  # Solver
        hgs_exe_path: str = None,
        time_limit: Union[int, float] = None,
        num_workers: int = 1,
        int_precision: int = 10000,
        nbIter: int = 20000,
        useSwapStar: bool = True
):
    assert (policy is None and hgs_exe_path is not None) or (policy is not None and hgs_exe_path is None), "Either \
    call HGS directly or hygese python wrapper."

    if policy is not None:
        return eval_hygese(solver=policy, data=data, problem=problem, precision=int_precision)
    else:
        return eval_HGS(data=data,
                        is_normalized=is_normalized,
                        hgs_exe_path=hgs_exe_path,
                        num_workers=num_workers,
                        int_prec=int_precision,
                        TIME_LIMIT=time_limit)


def prep_data_hygese(problem: str,
                     dat: Union[List[TSPInstance], List[CVRPInstance]],
                     precision: int = 10000) -> List[dict]:
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    # init data dict

    data_list = []
    if problem.lower() == "cvrp":
        for instance in dat:
            data = dict()
            data['x_coordinates'] = instance.coords[:, 0] * precision  # ).astype(int)
            data['y_coordinates'] = instance.coords[:, 1] * precision  # ).astype(int)

            # You may also supply distance_matrix instead of coordinates, or in addition to coordinates
            # If you supply distance_matrix, it will be used for cost calculation.
            # The additional coordinates will be helpful in speeding up the algorithm.
            # data['distance_matrix'] = dist_mtx
            data['service_times'] = np.zeros(instance.graph_size)
            demands = (instance.node_features[:, instance.constraint_idx[0]] * precision).astype(int)
            data['demands'] = demands
            assert len(data['x_coordinates']) == len(data['demands']) == instance.graph_size
            data['vehicle_capacity'] = int(instance.vehicle_capacity * precision)
            data['num_vehicles'] = CVRP_DEFAULTS[instance.graph_size - 1][0]
            data['depot'] = 0
            data_list.append(data)
        return data_list

    elif problem.lower() == "tsp":
        data = dict()
        data['distance_matrix'] = [
            [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
            [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
            [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
            [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
            [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
            [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
            [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
            [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
            [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
            [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
            [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
            [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
            [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
        ]

        return [data]
    else:
        raise NotImplementedError


def make_RPSolution(problem: str, instance: Union[TSPInstance, CVRPInstance], result, prec) -> RPSolution:
    """Parse model solution back to RPSolution for consistent evaluation"""

    return RPSolution(
        solution=result.routes if result.routes is not None else None,
        cost=result.cost / prec if result.cost is not None else None,
        num_vehicles=result.n_routes if result.n_routes is not None else None,
        run_time=result.time if result.time is not None else None,
        problem=problem,
        instance=instance,
    )


def l1_distance(x1, y1, x2, y2):
    """2d Manhattan distance, returns only integer part"""
    return abs(x1 - x2) + abs(y1 - y2)


def l2_distance(x1, y1, x2, y2):
    """Normal 2d euclidean distance."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_distances(locations, distance_metric=None, round_to_int=True):
    """Calculate distances between locations as matrix.
    If no distance_metric is specified, uses l2 euclidean distance"""
    metric = l2_distance if distance_metric is None else distance_metric

    num_locations = len(locations)
    matrix = {}

    for from_node in range(num_locations):
        matrix[from_node] = {}
        for to_node in range(num_locations):
            x1 = locations[from_node][0]
            y1 = locations[from_node][1]
            x2 = locations[to_node][0]
            y2 = locations[to_node][1]
            if round_to_int:
                matrix[from_node][to_node] = int(round(metric(x1, y1, x2, y2), 0))
            else:
                matrix[from_node][to_node] = metric(x1, y1, x2, y2)

    return matrix


# TESTS
# =================================
def _create_data():
    """Stores the data for the problem."""
    rnds = np.random.RandomState(1)
    n = 21
    k = 4
    data = {}
    data['n'] = n
    locs = rnds.uniform(0, 1, size=(n, 2))
    locs[0] = [0.5, 0.5]
    data['locations'] = locs

    dists = calculate_distances(locs * 100)
    data['distance_matrix'] = dists
    data['k'] = k
    data['depot'] = 0

    # CVRP
    data['demands'] = list(np.maximum(rnds.poisson(2, n), [1]))
    data['demands'][0] = 0
    print(data['demands'])
    data['vehicle_capacities'] = [16] * k
    print(data['vehicle_capacities'])

    return data


def _test_tsp():
    data = _create_data()
    data['k'] = 1

    solver = TSPSolver()
    solver.create_model(data)
    solution, info = solver.solve(maximum_cap=1000)

    print(solution)
    print(info)

    solver.plot_solution()
    solver.plot_search_trajectory()


def _test_cvrp():
    data = _create_data()

    solver = CVRPSolver()
    solver.create_model(data)
    solution, info = solver.solve(first_solutions_strategy='Savings',
                                  local_search_strategy='guided_local_search',
                                  time_limit=10,
                                  verbose=True)

    print(solution)
    print(info)

    solver.plot_solution()
    solver.plot_search_trajectory()
