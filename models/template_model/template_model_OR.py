#
import warnings
import os
import logging
from typing import Union, List
from timeit import default_timer as timer

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from formats import CVRPInstance, RPSolution
import tqdm
from subprocess import check_call

from visualization import Viewer

logger = logging.getLogger(__name__)


# necessary functions in template_model_OR.py:
# (a) function that transforms CVRPInstances to the data format used by model
# (b) function that transforms the model solutions to the RPSolution format
# (c) function that calls the model’s internal evaluation function and processes the transformations (a) and (b)

# Function (a)
def write_instance(instance, instance_name, instance_filename):
    # code mainly from write_instance in NeuroLKH / NeuroLS source code
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


# Function (b)
def make_RPSolution(problem: str, instance: CVRPInstance, result, prec) -> RPSolution:
    """Parse model solution back to RPSolution for consistent evaluation"""

    return RPSolution(
        solution=result.routes if result.routes is not None else None,
        cost=result.cost / prec if result.cost is not None else None,
        num_vehicles=result.n_routes if result.n_routes is not None else None,
        run_time=result.time if result.time is not None else None,
        problem=problem,
        instance=instance,
    )


# Function (c) - 1
def eval_OR_method(
        data: List[CVRPInstance],
        exe_path: str,
        problem: "CVRP",
        num_workers: int = 1,
        is_normalized: bool = True,
        int_prec: int = 10000,
        time_limit: int = None
):
    # re-adjust int_prec if data is original (un-normalized scale) to avoid wrongly scaled outputs
    int_prec = 1 if not is_normalized else int_prec
    dataset_name = "CVRP"  # customizable
    rerun = True
    # print("NUM_WORKERS", num_workers)
    if num_workers > os.cpu_count():
        warnings.warn(f"num_workers > num logical cores! This can lead to "
                      f"decrease in performance if env is not IO bound.")

    # set up directories
    os.makedirs("result/" + dataset_name + "/method_log", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)

    if is_normalized:
        dataset = [
            [
                (d.coords * int_prec).astype(int).tolist(),
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
    # run solver
    if num_workers <= 1:
        if time_limit is not None:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                                rerun, time_limit, exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
        else:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                                rerun, data[i].time_limit, exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
    else:
        if time_limit is not None:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, time_limit, exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))
        else:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, data[i].time_limit, exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))

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

    # some customizable info
    objs = np.asarray([np.asarray(r['running_costs']) for r in results if r is not None], dtype=object)
    objs = objs / int_prec
    final_objs = [r['final_obj'] / int_prec for r in results if r is not None]
    final_rts = [r['runtime'] for r in results if r is not None]
    results_ = {
        "objs": objs,
        "final_objs": final_objs,
        "runtime": final_rts,
    }
    return results_, solutions

# Function (c) - 2
def method_wrapper(args):
    return solve_OR_search(*args)

# Function (c) - 3
def solve_OR_search(dataset_name,
                    instance,
                    cvrpinstance,
                    instance_name,
                    rerun,
                    time_limit,
                    exe_path):
    log_filename = "result/" + dataset_name + "/method_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".vrp"
    out_path = "result/" + dataset_name + "/method_log/"
    solution_filename = "result/" + dataset_name + "/method_log/" + instance_name + ".sol"
    # For example:
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        with open(log_filename, "w") as f:
            # call method execution path with arguments (such as time limit), while writing output to log
            check_call([str(exe_path), instance_filename, solution_filename, '-t', str(time_limit)], stdout=f)
    try:
        res = read_results(log_filename, solution_filename)
    except FileNotFoundError:
        warnings.warn(f"Method could not find solution for instance {instance_name}. See dir <outputs/visualisations/> "
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


# Additional Helper function
def read_results(log_filename, sol_filename):
    sol_summary = sol_filename + ".PG.csv"  # to get results for running values of search progress file
    running_objs = []
    running_times = []
    line_count = 1
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read log file to get running objective & times
            ...
    final_obj = running_objs[-1]
    final_runtime = running_times[-1]
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