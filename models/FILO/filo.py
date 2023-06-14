import os
import time
import warnings
import tqdm
from typing import List
from multiprocessing import Pool
from subprocess import check_call
import tempfile

import numpy as np
import torch
from torch.autograd import Variable

from typing import Optional, List, Union
from formats import TSPInstance, CVRPInstance, RPSolution
from visualization import Viewer

CVRP_DEFAULTS = {  # num vehicles and integer capacity per problem size --> NOT FOR UCHOA DATA
    20: [4, 30],
    50: [16, 40],
    100: [100, 50]
}

def method_wrapper(args):
    return solve_FILO(*args)

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
        # s = 1000000
        # * s
        for i in range(n_nodes + 1):
            f.write(
                " " + str(i + 1) + " " + str(instance[0][i][0])[:15] + " " + str(instance[0][i][1])[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2) + " " + str(instance[1][i]) + "\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def read_results(log_filename, sol_filename):
    # s = 1000000.0  # precision hardcoded
    running_objs = []
    running_times = []
    num_vehicles = 0
    line_count = 1
    prep_time = 0
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            # print('line', line)
            # print('line.strip().split()', line.strip().split())
            # print('len(line.strip().split())', len(line.strip().split()))
            if line[:4] == "Done":
                # line = line.strip().split()
                prep_time += int(line.strip().split()[2]) / 1000  # because prep time in milliseconds
            if line[:16] == "Initial solution":
                # line = line.strip().split(" ")
                running_objs.append(int(line.strip().split(" ")[4][:-1]))  # add initial obj
                running_times.append(float(prep_time))  # in seconds
            # if better obj found that wasn't recorded yet
            if line_count > 32 and len(line.strip().split()) == 10 and not line[:5] == '*****':
                # print('line', line)
                # print('line.strip().split()', line.strip().split())
                # if len(line.strip().split()) == 10:
                if int(line.strip().split()[2]) < running_objs[-1]:
                    # print('line.strip().split()', line.strip().split())
                    # print('float(line.strip().split()[4])', float(line.strip().split()[4]))
                    running_objs.append(int(line.strip().split()[2]))
                    running_times.append(float(line.strip().split()[4]))
            if line[:3] == "obj":
                # print('running_times 1', running_times)
                # print('running_obj 1', running_objs)
                line_ = line.strip().split()
                final_obj = int(line_[2][:-1])
                # print('final_obj', final_obj)
                final_rt = float(line_[-2][1:]) / 1000  # (in milliseconds)
                # print('final_rt',final_rt)
                num_vehicles = int(line_[6][:-1])
                if final_obj != running_objs[-1]:
                    running_objs.append(final_obj)
                if final_rt != running_times[-1] and len(running_objs) == len(running_times):
                    # replace final running times with milliseconds r.t. if 0s
                    running_times[-1] = final_rt
                elif final_rt != running_times[-1]:
                    running_times.append(final_rt)
                # print('running_times 2', running_times)
                # print('running_obj 2', running_objs)
            if line[:13] == "Run completed":
                final_runtime = float(line.strip().split()[3])  # seconds
            line_count += 1
        if len(running_times) != len(running_objs):
            print('running_times', running_times)
            print('running_objs', running_objs)
        assert len(running_times) == len(running_objs)
        assert running_objs[-1] == final_obj
        assert running_times[-1] == final_rt
    # print('len(running_objs)', len(running_objs))
    # print('len(running_times)', len(running_times))
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

    assert len(tours) == num_vehicles

    # return objs, runtimes, ...
    return {
        "final_obj": final_obj,
        "runtime": final_runtime,
        "num_vehicles": num_vehicles,
        "solution": tours,
        "running_costs": running_objs,
        "running_times": running_times,
    }


def solve_FILO(dataset_name,
               instance,
               cvrpinstance,
               instance_name,
               rerun=False,
               time_limit=10,
               exe_path=None):
    log_filename = "result/" + dataset_name + "/FILO_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    out_path = "result/" + dataset_name + "/FILO_log/"
    solution_filename = "result/" + dataset_name + "/FILO_log/" + instance_name + ".cvrp_seed-0.vrp.sol"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        with open(log_filename, "w") as f:
            # check_call(["./LKH", para_filename], stdout=f)
            check_call([str(exe_path), instance_filename, '--outpath', out_path, '--time', str(time_limit)], stdout=f)
    try:
        res = read_results(log_filename, solution_filename)
    except FileNotFoundError:
        warnings.warn(f"FILO could not find solution for instance {instance_name}. See dir <outputs/visualisations/> "
                      f"for further information on this instance.")

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


def cvrp_inference(
        data: List[CVRPInstance],
        filo_exe_path: str,
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
    os.makedirs("result/" + dataset_name + "/FILO_log", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)

    # convert data to input format for FILO
    # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
    # in l2O-meta originally: 'np.ceil(1 + d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist()
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
    # dataset = [
    #     [
    #         d.coords.tolist(),
    #         np.ceil(d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist(),
    #         int(d.vehicle_capacity * int_prec)
    #     ] for d in data
    # ]

    # run solver
    if num_workers <= 1:
        if TIME_LIMIT is not None:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                           rerun, TIME_LIMIT, filo_exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
        else:
            results = list(tqdm.tqdm([
                method_wrapper((dataset_name, dataset[i], data[i], str(data[i].instance_id),
                           rerun, data[i].time_limit, filo_exe_path))
                for i in range(len(dataset))
            ], total=len(dataset)))
    else:
        if TIME_LIMIT is not None:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, TIME_LIMIT, filo_exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))
        else:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    (dataset_name, dataset[i], data[i], str(data[i].instance_id),
                     rerun, data[i].time_limit, filo_exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))

    # s = 1000000.0  # precision hardcoded by authors in write_instance()
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
            running_times=[r['running_times'][t] for t in range(len(r['running_times']))],
            # if r is not None else None,
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
