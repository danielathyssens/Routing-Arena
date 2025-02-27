#
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
from formats import TSPInstance, CVRPInstance, CVRPTWInstance, RPSolution
from visualization import Viewer

# from baselines.CVRP.formats import CVRP_DEFAULTS
from models.NeuroLKH.NeuroLKH.net.sgcn_model import SparseGCNModel
from models.NeuroLKH.NeuroLKH.CVRP_test import write_candidate as write_candidate_cvrp
from models.NeuroLKH.NeuroLKH.CVRP_test import read_feat as read_feat_cvrp
from models.NeuroLKH.NeuroLKH.neurolkh_utils import write_instance_vrptw, write_para_vrptw, get_features_vrptw, \
    write_candidate_vrptw, infer_SGN_vrptw, prep_data_NeuroLKH, infer_SGN_cvrp, write_para_cvrp, \
    write_instance_cvrp, write_para_tsp
from models.NeuroLKH.NeuroLKH.test import write_instance as write_instance_tsp
# from models.NeuroLKH.NeuroLKH.test import write_para as write_para_tsp
from models.NeuroLKH.NeuroLKH.test import read_feat as read_feat_tsp
from models.NeuroLKH.NeuroLKH.test import write_candidate_pi as write_candidate_tsp
from models.NeuroLKH.NeuroLKH.test import infer_SGN as infer_SGN_tsp

#
CVRP_DEFAULTS = {  # num vehicles and integer capacity per problem size --> NOT FOR UCHOA DATA
    20: [4, 30],
    50: [16, 40],
    100: [100, 50]
}


def read_results(log_filename, sol_filename):
    s = 1000000.0  # precision hardcoded by authors in write_instance()
    objs = []
    running_objs = []
    objs_final = None
    # final_obj = None
    # final_rt = None
    penalties = []
    runtimes = []
    running_times = []
    num_vehicles = 0
    # print('log_filename', log_filename)
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            if "VEHICLES" in line:
                l = line.strip().split(" ")
                num_vehicles = int(l[-1])
            elif line[:6] == "-Trial":
                line = line.strip().split(" ")
                # assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]) if not "CVRPTW" in log_filename else int(line[-2])/s)
                penalties.append(int(line[-3]))
                runtimes.append(float(line[-1]))
            # print('objs', objs)
            # print('runtimes', runtimes)
        if "CVRPTW" in log_filename:
            log_tour = [int(i)/s for i in lines[-1].strip().split(" ")]
            #      line = line.strip().split(" ")
            #      running_objs.append(int(line[-5][2:-1])/s)
            #      running_times.append(float(line[-2]))
        try:
            if "CVRPTW" in log_filename:
                # for CVRPTW the log file also gives the tour at the last line
                final_obj = int(lines[-11].split(",")[0].split(" ")[-1])/s
                final_rt = float(lines[-2].split("=")[1].split(" ")[1])
            elif "CVRP" in log_filename:
                final_obj = int(lines[-12].split(",")[0].split(" ")[-1])
                final_rt = float(lines[-2].split("=")[1].split(" ")[1])
            else:
                assert "TSP" in log_filename
                final_obj = int(lines[-6].split(",")[0].split(" ")[-1])
                final_rt = float(lines[-2].split("=")[1].split(" ")[1])
                # print('final_rt', final_rt)
            # print('final_obj', final_obj)
            # need to delete better costs than final solution (else PI, WRAP not working) --> seems like bug in LKH
            objs_final = []
            runtimes_final = []
            objs = running_objs if not objs else objs
            runtimes = running_times if not runtimes else runtimes
            for obj, r_t in zip(objs, runtimes):
                # print('obj, r_t', obj, r_t)
                if obj >= final_obj:
                    objs_final.append(obj)
                    runtimes_final.append(r_t)
                if objs_final:
                    if objs_final[-1] == final_obj:
                        break
                # print('objs_final', objs_final)
            # print('runtimes_final[-5:]', runtimes_final[-5:])
            # print('objs_final[-5:]', objs_final[-5:])
            # print('objs_final[-1] == final_obj', objs_final[-1] == final_obj)
            assert objs_final[-1] == final_obj
        except ValueError:
            warnings.warn("LKH did not find a solution in Time Limit")
            final_obj = None
        except AssertionError:
            print('final_obj', final_obj)
            print('objs[-8:]', objs[-8:])
            print('len(objs)', len(objs))
            print('len(runtimes)', len(runtimes))
        except IndexError:
            print('log_filename', log_filename)
            print('final_obj', final_obj)
            print('objs[-8:]', objs[-8:])
            print('len(objs)', len(objs))
            print('len(runtimes)', len(runtimes))
            if final_obj is not None:
                # only have one (final) solution
                objs_final = [final_obj]
                runtimes_final = [final_rt]

    tours = []
    dim, total_length = 0, 0
    try:
        with open(sol_filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):  # read out solution tours
                if "DIMENSION" in line:
                    l = line.strip().split(" ")
                    dim = int(l[-1])
                elif "Length" in line:
                    l = line.strip().split(" ")
                    total_length = int(l[-1])
                elif i > 5 and not "EOF" in line:
                    idx = int(line)
                    if i == 6:
                        assert idx == 1
                    tours.append(idx)
        assert tours[-1] == -1
        assert len(tours) == dim + 1
    except FileNotFoundError:
        print("No solution file - for CVRPTW can get it from the .log file")
        print('log_tour', log_tour)
        tours = log_tour
    # print('tours', tours)

    N = dim - num_vehicles

    # reformat tours
    tours = (np.array(tours) - 1).tolist()  # reduce idx by 1 (since TSPLIB format starts at 1)
    plan = []
    t = []

    if "TSP" in log_filename:
        for n in tours[0:]:
            if n < 0 or n > N:
                plan.append(t)
                t = []
            else:
                t.append(n)
    else:
        for n in tours[1:]:
            if n <= 0 or n > N:
                plan.append(t)
                t = []
            else:
                t.append(n)

    assert len(plan) == num_vehicles

    # delete empty tours (to get correct number of vehicles used)
    plan_ = [tour for tour in plan if len(tour) != 0]
    num_vehicles_ = len(plan_)
    # print('plan_', plan_)
    # return objs, penalties, runtimes
    return {
        "final_obj": final_obj,
        "objs": objs_final,
        "penalties": penalties,
        "runtimes": runtimes_final,
        "N": N,
        "num_vehicles": num_vehicles_,
        "total_length": total_length,
        "solution": plan_,
        # "running_costs": running_objs,
        # "running_times": running_times,
    }


def solve_LKH(dataset_name,
              instance,
              cvrpinstance,
              instance_name,
              write_instance,
              write_para,
              rerun=False,
              max_trials=1000,
              time_limit=10,
              exe_path=None):
    para_filename = "result/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    # instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    instance_filename = "result/" + dataset_name + "/" + dataset_name.lower() + "/" + instance_name + "." + dataset_name.lower()
    solution_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".sol"
    instance = instance[0] if len(instance) == 1 else instance
    # intermed_sol_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + "inter" + ".sol"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename,
                   "LKH", para_filename, max_trials=max_trials,
                   time_limit=time_limit, solution_filename=solution_filename)
        # vrp_size=,
        # intermed_solution_filename=intermed_sol_filename)
        with open(log_filename, "w") as f:
            # check_call(["./LKH", para_filename], stdout=f)
            check_call([str(exe_path), para_filename], stdout=f)
    try:
        res = read_results(log_filename, solution_filename)
    except FileNotFoundError:
        warnings.warn(f"LKH could not find solution for instance {instance_name}. See dir <outputs/visualisations/> "
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


def solve_NeuroLKH(dataset_name,
                   instance,
                   rp_instance,
                   instance_name,
                   candidate,
                   n_nodes_extend,
                   write_instance,
                   write_para,
                   write_candidate,
                   rerun=False,
                   max_trials=1000,
                   time_limit=10,
                   exe_path=None):
    para_filename = "result/" + dataset_name + "/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/" + dataset_name.lower() + "/" + instance_name + "." + dataset_name.lower()
    solution_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".sol"
    # print('write_para', write_para)
    # print('write_instance', write_instance)
    # print('WRITING INSTANCE IN SOLVE_NEUROLKH')
    # print('instance', instance)
    instance = instance[0] if len(instance) == 1 else instance
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename,
                   "NeuroLKH", para_filename, max_trials=max_trials, time_limit=time_limit,
                   solution_filename=solution_filename)
        write_candidate(dataset_name, instance_name, candidate, n_nodes_extend)
        with open(log_filename, "w") as f:
            check_call([str(exe_path), para_filename], stdout=f)
    try:
        res = read_results(log_filename, solution_filename)
    except FileNotFoundError:
        warnings.warn(f"LKH could not find solution for instance {instance_name}. See dir <outputs/visualisations/> "
                      f"for further information on this instance.")

        # if instance is not solved, set res to None and save instance gif to outputs/visualisations/
        res = None
        # plot instances which failed -> saved in visualisations dir
        SAVE_PATH = os.path.join(os.getcwd(), 'visualisations/failed_')
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        save_name = "c_dist_" + str(rp_instance.coords_dist) + "_d_dist_" + str(rp_instance.demands_dist) \
                    + "_depot_type_" + str(rp_instance.depot_type) + "_instance_" + instance_name
        view = Viewer(locs=rp_instance.coords, save_dir=SAVE_PATH, gif_naming=save_name)

    return res


def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen-CVRP":
        return generate_feat_cvrp(*args[1:])
    elif args[0] == "FeatGen-TSP":
        return generate_feat_tsp(*args[1:])


# , dataset_name, dataset[i], str(i), max_nodes, write_instance_cvrp, write_para_cvrp, exe_path)

def inference(
        problem: str,
        data: Union[List[CVRPInstance], List[TSPInstance], List[CVRPTWInstance]],
        method: str,
        model_path: str,
        exe_path: str,
        batch_size: int = 100,
        num_workers: int = 1,
        max_trials: int = 1000,
        device=torch.device("cpu"),
        int_prec: int = 10000,
        is_normalized: bool = True,
        TIME_LIMIT: int = None
):
    # re-adjust int_prec if data is original (unnormalized scale) to avoid wrongly scaled outputs
    int_prec = 1 if not is_normalized else int_prec
    assert method in ["NeuroLKH", "LKH", "NeuroLKH_M"]
    if method == "NeuroLKH_M":
        method = "NeuroLKH"  # just depends on loaded model checkpoint
    n_samples = len(data)
    if problem.upper() == "CVRP":
        dataset_name = "CVRP"
        # set up directories
        os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)
        get_features = get_features_cvrp
        write_instance = write_instance_cvrp
        write_para = write_para_cvrp
        write_candidate = write_candidate_cvrp
    elif problem.upper() == "CVRPTW":
        dataset_name = "CVRPTW"
        # set up directories
        os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/cvrptw", exist_ok=True)
        get_features = get_features_vrptw
        write_instance = write_instance_vrptw
        write_para = write_para_vrptw
        write_candidate = write_candidate_vrptw
        int_prec = 1 # for CVRPTW precision hard coded in Write_instance function
    else:
        dataset_name = "TSP"
        # set up directories
        os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True)
        os.makedirs("result/" + dataset_name + "/tsp", exist_ok=True)
        get_features = get_features_tsp # read_feat_tsp
        write_instance = write_instance_tsp
        write_para = write_para_tsp
        write_candidate = write_candidate_tsp
        # int_prec = 1

    rerun = True
    if num_workers > os.cpu_count():
        warnings.warn(f"num_workers > num logical cores! This can lead to "
                      f"decrease in performance if env is not IO bound.")

    # convert data to input format for LKH
    # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
    # in l2O-meta originally: 'np.ceil(1 + d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist()
    # dataset = [
    #     [
    #         d.coords.tolist(),
    #         np.ceil(d.node_features[1:, d.constraint_idx[0]] * int_prec).astype(int).tolist(),
    #         int(d.vehicle_capacity * int_prec)
    #     ] for d in data
    # ]
    dataset = prep_data_NeuroLKH(problem, data, is_normalized, int_prec)

    # run solver
    if method == "NeuroLKH":
        # note: in cvrptw n_nodes_extend is candidate 2
        candidate, n_nodes_extend, sgn_runtime, feat_runtime = get_features(data, dataset, dataset_name,
                                                                            model_path, batch_size,
                                                                            exe_path, num_workers, device)
        if TIME_LIMIT is not None:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    ("NeuroLKH", dataset_name, dataset[i], data[i],
                     str(data[i].instance_id), candidate[i], n_nodes_extend[i], write_instance, write_para,
                     write_candidate, rerun, max_trials, TIME_LIMIT, exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))
        else:
            with Pool(num_workers) as pool:
                results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                    ("NeuroLKH", dataset_name, dataset[i], data[i],
                     str(data[i].instance_id), candidate[i], n_nodes_extend[i], write_instance, write_para,
                     write_candidate, rerun, max_trials, data[i].time_limit, exe_path)
                    for i in range(len(dataset))
                ]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0

        if num_workers <= 1:
            if TIME_LIMIT is not None:
                results = list(tqdm.tqdm([
                    method_wrapper(("LKH", dataset_name, dataset[i], data[i], str(data[i].instance_id), write_instance,
                                    write_para, rerun, max_trials, TIME_LIMIT, exe_path))
                    for i in range(len(dataset))
                ], total=len(dataset)))
            else:
                results = list(tqdm.tqdm([
                    method_wrapper(("LKH", dataset_name, dataset[i], data[i], str(data[i].instance_id), write_instance,
                                    write_para, rerun, max_trials, data[i].time_limit, exe_path))
                    for i in range(len(dataset))
                ], total=len(dataset)))
        else:
            if TIME_LIMIT is not None:
                with Pool(num_workers) as pool:
                    results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                        ("LKH", dataset_name, dataset[i], data[i], str(data[i].instance_id), write_instance, write_para,
                         rerun, max_trials, TIME_LIMIT, exe_path)
                        for i in range(len(dataset))
                    ]), total=len(dataset)))
            else:
                with Pool(num_workers) as pool:
                    results = list(tqdm.tqdm(pool.imap(method_wrapper, [
                        ("LKH", dataset_name, dataset[i], data[i], str(data[i].instance_id), write_instance, write_para,
                         rerun, max_trials, data[i].time_limit, exe_path)
                        for i in range(len(dataset))
                    ]), total=len(dataset)))

    # s = 1000000.0  # precision hardcoded by authors in write_instance()
    # objs = np.asarray([np.asarray(r['objs']) for r in results if r is not None], dtype=object)
    objs = [np.array(r['objs']) / int_prec if r is not None else None for r in results]
    final_objs = [r["final_obj"] / int_prec if r is not None else None for r in results]
    # print('objs', objs)
    # objs = objs / s
    penalties = [r['penalties'] if r is not None else None for r in results]
    runtimes = [r['runtimes'] if r is not None else None for r in results]
    t_base = feat_runtime + sgn_runtime

    # read out trajectories
    trajectories = []
    count = 0
    # print('objs', objs)
    # print('runtimes', runtimes)
    for obj, rt in zip(objs, runtimes):

        if obj is not None and rt is not None:
            trj_obs, trj_rts, trj_iter = [], [], []
            assert len(obj) == len(rt)
            best_obj = np.inf
            for i in range(len(obj)):
                if obj[i] < best_obj:
                    best_obj = obj[i]
                    trj_obs.append(obj[i])
                    trj_rts.append(rt[i] + t_base)
                    trj_iter.append(i)
            # print('final_objs', final_objs)
            # print('trj_obs', trj_obs)
            # assert final_objs[count] == trj_obs[-1]
            trajectories.append({
                "iter": np.array(trj_iter),
                "time": trj_rts,  # np.array(trj_rts),
                "cost": trj_obs,  # np.array(trj_obs),
            })
            count += 1
        else:
            trajectories.append({
                "iter": None,
                "time": [],  # np.array(trj_rts),
                "cost": [],  # np.array(trj_obs),
            })
            count += 1

    if problem.upper() == "TSP":
        for r in results:
            r['solution'] = r['solution'][0]

    solutions = [
        RPSolution(
            solution=r['solution'] if r is not None else None,
            num_vehicles=r["num_vehicles"] if r is not None else None,
            run_time=r['runtimes'][-1] + feat_runtime + sgn_runtime if r is not None else None,
            cost=t["cost"][-1],
            method_internal_cost=t["cost"][-1],
            running_costs=t["cost"],  # r['running_costs'] if r is not None else None,
            running_times=t["time"],
            # [r['running_times'][t] + feat_runtime + sgn_runtime for t in range(len(r['running_times']))] if r is
            # not None else None,
            problem=dataset_name.upper(),
            instance=d
        ) for r, t, d in zip(results, trajectories, data)
    ]
    # print('solutions[0]', solutions[0])
    # results_by_trial = {}
    # trials = 1
    # print('objs.shape', objs.shape)
    # print('objs[0].shape', objs[0].shape)
    # print('objs[1].shape', objs[1].shape)
    # while trials <= objs.shape[1]:
    #    results_by_trial[trials] = {
    #        "objs": objs.mean(0)[trials - 1],
    #        "penalties": penalties.mean(0)[trials - 1],
    #        "runtimes": runtimes.sum(0)[trials - 1],
    #    }
    #    trials *= 10

    results_ = {
        "objs": objs,
        "penalties": penalties,
        "runtimes": runtimes,
    }
    # "results_by_trial": results_by_trial,
    return results_, solutions


def get_features_cvrp(data, dataset, dataset_name, model_path, batch_size, exe_path, num_workers, device):
    # convert data to input format for NeuroLKH
    n_samples = len(data)
    x = np.stack([d.coords for d in data])
    # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
    demand = np.stack([d.node_features[1:, d.constraint_idx[0]] for d in data])

    os.makedirs("result/" + dataset_name + "/featgen_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/feat", exist_ok=True)
    n_nodes = len(dataset[0][0]) - 1  # w/o depot
    # max_nodes = int(n_nodes * 1.15)

    N = x.shape[1] - 1
    try:
        K = CVRP_DEFAULTS[N][0]
    except KeyError:
        K = N
    # print('x', x)
    # print('demand', demand)
    max_nodes = N + K + 1
    # n_samples = args.n_samples
    n_neighbours = 20

       # compute features
    with Pool(num_workers) as pool:
        feats = list(tqdm.tqdm(pool.imap(method_wrapper, [("FeatGen-CVRP", dataset_name, dataset[i], str(i), max_nodes, write_instance_cvrp, write_para_cvrp, exe_path)
            for i in range(len(dataset))
        ]), total=len(dataset)))
    edge_index, n_nodes_extend, feat_runtime = list(zip(*feats))
    # print('n_nodes_extend:', n_nodes_extend)
    # consolidate features
    feat_runtime = np.sum(feat_runtime)
    feat_start_time = time.time()
    edge_index = np.concatenate(edge_index, 0)
    demand = np.concatenate([np.zeros([n_samples, 1]), demand, np.zeros([n_samples, max_nodes - n_nodes - 1])], -1)
    if demand.max() > 1.0:
        demand = demand / dataset[0][2]
    # print('max_nodes:', max_nodes)
    capacity = np.zeros([n_samples, max_nodes])
    capacity[:, 0] = 1
    capacity[:, n_nodes + 1:] = 1
    x = np.concatenate([x] + [x[:, 0:1, :] for _ in range(max_nodes - n_nodes - 1)], 1)
    node_feat = np.concatenate(
        [x, demand.reshape([n_samples, max_nodes, 1]), capacity.reshape([n_samples, max_nodes, 1])], -1)
    dist = node_feat[:, :, :2].reshape(n_samples, max_nodes, 1, 2) - node_feat[:, :, :2].reshape(n_samples, 1,
                                                                                                 max_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1))
    edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
    inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype="int")
    inverse_edge_index[
        np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(
        n_neighbours).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[
        np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
    feat_runtime += time.time() - feat_start_time
    feat_runtime /= n_samples  # per instance avg

    # load SGN
    net = SparseGCNModel(problem="cvrp")
    # net.cuda()
    net.to(device=device)
    if device != torch.device('cpu'):
        ckpt = torch.load(model_path)
    else:
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)  # load on to cpu
    net.load_state_dict(ckpt["model"])

    # do inference
    sgn_start_time = time.time()
    with torch.no_grad():
        candidate = infer_SGN_cvrp(net, node_feat, edge_index, edge_feat, inverse_edge_index,
                                   batch_size=batch_size, device=device)
    sgn_runtime = time.time() - sgn_start_time
    sgn_runtime /= n_samples  # per instance avg
    os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
    # None because vrptw returns two candidate variables
    return candidate, n_nodes_extend, sgn_runtime, feat_runtime
    # candidate, n_nodes_extend, sgn_runtime, feat_runtime


def generate_feat_cvrp(dataset_name,
                       instance,
                       instance_name,
                       max_nodes,
                       write_instance,
                       write_para,
                       exe_path=None):
    para_filename = "result/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    # print('para_filename', para_filename)
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    feat_filename = "result/" + dataset_name + "/feat/" + instance_name + ".txt"
    write_instance(instance, instance_name, instance_filename)
    write_para(dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call([str(exe_path), para_filename], stdout=f)
    return read_feat_cvrp(feat_filename, max_nodes)


def get_features_tsp(data, dataset, dataset_name, model_path, batch_size, exe_path, num_workers, device):
    n_samples = len(data)
    print('n_samples', n_samples)
    x = np.stack([d.coords for d in data])
    print('len(dataset', len(dataset))
    print('len(dataset[0]', len(dataset[0]))
    print('len(dataset[1]', len(dataset[1]))
    dataset = [data[0] for data in dataset]
    print('len(dataset', len(dataset))
    os.makedirs("result/" + dataset_name + "/featgen_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/feat", exist_ok=True)
    n_node = len(dataset[0])

    N = x.shape[1] - 1

    # str(i),

    # compute features
    with Pool(num_workers) as pool:
        feats = list(tqdm.tqdm(pool.imap(method_wrapper, [
            (
            "FeatGen-TSP", dataset_name, dataset[i], str(i), N, write_instance_tsp, write_para_tsp, exe_path)
            for i in range(len(dataset))
        ]), total=len(dataset)))
    # feats = list(zip(*feats))
    edge_index, edge_feat, inverse_edge_index, feat_runtime = list(zip(*feats))

    feat_runtime = np.sum(feat_runtime)
    feat_start_time = time.time()
    edge_index = np.concatenate(edge_index)
    edge_feat = np.concatenate(edge_feat)
    inverse_edge_index = np.concatenate(inverse_edge_index)
    feat_runtime += time.time() - feat_start_time
    feat_runtime /= n_samples

    net = SparseGCNModel(problem="tsp")
    net.to(device=device)
    if device != torch.device('cpu'):
        ckpt = torch.load(model_path)
    else:
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)  # load on to cpu
    net.load_state_dict(ckpt["model"])

    # print('dataset', dataset)
    # print('dataset.shape', np.array(dataset).shape)
    # print('dataset.size', np.array(dataset).size)
    # do inference
    sgn_start_time = time.time()
    with torch.no_grad():
        candidate_Pi = infer_SGN_tsp(net, np.array(dataset), edge_index, edge_feat, inverse_edge_index,
                                    batch_size=batch_size)
    sgn_runtime = time.time() - sgn_start_time
    sgn_runtime /= n_samples

    #n_node = len(dataset[0])
    candidate = candidate_Pi[:, :n_node * 5].reshape(-1, n_node, 5)
    pi = candidate_Pi[:, n_node * 5:].reshape(-1, n_node)
    os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/pi", exist_ok=True)

    return candidate, pi, sgn_runtime, feat_runtime

def generate_feat_tsp(dataset_name,
                      instance,
                      instance_name,
                      max_nodes,
                      write_instance,
                      write_para,
                      exe_path=None):

    para_filename = "result/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    instance_filename = "result/" + dataset_name + "/tsp/" + instance_name + ".tsp"
    feat_filename = "result/" + dataset_name + "/feat/" + instance_name + ".txt"
    # print('WRITING INSTANCE IN GENERATE FEAT TSP')
    write_instance(instance, instance_name, instance_filename)
    write_para(dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call([str(exe_path), para_filename], stdout=f)
    return read_feat_tsp(feat_filename)



# def write_para_cvrp(dataset_name,
#                     instance_name,
#                     instance_filename,
#                     method,
#                     para_filename,
#                     # vrp_size,
#                     max_trials=1000,
#                     time_limit=10,
#                     seed=1234,
#                     solution_filename=None):
#     with open(para_filename, "w") as f:
#         f.write("PROBLEM_FILE = " + instance_filename + "\n")
#         f.write("MAX_TRIALS = " + str(max_trials) + "\n")
#         f.write("SPECIAL\nRUNS = 10\n")
#         # f.write("SALESMEN = " + str(vrp_size) + "\n")
#         f.write("MTSP_MIN_SIZE = 0\n")
#         f.write("SEED = " + str(seed) + "\n")
#         f.write("TIME_LIMIT = " + str(time_limit) + "\n")
#         f.write("TRACE_LEVEL = 1\n")
#         if method == "NeuroLKH":
#             f.write("SUBGRADIENT = NO\n")
#             f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
#         elif method == "FeatGenerate":
#             f.write("GerenatingFeature\n")
#             f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
#             f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
#             f.write("MAX_CANDIDATES = 20\n")
#         else:
#             assert method == "LKH"
#         if solution_filename is not None:
#             f.write(f"TOUR_FILE = {solution_filename}\n")  # to write best solution to file
#             # f.write(f"SINTEF_SOLUTION_FILE = {intermed_solution_filename}\n")
#         # f.write(f"OUTPUT_TOUR_FILE = {solution_filename}\n")
