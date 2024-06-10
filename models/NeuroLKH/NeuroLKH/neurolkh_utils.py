import os
import warnings
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
from models.NeuroLKH.NeuroLKH.net.sgcn_model import SparseGCNModel
import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile


### VRPTW ###


def write_instance_vrptw(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        x = instance[0]
        demand = instance[1]
        capacity = instance[2]
        a = instance[3]
        b = instance[4]
        service_time = instance[5]
        # n_nodes = x.shape[0]
        n_nodes = len(instance[0])  #  - 1
        # print('len(instance[0])', len(instance[0]))
        # print('n_nodes', n_nodes)

        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRPTW\n")
        f.write("VEHICLES : 20\n") # changed
        # f.write("VEHICLES : " + str(len(instance[0]) - 10) + "\n")
        f.write("CAPACITY : " + str(capacity) + "\n")
        f.write("SERVICE_TIME : " + str(service_time * 1000000) + "\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\nEDGE_WEIGHT_TYPE : EUC_2D\n")

        f.write("NODE_COORD_SECTION\n")
        for l in range(n_nodes):
            f.write(" " + str(l + 1) + " " + str(x[l][0] * 1000000)[:15] + " " + str(x[l][1] * 1000000)[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for l in range(n_nodes - 1):
            f.write(str(l + 2) + " " + str(int(demand[l])) + "\n")
        f.write("TIME_WINDOW_SECTION\n")
        f.write("1 0 10000000\n")
        for l in range(n_nodes - 1):
            f.write(str(l + 2) + " " + str(int(a[l] * 1000000)) + " " + str(int(b[l] * 1000000)) + "\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def write_para_vrptw(dataset_name, instance_name, instance_filename, method, para_filename,
                     max_trials=1000, time_limit=10, seed=1234, solution_filename=None):
    # print('solution_filename', solution_filename)
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        # f.write("VEHICLES = 100\n")  # added
        # f.write("MTSP_MIN_SIZE = 0\n")    # added
        f.write("SEED = " + str(seed) + "\n")
        f.write("TIME_LIMIT = " + str(time_limit) + "\n")   # added
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"
        if solution_filename is not None:
            # print('naming sol file, solution_filename is', solution_filename)
            f.write(f"TOUR_FILE = {solution_filename}\n")  # to write best solution to file


def write_candidate_vrptw(dataset_name, instance_name, candidate1, candidate2):
    n_node = candidate1.shape[0] - 1
    # print('n_node', n_node)
    candidate1 = candidate1.astype("int")
    candidate2 = candidate2.astype("int")

    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str((n_node + 20) * 2) + "\n")
        line = "1 0 5 " + str(1 + n_node + 20) + " 0"
        for _ in range(4):
            line += " " + str(2 * n_node + 2 * 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1) + " 0 5 " + str(j + 1 + n_node + 20) + " 1"
            for _ in range(4):
                line += " " + str(candidate2[j, _] + 1 + n_node + 20) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 1 + 1 + j) + " 0 5 " + str(n_node + 1 + 1 + j + n_node + 20) + " 0 " + str(
                1 + n_node + 20) + " 1"
            for _ in range(3):
                line += " " + str(n_node + 2 + _ + n_node + 20) + " 1"
            f.write(line + "\n")

        line = str(1 + n_node + 20) + " 0 5 1 0"
        for _ in range(4):
            line += " " + str(n_node + 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1 + n_node + 20) + " 0 5 " + str(j + 1) + " 1"
            for _ in range(4):
                line += " " + str(candidate1[j, _] + 1) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 2 + j + n_node + 20) + " 0 5 " + str(n_node + 2 + j) + " 0"
            for _ in range(4):
                line += " " + str(n_node + 20 - _) + " 1"
            f.write(line + "\n")
        f.write("-1\nEOF\n")




# get_features_vrptw


# data, dataset, dataset_name, model_path, batch_size, exe_path, num_workers, device


def get_features_vrptw(data, dataset, dataset_name, model_path, batch_size, exe_path, num_workers, device):
        feat_start_time = time.time()
        n_neighbours = 20
        # n_samples = dataset["loc"].shape[0]
        # n_nodes = dataset["loc"].shape[1] - 1
        # demand = np.concatenate([np.zeros((n_samples, 1)), dataset['demand'] / 50], -1)
        # start = np.concatenate([np.zeros((n_samples, 1)), dataset['start'] / 10], -1)
        # end = np.concatenate([np.ones((n_samples, 1)), dataset['end'] / 10], -1)
        # capacity = np.concatenate([np.ones((n_samples, 1)), np.zeros((n_samples, n_nodes))], -1)
        # x = dataset['loc']
        n_samples = len(data)
        n_nodes = data[0].coords.shape[0] - 1  # w/o depot
        # print('n_nodes', n_nodes)
        dem = np.stack([(dat.node_features[1:, 3] * dat.original_capacity) for dat in data])
        # print('dem.shape', dem.shape)
        demand = np.concatenate([np.zeros((n_samples, 1)), dem / 50], -1)
        # print('demand.shape in get_features', demand.shape)
        st = np.stack([dat.tw[1:, 0] for dat in data])
        start = np.concatenate([np.zeros((n_samples, 1)), st], -1)
        # print('start.shape in get_features', start.shape)
        en = np.stack([dat.tw[1:, 1] for dat in data])
        end = np.concatenate([np.ones((n_samples, 1)), en], -1)
        # print('end.shape in get_features', end.shape)
        capacity = np.concatenate([np.ones((n_samples, 1)), np.zeros((n_samples, n_nodes))], -1)
        x = np.stack([d.coords for d in data])

        node_feat = np.concatenate([x,
                                    demand.reshape(n_samples, n_nodes + 1, 1),
                                    start.reshape(n_samples, n_nodes + 1, 1),
                                    end.reshape(n_samples, n_nodes + 1, 1),
                                    capacity.reshape(n_samples, n_nodes + 1, 1)], -1)
        n_nodes += 1
        dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
        edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]

        inverse_edge_index = -np.ones(shape=[n_samples, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_neighbours
        inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        feat_runtime = time.time() - feat_start_time

        net = SparseGCNModel(problem="cvrptw")
        net.to(device)
        net.train()
        saved = torch.load(model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate1, candidate2 = infer_SGN_vrptw(net, node_feat, edge_index, edge_feat,
                                                     inverse_edge_index, batch_size=batch_size)

        sgn_runtime = time.time() - sgn_start_time
        os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
        return candidate1, candidate2, sgn_runtime, feat_runtime


def infer_SGN_vrptw(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index,
                    batch_size=100):
    candidate1 = []
    candidate2 = []
    for i in trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1, 1)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor),
                                      requires_grad=False).view(batch_size, -1)
        y_edges1, y_edges2, _, _, y_nodes = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index,
                                                                 None, None, None, 20)

        y_edges1 = y_edges1.detach().cpu().numpy()
        y_edges1 = y_edges1[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges1 = np.argsort(-y_edges1, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges1.shape[1], 20)
        candidate_index = edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges1.shape[1]).reshape(1, -1, 1), y_edges1]
        candidate1.append(candidate_index[:, :, :20])

        y_edges2 = y_edges2.detach().cpu().numpy()
        y_edges2 = y_edges2[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges2 = np.argsort(-y_edges2, -1)
        candidate_index = edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges2.shape[1]).reshape(1, -1, 1), y_edges2]
        candidate2.append(candidate_index[:, :, :20])
    candidate1 = np.concatenate(candidate1, 0)
    candidate2 = np.concatenate(candidate2, 0)
    return candidate1, candidate2


### CVRP ###

def write_instance_cvrp(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        n_nodes = len(instance[0]) - 1
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        f.write("VEHICLES : " + str(len(instance[0]) - 10) + "\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : " + str(instance[2]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        # s = 1000000
        for i in range(n_nodes + 1):
            f.write(" " + str(i + 1) + " " + str(instance[0][i][0])[:15] + " " + str(instance[0][i][1])[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2) + " " + str(instance[1][i]) + "\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def write_para_cvrp(dataset_name,
                    instance_name,
                    instance_filename,
                    method,
                    para_filename,
                    # vrp_size,
                    max_trials=1000,
                    time_limit=10,
                    seed=1234,
                    solution_filename=None):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        # f.write("SALESMEN = " + str(vrp_size) + "\n")
        f.write("MTSP_MIN_SIZE = 0\n")
        f.write("SEED = " + str(seed) + "\n")
        f.write("TIME_LIMIT = " + str(time_limit) + "\n")
        f.write("TRACE_LEVEL = 1\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"
        if solution_filename is not None:
            f.write(f"TOUR_FILE = {solution_filename}\n")  # to write best solution to file
            # f.write(f"SINTEF_SOLUTION_FILE = {intermed_solution_filename}\n")
        # f.write(f"OUTPUT_TOUR_FILE = {solution_filename}\n")


def infer_SGN_cvrp(net,
                   dataset_node_feat,
                   dataset_edge_index,
                   dataset_edge_feat,
                   dataset_inverse_edge_index,
                   batch_size=100,
                   device=torch.device("cpu"),
                   ):
    candidate = []
    assert dataset_edge_index.shape[0] % batch_size == 0, f"Dataset size needs to be divisible by batch size to run SGN"
    for i in tqdm.trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.FloatTensor),
                             requires_grad=False).to(device)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.FloatTensor),
                             requires_grad=False).view(batch_size, -1, 1).to(device)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.FloatTensor),
                              requires_grad=False).view(batch_size, -1).to(device)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.FloatTensor),
                                      requires_grad=False).view(batch_size, -1).to(device)
        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    return candidate


### TSP ###

def write_para_tsp(dataset_name, instance_name, instance_filename, method, para_filename,
                   time_limit=10, max_trials=1000, seed=1234, solution_filename=None):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")
        f.write("TIME_LIMIT = " + str(time_limit) + "\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
            f.write("Pi_FILE = result/" + dataset_name + "/pi/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("Feat_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
        else:
            assert method == "LKH"
        if solution_filename is not None:
            f.write(f"TOUR_FILE = {solution_filename}\n")  # to write best solution to file

### MISC ###
def prep_data_NeuroLKH(problem, data, is_normalized, int_prec):
    # n_nodes = len(instance[0]) - 1
    if problem.upper() == "CVRPTW":
        if is_normalized:
            # print('d.tw[:, 0]', data[0].tw[:, 0])
            # print('d.service_time', data[0].service_time)
            # print('d.original_capacity', data[0].original_capacity)
            # print('d.node_features[:2, :] * d.original_capacity',
            #       data[0].node_features[:2, :] * data[0].original_capacity)
            # unnorm data again
            # dataset = [
            #     [
            #         (d.coords * int_prec).astype(int).tolist(),
            #         np.ceil(d.node_features[1:, 3] * int_prec).astype(int).tolist(),
            #         int(d.vehicle_capacity * int_prec),
            #         d.tw[:, 0],
            #         d.tw[:, 1],
            #         d.service_time
            #     ] for d in data
            # ]
            dataset = [
                [
                    d.coords,
                    d.node_features[1:, 4],
                    d.original_capacity,
                    d.tw[1:, 0],
                    d.tw[1:, 1],
                    d.service_time
                ] for d in data
            ]
        else:
            dataset = [
                [
                    (d.coords).astype(int).tolist(),
                    d.node_features[1:, 4],
                    int(d.original_capacity),
                    d.tw[1:, 0],
                    d.tw[1:, 1],
                    d.service_time
                ] for d in data
            ]

    elif problem.upper() == "CVRP":
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
    elif problem.upper() == "TSP":
        if is_normalized:
            dataset = [
                [
                    (d.coords * int_prec).astype(int).tolist()
                ] for d in data
            ]
            # print('dataset[0]', dataset[0])
        else:
            dataset = [
                [
                    d.coords.tolist()
                ] for d in data
            ]
    else:
        # dataset = None
        warnings.warn(f"Problem {problem.upper()} not implemented - Must be in [TSP, CVRP, CVRPTW]")
        raise NotImplementedError
    return dataset

def min_needed_vs(uchoa_dat):
    min_needed_bef = 0
    nr_vs_needed = min_needed_bef
    for i in range(len(uchoa_dat)):
        sum_dem = sum(uchoa_dat[i].node_features[:, -1])
        capa = uchoa_dat[i].vehicle_capacity
        # print(sum_dem)
        min_needed_vs = sum_dem / capa
        if min_needed_vs > min_needed_bef:
            nr_vs_needed = min_needed_vs
            min_needed_bef = min_needed_vs
    print('final needed vs:', nr_vs_needed)
