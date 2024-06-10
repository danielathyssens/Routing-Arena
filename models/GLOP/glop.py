import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Tuple
from omegaconf import DictConfig

from formats import CVRPInstance, TSPInstance, RPSolution

from models.GLOP.GLOP.utils import load_model
from torch.utils.data import DataLoader
import time
from models.GLOP.GLOP.utils.functions import reconnect
from models.GLOP.GLOP.utils.functions import load_problem
import pprint as pp
from models.GLOP.GLOP.utils.insertion import random_insertion
from models.GLOP.GLOP.heatmap.cvrp.infer import load_partitioner
from models.GLOP.GLOP.heatmap.cvrp.inst import sum_cost
from models.GLOP.GLOP.nets.attention_local import AttentionModel


def eval_model(data: Union[List[TSPInstance], List[CVRPInstance]],
               problem: str,
               revisers: List[AttentionModel],
               time_limit: int,
               tester_cfg: DictConfig,
               device: str,
               batch_size: int = 1):
    dataset_ = prep_data_glop('tsp', data, offset=0) if problem.lower() == 'tsp' \
        else prep_data_glop('cvrp', data, offset=0)

    val_size = tester_cfg.val_size if tester_cfg.val_size is not None else len(data)
    problem_type = tester_cfg.problem_type
    dataset, pi_all, n_tsps_per_route_lst, results = None, None, None, None
    start = time.time()
    if problem_type == "tsp":
        dataset = dataset_
        # To decode idx:  # added
        global tsp_data
        # print('dataset_[0]', dataset_[0])
        tsp_data = dataset_   # load_dataset(dataset_path)
        if tester_cfg.problem_size <= 100:
            if tester_cfg.width >= 4:
                tester_cfg.width //= 4
                tester_cfg.tsp_aug = True
            else:
                tester_cfg.tsp_aug = False

        # global orders
        orders = [torch.randperm(tester_cfg.problem_size) for i in range(tester_cfg.width)]
        pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for instance in
                  dataset]  # instance: (p_size, 2)
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(len(orders), val_size,
                                                                         tester_cfg.problem_size)  # width, val_size, p_size
        # print('orders', orders)
    elif problem_type == 'cvrp':

        # To decode idx:  # added
        global cvrp_data
        cvrp_data = dataset_   # load_dataset(dataset_path)

        dataset, n_tsps_per_route_lst = init_cvrp(dataset_, tester_cfg, val_size, device)
        # print('len n_tsps_per_route_lst', len(n_tsps_per_route_lst))
        # print('len(dataset)', len(dataset))
        # print('len(dataset[0])', len(dataset[0]))
        # print('dataset[0][0])', dataset[0][0])
        # print('dataset[0][0][:10]', dataset[0][0][:10])
        tester_cfg.eval_batch_size = 1
    elif problem_type == 'cvrplib':
        partitioner = load_partitioner(2000, device, tester_cfg.ckpt_path_partitioner, 300, 6)
        dataset, n_tsps_per_route_lst = init_cvrp(dataset_, tester_cfg, val_size, device, partitioner)
        tester_cfg.eval_batch_size = 1
    else:
        warnings.warn(f"problem-type {problem} not known...")
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=tester_cfg.eval_batch_size)

    problem_ = load_problem('tsp')
    get_cost_func = lambda input_, pi: problem_.get_costs(input_, pi, return_local=True)

    results, total_runtime = eval_loop(problem, start, device, dataloader, revisers, get_cost_func,
                                       tester_cfg, pi_all, n_tsps_per_route_lst)


    sols_internal, costs = solution_post_processing(problem, results, total_runtime, problem_type, dataset_, device)
    # print('len(sols_internal)', len(sols_internal))
    # # overwrite for testing
    # sols_internal = [torch.tensor(sol_hard_coded_1, dtype=torch.int), torch.tensor(sol_hard_coded_2, dtype=torch.int)]
    return {}, make_RPSolution(problem, sols_internal, costs, total_runtime, data)


def eval_loop(problem: str, start, device, dataloader, revisers, get_cost_func,
              tester_cfg, pi_all=None, n_tsps_per_route_lst=None):
    results, costs_revised = [], None
    problem_type = tester_cfg.problem_type.lower()
    for batch_id, batch in tqdm(enumerate(dataloader), disable=tester_cfg.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        with torch.no_grad():
            if problem_type in ['tsp', 'pctsp']:
                p_size = batch.size(1)
                # print('p_size', p_size)
                # print('tester_cfg.width', tester_cfg.width)
                batch = batch.repeat(tester_cfg.width, 1, 1)  # (1,1,1) for pctsp
                # print('batch.shape', batch.shape)
                pi_batch = pi_all[:, batch_id * tester_cfg.eval_batch_size: (batch_id + 1) * tester_cfg.eval_batch_size,
                           :].reshape(
                    -1, p_size)
                # print('pi_batch.shape, ', pi_batch.shape)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1, 1, 2))
                # print('seed.shape, ', seed.shape)
            elif problem_type in ['cvrp', 'cvrplib']:
                batch = batch.squeeze()  # (n_subTSPs_for_width_routes, max_seq_len, 2)
                n_subTSPs, max_seq_len, _ = batch.shape
                n_tsps_per_route = n_tsps_per_route_lst[batch_id]
                assert sum(n_tsps_per_route) == n_subTSPs
                tester_cfg.eval_batch_size = n_subTSPs
                order = torch.arange(max_seq_len)
                pi_batch = [random_insertion(instance, order)[0] for instance in batch]
                pi_batch = torch.tensor(np.array(pi_batch).astype(np.int64))
                assert pi_batch.shape == (n_subTSPs, max_seq_len)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1, 1, 2))
                assert seed.shape == (n_subTSPs, max_seq_len, 2)
            else:
                raise NotImplementedError

            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2,
                                                                                                              dim=1)
            if problem_type in ['tsp', 'pctsp']:
                cost_ori, _ = cost_ori.reshape(-1, tester_cfg.eval_batch_size).min(0)  # width, bs
                # print('cost_ori.shape', cost_ori.shape)
                avg_cost = cost_ori.mean().item()
            elif problem_type in ['cvrp', 'cvrplib']:
                avg_cost = sum_cost(cost_ori, n_tsps_per_route).min()
            else:
                raise NotImplementedError

            prob_size = tester_cfg.problem_size if tester_cfg.problem_size is not None else len(batch[0])
            if prob_size <= 100 and problem_type == 'tsp' and tester_cfg.tsp_aug:
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)

            tours, costs_revised = reconnect(
                get_cost_func=get_cost_func,
                batch=seed,
                opts=tester_cfg.reconnect_cfg,
                revisers=revisers,
            )
            # print('tours shape', tours.shape)

        if problem_type in ['cvrp', 'cvrplib']:
            assert costs_revised.shape == (n_subTSPs,)
            costs_revised, best_partition_idx = sum_cost(costs_revised, n_tsps_per_route).min(dim=0)
            subtour_start = sum(n_tsps_per_route[:best_partition_idx])
            tours = tours[subtour_start: subtour_start + n_tsps_per_route[best_partition_idx]]
            assert tours.shape == (n_tsps_per_route[best_partition_idx], max_seq_len, 2)

            # Get tour idx  # added
            idx = get_idx(batch_id, tours, tester_cfg, device, problem)
            # print('\n idx', idx)

            tours = tours.reshape(-1, 2)

            results.append((avg_cost, costs_revised, None, idx))
        else:
            # print('tours[0, :5]', tours[0, :5])
            # print('tours.shape', tours.shape)
            idx = get_idx(batch_id, tours, tester_cfg, device, problem)
            # print('idx', idx)

            results.append((avg_cost, costs_revised, None, idx))

    duration = time.time() - start

    return results, duration


def prep_data_glop(problem: str, dat: List, offset=0):
    """Transfer TSPInstance List to glop-type input data"""
    if problem == 'tsp':
        return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    elif problem == 'cvrp':
        # this is data prep for glop_neural --> when using lkh as sub-solver is maybe different...
        return [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]


def make_cvrp_instance(instance: CVRPInstance):
    depot = instance.coords[0].tolist()
    loc = instance.coords[1:].tolist()
    demand = instance.node_features[1:, instance.constraint_idx[0]].tolist()  # uses unnormalized demands
    # capa = instance.original_capacity
    return depot, loc, [int(d) for d in demand], float(instance.original_capacity)


# Transform solution returned from POSTPROCESSING to List[List]
def sol_to_list(sol: Union[torch.tensor, np.ndarray, List], depot_idx: int = 0) -> List[List]:
    sol_lst, lst = [], [0]
    for e in sol:
        if e == depot_idx:
            if lst[-1] == 0:
                pass
            elif len(lst) > 1:
                lst.append(0)
                sol_lst.append(lst)
                lst = [0]
        else:
            if isinstance(e, torch.Tensor):
                lst.append(e.item())
            else:
                lst.append(e)
    # print(f'sol_lst: {sol_lst}')
    return sol_lst


def make_RPSolution(problem: str, sols: Union[List[torch.Tensor], torch.Tensor], costs, total_duration, rp_data):
    # transform solution torch.Tensor -> List[List]
    if problem == 'cvrp':
        sol_list = [sol_to_list(sol_) for sol_ in sols]
    else:
        sol_list = [sol.tolist() for sol in sols]
    # print('sol_list', sol_list)
    times = [total_duration / len(sols)] * len(sols)
    return [
        RPSolution(
            solution=sol,
            cost=c.item() if isinstance(c, torch.Tensor) else c,
            num_vehicles=len(sol) if problem.upper() == 'CVRP' else len([sol]),
            run_time=t.item() if isinstance(t, torch.Tensor) else t,  # float(t[:-1]),
            problem=problem,
            instance=inst,
            method_internal_cost=c.item() if isinstance(c, torch.Tensor) else c
        )
        for sol, c, t, inst in zip(sol_list, costs, times, rp_data)
    ]


def solution_post_processing(problem: str, results: Tuple, duration: float, problem_type,
                             data: torch.Tensor, device=None):
    costs, costs_revised, costs_revised_with_penalty, tours = zip(*results)
    costs = torch.tensor(costs)
    # print('tours', tours)
    # print('len(tours)', len(tours))
    if problem_type.lower() in ['cvrp', 'cvrplib']:
        costs_revised = torch.stack(costs_revised)
    else:
        costs_revised = torch.cat(costs_revised, dim=0)

    if problem_type.lower() == 'pctsp':
        costs_revised_with_penalty = torch.cat(costs_revised_with_penalty, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(),
                                                  (2 * torch.std(costs_revised) / math.sqrt(
                                                      len(costs_revised))).item()))
    if problem_type.lower() == 'pctsp':
        print("Average cost_revised with penalty: {} +- {}".format(costs_revised_with_penalty.mean().item(),
                                                                   (2 * torch.std(
                                                                       costs_revised_with_penalty) / math.sqrt(
                                                                       len(costs_revised_with_penalty))).item()))
    print("Total duration: {}".format(duration))

    # if problem.lower() != 'cvrp':
    #     print('tours before cat', tours)
    #     tours = torch.cat(tours, dim=0)
    #     print('tours in post', tours)
    #     return tours, costs_revised
    # else:
    if problem_type.lower() == 'cvrp':
        # post-process tours from coordinates to node-indices
        final_tours = [torch.tensor(tour) for tour in tours]
    else:
        final_tours = tours

    return final_tours, costs_revised


# taken from GLOP/cvrp.py --> without datapath



def concat_list(depot_coor, coors, demand, opts, device):
    coor = torch.cat([torch.tensor(depot_coor, device=device).unsqueeze(0),
                      torch.tensor(coors, device=device)], dim=0)  # 1+p_size, 2
    demand = torch.cat([torch.zeros((1), device=device),
                        torch.tensor(demand, device=device)])  # 1+p_size
    return coor, demand


def init_cvrp(data_cvrp, opts, val_size, device,
              partitioner=None):  # for cvrplib, partioner is not None, and depth is set to 6, k_sparse to 300
    from models.GLOP.GLOP.heatmap.cvrp.infer import infer, load_partitioner
    from models.GLOP.GLOP.heatmap.cvrp.sampler import Sampler
    from models.GLOP.GLOP.heatmap.cvrp.inst import trans_tsp
    is_cvrplib = True if opts.partitioner is not None else False
    k_sparse = 300 if is_cvrplib else None

    # data = load_dataset(path)
    greedy_mode = True if opts.n_partition == 1 else False
    prob_size = opts.problem_size if opts.problem_size is not None else len(data_cvrp[0][1])
    print('prob_size', prob_size)
    partitioner = load_partitioner(prob_size, device,
                                   opts.ckpt_path_partitioner) if partitioner is None else partitioner
    dataset = []
    n_tsps_per_route_lst = []
    for inst_id, inst in enumerate(data_cvrp[:val_size]):
        # print('val_size', val_size)
        # print('len(inst)', len(inst))
        # print('inst', inst)
        depot_coor, coors, demand, capacity = inst
        coors, demand = concat_list(depot_coor, coors, demand, opts, device)
        heatmap = infer(partitioner, coors, demand, capacity, k_sparse, is_cvrplib)
        sampler = Sampler(demand, heatmap, capacity, opts.n_partition, 'cpu')
        routes = sampler.gen_subsets(require_prob=False, greedy_mode=greedy_mode) # n_partition, max_len

        # You can check the capacity constraint by the following code --> added
        sum_demands = 0
        for i in range(1, routes.shape[1]):
            sum_demands += demand[routes[0][i]]
            if routes[0][i] == 0:
                assert sum_demands <= capacity, f"capacity: {capacity}, sum_demands: {sum_demands}"
                sum_demands = 0

        assert routes.size(0) == opts.n_partition
        tsp_insts, n_tsps_per_route = trans_tsp(coors.cpu(), routes)
        assert tsp_insts.size(0) == sum(n_tsps_per_route)
        dataset.append(tsp_insts)
        n_tsps_per_route_lst.append(n_tsps_per_route)
    return dataset, n_tsps_per_route_lst

def get_idx(batch_id, tours, opts, device, problem):
    "Get solutions in idx format for CVRP"
    if problem.lower() == 'cvrp':
        _depot_coor, _coors, _demand, capacity = cvrp_data[batch_id]  # batch size is 1
        coors, demands = concat_list(_depot_coor, _coors, _demand, opts, device)
        n_tsps, max_seq_len, _ = tours.shape
        cvrp_idx_solution = []
        for tsp_idx in range(n_tsps):
            tsp_idx_solution = []
            sum_demand = 0
            for node_idx in range(max_seq_len):
                node = tours[tsp_idx, node_idx]
                match = torch.isclose(node, coors, atol=1e-5).all(dim=1)
                idx = torch.nonzero(match).item()
                tsp_idx_solution.append(idx)
                sum_demand += demands[idx]
            # print('tsp_idx_solution list: ', tsp_idx_solution)

            first_depot_idx = tsp_idx_solution.index(0)
            # print('first_depot_idx in n_tsps loop', first_depot_idx)
            last_depot_idx = len(tsp_idx_solution) - tsp_idx_solution[::-1].index(0)
            if first_depot_idx == 0 and last_depot_idx == len(tsp_idx_solution):
                rolled_tour = [0] + [idx for idx in tsp_idx_solution if idx != 0] + [0]
            else:
                # print('last_depot_idx in n_tsps loop', last_depot_idx)
                rolled_tour = [0] + tsp_idx_solution[last_depot_idx:] + tsp_idx_solution[:first_depot_idx]

            # Here we can check the total demand of each subTSP solution (sub-route)
            assert sum_demand <= capacity, f"sum_demand: {sum_demand}, capacity: {capacity}"
            cvrp_idx_solution.extend(rolled_tour)
        cvrp_idx_solution.append(0)

        # Compute the total length of the solution
        total_length = 0
        for i in range(len(cvrp_idx_solution) - 1):
            total_length += torch.norm(coors[cvrp_idx_solution[i]] - coors[cvrp_idx_solution[i + 1]])
        # print('\ntotal_length.item()', total_length.item())
        # print('\ncvrp_idx_solution', cvrp_idx_solution)

        return cvrp_idx_solution
    if problem == 'tsp':
        # print('batch_id', batch_id)
        coors = tsp_data[batch_id].to(device)
        tsp_idx_solution = []
        for node_idx in range(tours.shape[1]):
            node = tours[0, node_idx]
            match = torch.isclose(node, coors, atol=1e-5).all(dim=1)
            print('match', match)
            idx = torch.nonzero(match).item()
            tsp_idx_solution.append(idx)
        # print('tsp_idx_solution', tsp_idx_solution)
        return torch.tensor(tsp_idx_solution)





def init_cvrp_old(data_cvrp, opts, val_size, device,
              partitioner=None):  # for cvrplib, partioner is not None, and depth is set to 6, k_sparse to 300
    from models.GLOP.GLOP.heatmap.cvrp.infer import infer, load_partitioner
    from models.GLOP.GLOP.heatmap.cvrp.sampler import Sampler
    from models.GLOP.GLOP.heatmap.cvrp.inst import trans_tsp
    is_cvrplib = True if opts.partitioner is not None else False
    k_sparse = 300 if is_cvrplib else None

    # data = load_dataset(path)
    greedy_mode = True if opts.n_partition == 1 else False
    partitioner = load_partitioner(opts.problem_size, device,
                                   opts.ckpt_path_partitioner) if partitioner is None else partitioner
    dataset = []
    n_tsps_per_route_lst = []
    for inst_id, inst in enumerate(data_cvrp[:val_size]):
        # print('val_size', val_size)
        # print('len(inst)', len(inst))
        # print('inst', inst)
        depot_coor, coors, demand, capacity = inst
        coors, demand = concat_list(depot_coor, coors, demand, opts, device)
        heatmap = infer(partitioner, coors, demand, capacity, k_sparse, is_cvrplib)
        # print('heatmap.size()', heatmap.size())
        sampler = Sampler(demand, heatmap, capacity, opts.n_partition, 'cpu')
        routes = sampler.gen_subsets(require_prob=False, greedy_mode=greedy_mode)  # n_partition, max_len
        # print('routes.size()', routes.size())
        assert routes.size(0) == opts.n_partition
        tsp_insts, n_tsps_per_route = trans_tsp(coors.cpu(), routes)
        # print('tsp_insts.size()', tsp_insts.size())
        assert tsp_insts.size(0) == sum(n_tsps_per_route)
        dataset.append(tsp_insts)
        n_tsps_per_route_lst.append(n_tsps_per_route)
    return dataset, n_tsps_per_route_lst

# sol_hard_coded_1 = [457, 539, 980, 972, 160, 137, 354, 691, 336, 25, 153, 114, 804, 709, 558, 6, 701, 503, 687, 869, 94, 465, 417, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 714, 607, 16, 146, 194, 203, 798, 911, 668, 331, 387, 263, 421, 4, 466, 797, 395, 472, 459, 937, 806, 757, 451, 66, 758, 625, 563, 154, 639, 595, 824, 763, 240, 837, 0, 0, 0, 0, 0, 0, 22, 312, 884, 765, 24, 603, 956, 272, 78, 782, 480, 836, 491, 905, 524, 221, 97, 453, 990, 87, 476, 613, 881, 647, 129, 223, 519, 381, 141, 423, 346, 872, 542, 832, 609, 242, 84, 541, 136, 578, 939, 577, 287, 482, 29, 168, 142, 903, 461, 429, 681, 496, 0, 0, 0, 0, 0, 0, 549, 596, 922, 651, 366, 264, 382, 292, 600, 202, 74, 579, 859, 633, 436, 21, 488, 57, 41, 171, 158, 717, 781, 233, 164, 315, 855, 970, 397, 898, 894, 475, 660, 40, 159, 418, 732, 673, 745, 0, 0, 0, 0, 0, 0, 0, 0, 469, 662, 838, 674, 224, 967, 370, 102, 731, 456, 896, 821, 571, 854, 486, 378, 401, 284, 115, 309, 995, 407, 840, 750, 106, 949, 748, 534, 716, 553, 929, 352, 523, 738, 478, 267, 0, 0, 0, 0, 0, 0, 755, 133, 724, 803, 543, 384, 32, 532, 298, 108, 51, 698, 132, 981, 398, 442, 67, 177, 512, 669, 241, 173, 181, 367, 420, 65, 666, 652, 786, 684, 569, 410, 848, 468, 238, 850, 612, 752, 314, 819, 648, 727, 68, 966, 54, 664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 991, 253, 343, 432, 622, 190, 23, 310, 796, 566, 768, 597, 555, 733, 316, 258, 501, 546, 157, 833, 214, 677, 406, 446, 807, 305, 489, 204, 58, 915, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 412, 248, 885, 246, 299, 504, 827, 250, 818, 122, 608, 176, 773, 139, 317, 730, 700, 209, 968, 1000, 196, 889, 254, 2, 380, 103, 947, 433, 239, 591, 548, 279, 99, 499, 629, 393, 876, 341, 111, 511, 634, 743, 255, 26, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 285, 389, 19, 237, 313, 10, 917, 792, 414, 592, 826, 11, 205, 144, 627, 219, 259, 371, 787, 554, 780, 439, 434, 444, 800, 886, 75, 723, 817, 960, 623, 590, 861, 785, 481, 247, 530, 220, 0, 0, 0, 0, 0, 626, 15, 936, 377, 100, 535, 148, 950, 789, 965, 844, 104, 169, 938, 82, 228, 17, 988, 447, 311, 810, 274, 680, 640, 163, 282, 334, 47, 734, 564, 514, 823, 665, 890, 257, 76, 983, 654, 540, 38, 971, 982, 206, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 638, 685, 661, 402, 594, 413, 581, 729, 288, 887, 631, 440, 231, 900, 71, 963, 515, 374, 79, 820, 863, 812, 222, 877, 170, 260, 166, 49, 273, 59, 143, 892, 562, 245, 0, 0, 0, 0, 0, 0, 0, 547, 690, 428, 126, 656, 899, 349, 175, 490, 809, 852, 295, 679, 192, 527, 14, 649, 696, 180, 921, 712, 243, 962, 174, 692, 404, 286, 644, 232, 463, 386, 37, 185, 987, 396, 853, 559, 449, 5, 919, 760, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 485, 131, 289, 958, 642, 617, 912, 954, 359, 582, 12, 98, 718, 834, 435, 1, 835, 161, 149, 179, 959, 373, 520, 977, 191, 495, 739, 856, 777, 697, 725, 362, 924, 0, 0, 0, 0, 0, 0, 252, 722, 186, 923, 335, 92, 506, 347, 422, 874, 878, 533, 996, 793, 403, 477, 107, 363, 551, 813, 985, 109, 151, 236, 44, 33, 536, 815, 89, 528, 218, 783, 3, 101, 460, 882, 256, 829, 105, 464, 426, 635, 0, 0, 0, 0, 0, 0, 0, 0, 526, 614, 145, 332, 329, 361, 641, 211, 471, 646, 940, 953, 60, 112, 875, 120, 895, 39, 344, 822, 979, 686, 113, 392, 901, 201, 227, 866, 128, 586, 636, 31, 868, 28, 7, 448, 828, 505, 483, 560, 0, 0, 0, 0, 0, 0, 0, 183, 325, 659, 795, 941, 441, 897, 525, 910, 296, 35, 83, 302, 27, 619, 779, 655, 355, 208, 415, 598, 961, 568, 599, 195, 772, 454, 116, 124, 405, 278, 561, 808, 88, 327, 18, 198, 212, 794, 831, 545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 671, 521, 408, 235, 152, 658, 969, 516, 531, 216, 134, 61, 230, 326, 70, 95, 857, 703, 907, 270, 462, 73, 976, 390, 913, 585, 291, 719, 424, 964, 165, 927, 234, 736, 667, 411, 140, 268, 13, 0, 0, 0, 0, 0, 0, 0, 522, 615, 843, 306, 277, 290, 350, 517, 769, 513, 945, 584, 338, 759, 280, 470, 494, 934, 50, 925, 858, 379, 42, 587, 573, 135, 53, 452, 999, 570, 215, 91, 365, 369, 388, 172, 721, 830, 791, 507, 394, 0, 0, 0, 0, 0, 0, 0, 357, 138, 323, 303, 710, 281, 498, 249, 811, 790, 637, 529, 909, 69, 713, 973, 351, 908, 487, 497, 801, 589, 998, 62, 207, 932, 368, 770, 741, 747, 125, 997, 918, 502, 948, 30, 754, 358, 744, 683, 269, 0, 0, 0, 0, 0, 0, 356, 711, 682, 728, 847, 994, 630, 360, 86, 385, 611, 265, 695, 261, 557, 123, 574, 689, 225, 293, 304, 63, 376, 552, 117, 751, 294, 155, 188, 805, 933, 735, 993, 72, 409, 178, 880, 774, 364, 262, 870, 93, 0, 0, 0, 0, 0, 0, 0, 544, 708, 493, 193, 510, 301, 200, 888, 841, 643, 118, 784, 538, 337, 55, 879, 740, 425, 266, 20, 467, 891, 764, 906, 775, 518, 975, 984, 443, 156, 34, 244, 663, 955, 699, 920, 632, 52, 345, 883, 147, 0, 0, 0, 0, 0, 0, 0, 0, 199, 339, 130, 618, 508, 616, 670, 492, 737, 864, 628, 756, 839, 672, 904, 64, 726, 893, 621, 914, 946, 842, 930, 767, 814, 706, 620, 588, 916, 96, 445, 873, 348, 799, 121, 56, 715, 986, 479, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 788, 43, 746, 761, 90, 229, 330, 197, 276, 862, 319, 974, 271, 693, 431, 399, 816, 419, 307, 213, 127, 80, 778, 957, 694, 189, 110, 45, 320, 825, 575, 580, 372, 556, 860, 353, 438, 624, 416, 0, 0, 0, 742, 226, 606, 926, 187, 308, 707, 437, 610, 978, 766, 328, 849, 550, 802, 182, 391, 572, 484, 583, 455, 704, 846, 604, 771, 340, 943, 645, 935, 318, 458, 657, 509, 300, 702, 944, 753, 565, 275, 375, 676, 931, 867, 593, 567, 0, 650, 952, 989, 992, 400, 81, 865, 383, 450, 902, 342, 283, 776, 251, 167, 720, 85, 762, 601, 217, 602, 474, 500, 77, 678, 749, 333, 184, 951, 688, 942, 845, 8, 48, 928, 322, 430, 9, 576, 324, 605, 473, 321, 297, 537, 871, 705, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 851, 653, 36, 675]
# sol_hard_coded_2 = [196, 651, 527, 314, 616, 34, 418, 231, 721, 847, 61, 895, 362, 582, 246, 172, 958, 639, 12, 445, 65, 594, 120, 565, 566, 464, 348, 500, 0, 0, 0, 0, 0, 0, 0, 710, 873, 405, 740, 856, 976, 132, 319, 653, 605, 793, 557, 27, 211, 14, 274, 376, 157, 685, 192, 561, 446, 212, 591, 681, 103, 22, 576, 662, 374, 959, 425, 694, 890, 954, 0, 0, 0, 0, 0, 0, 0, 0, 53, 249, 817, 852, 645, 295, 820, 634, 440, 303, 363, 682, 676, 5, 378, 297, 844, 718, 145, 20, 511, 796, 841, 64, 429, 308, 915, 237, 46, 597, 419, 114, 943, 369, 197, 671, 184, 896, 67, 93, 275, 467, 805, 112, 0, 0, 0, 1000, 529, 323, 518, 888, 703, 420, 696, 840, 627, 113, 318, 476, 541, 968, 321, 151, 644, 289, 332, 869, 490, 501, 242, 294, 864, 602, 83, 148, 13, 596, 271, 79, 478, 775, 367, 857, 370, 474, 783, 952, 705, 666, 601, 329, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 572, 398, 264, 548, 898, 397, 238, 714, 670, 352, 946, 573, 121, 135, 426, 673, 836, 632, 722, 887, 134, 892, 994, 87, 300, 416, 439, 709, 357, 532, 658, 819, 870, 206, 125, 328, 567, 766, 808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 872, 372, 475, 768, 636, 660, 343, 638, 282, 241, 495, 424, 138, 782, 453, 913, 209, 663, 381, 95, 147, 748, 481, 39, 724, 798, 697, 16, 149, 468, 752, 751, 754, 551, 859, 610, 555, 742, 839, 687, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 223, 590, 553, 433, 646, 283, 832, 160, 25, 807, 402, 351, 712, 89, 74, 756, 505, 276, 364, 779, 176, 715, 434, 252, 441, 545, 311, 884, 243, 850, 878, 985, 334, 270, 908, 924, 458, 0, 523, 969, 128, 785, 109, 174, 259, 882, 593, 889, 105, 186, 141, 906, 488, 903, 977, 772, 353, 307, 758, 175, 199, 987, 517, 996, 55, 470, 942, 629, 173, 728, 920, 322, 749, 226, 136, 609, 665, 137, 764, 116, 350, 8, 313, 26, 978, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 810, 829, 932, 931, 394, 403, 101, 73, 866, 360, 691, 324, 707, 298, 28, 999, 961, 127, 43, 265, 776, 736, 855, 220, 628, 589, 462, 388, 699, 316, 661, 623, 0, 0, 0, 0, 0, 0, 0, 379, 509, 842, 626, 309, 617, 430, 643, 735, 533, 965, 10, 901, 992, 396, 9, 180, 119, 941, 443, 448, 773, 827, 88, 110, 143, 695, 512, 299, 225, 719, 480, 47, 635, 970, 24, 846, 787, 287, 950, 642, 130, 0, 0, 0, 0, 0, 0, 389, 52, 907, 537, 750, 75, 791, 99, 823, 427, 570, 510, 731, 571, 269, 450, 568, 366, 262, 927, 2, 58, 921, 578, 188, 281, 45, 550, 260, 997, 956, 229, 285, 50, 444, 290, 251, 228, 802, 991, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 716, 811, 449, 233, 975, 804, 6, 483, 191, 781, 182, 386, 984, 877, 822, 326, 865, 195, 336, 664, 678, 974, 983, 253, 152, 355, 21, 340, 713, 586, 538, 753, 883, 900, 584, 592, 438, 0, 0, 0, 0, 0, 0, 0, 0, 917, 928, 972, 784, 404, 431, 391, 546, 459, 423, 684, 286, 868, 359, 291, 91, 189, 680, 37, 622, 851, 947, 504, 461, 513, 371, 163, 301, 63, 543, 526, 158, 435, 654, 36, 686, 7, 293, 472, 683, 0, 0, 0, 0, 0, 0, 0, 0, 843, 934, 717, 989, 250, 780, 577, 620, 655, 674, 216, 261, 296, 818, 964, 187, 18, 72, 916, 871, 126, 338, 837, 552, 169, 971, 542, 310, 861, 117, 201, 693, 80, 539, 198, 469, 451, 918, 244, 813, 0, 0, 0, 0, 0, 0, 0, 213, 806, 569, 951, 278, 825, 333, 688, 652, 341, 579, 814, 421, 559, 15, 94, 247, 704, 484, 146, 957, 788, 769, 585, 521, 739, 118, 477, 51, 178, 912, 428, 90, 966, 853, 904, 619, 598, 734, 382, 437, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 982, 392, 354, 797, 720, 69, 32, 230, 669, 482, 891, 726, 496, 107, 177, 564, 816, 62, 939, 491, 777, 607, 767, 711, 306, 700, 506, 66, 224, 879, 556, 81, 826, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 280, 834, 268, 108, 408, 624, 640, 732, 737, 156, 544, 30, 637, 948, 575, 315, 48, 407, 442, 885, 897, 232, 41, 235, 204, 599, 771, 380, 97, 325, 515, 409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 167, 657, 854, 200, 155, 412, 534, 936, 919, 935, 144, 668, 949, 667, 471, 54, 962, 219, 880, 729, 86, 547, 202, 995, 164, 3, 925, 23, 828, 530, 763, 588, 692, 581, 35, 583, 273, 214, 519, 245, 923, 168, 432, 606, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 346, 980, 648, 100, 485, 256, 815, 560, 630, 263, 98, 499, 675, 347, 236, 998, 393, 672, 794, 894, 102, 902, 383, 179, 608, 410, 633, 358, 60, 600, 708, 497, 867, 104, 49, 611, 123, 761, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 790, 414, 746, 133, 727, 255, 831, 447, 738, 621, 603, 910, 875, 812, 580, 82, 755, 29, 57, 522, 886, 387, 618, 701, 111, 194, 824, 181, 85, 150, 945, 730, 331, 59, 317, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 417, 801, 330, 284, 821, 760, 990, 955, 778, 492, 689, 455, 986, 863, 631, 56, 725, 762, 930, 615, 210, 344, 222, 905, 327, 659, 4, 848, 792, 702, 335, 234, 858, 765, 723, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 507, 466, 558, 973, 166, 562, 960, 171, 929, 131, 337, 914, 574, 221, 845, 677, 153, 38, 227, 770, 967, 375, 240, 508, 489, 979, 741, 92, 835, 830, 860, 292, 479, 944, 305, 159, 940, 267, 0, 0, 0, 0, 0, 0, 0, 487, 129, 106, 413, 803, 563, 207, 937, 698, 277, 595, 185, 349, 356, 345, 400, 384, 650, 498, 988, 33, 365, 302, 452, 993, 926, 690, 239, 463, 909, 320, 757, 963, 494, 799, 368, 465, 68, 679, 203, 809, 0, 0, 0, 0, 0, 0, 0, 208, 899, 612, 304, 266, 76, 502, 473, 377, 862, 745, 205, 31, 122, 747, 170, 139, 933, 254, 833, 549, 70, 604, 893, 838, 154, 373, 656, 922, 161, 84, 288, 587, 874, 272, 514, 1, 456, 641, 881, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 457, 759, 849, 706, 422, 743, 554, 395, 525, 215, 78, 744, 342, 493, 614, 44, 279, 911, 613, 733, 77, 312, 953, 71, 401, 361, 40, 165, 19, 795, 460, 258, 625, 786, 531, 193, 257, 436, 540, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 528, 96, 339, 981, 789, 0, 0, 0, 0, 0, 0, 454, 415, 536, 385, 486, 649, 938, 406, 516, 162, 503, 399, 876, 520, 411, 183, 390, 524, 535, 774, 647, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]