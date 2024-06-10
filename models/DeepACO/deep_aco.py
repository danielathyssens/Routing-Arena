from typing import List, Union, Dict, Tuple
from formats import TSPInstance, CVRPInstance, RPSolution
from omegaconf import DictConfig

import time
import torch
import numpy as np
import logging
from torch.distributions import Categorical, kl
from d2l.torch import Animator

from models.DeepACO.DeepACO.tsp.net import Net as Net_tsp
from models.DeepACO.DeepACO.cvrp.net import Net as Net_cvrp
from models.DeepACO.DeepACO.tsp.aco import ACO as ACO_tsp
from models.DeepACO.DeepACO.cvrp_nls.aco import ACO as ACO_cvrp_nls
from models.DeepACO.DeepACO.tsp_nls.aco import ACO as ACO_tsp_nls
from models.DeepACO.DeepACO.cvrp.aco import ACO as ACO_cvrp
from models.DeepACO.DeepACO.cvrp.utils import gen_distance_matrix as gen_distance_matrix_cvrp
from models.DeepACO.DeepACO.cvrp_nls.utils import gen_distance_matrix as gen_distance_matrix_cvrp_nls

EPS = 1e-10

logger = logging.getLogger(__name__)


def eval_model(problem: str,
               model: Union[Net_tsp, Net_cvrp, None],
               aco_env: Union[ACO_tsp, ACO_cvrp, ACO_tsp_nls, ACO_cvrp_nls],
               data: Union[List[TSPInstance], List[CVRPInstance]],
               device: torch.device,
               tester_cfg: DictConfig,
               adjusted_time_budget: int):

    tester_cfg.update(time_limit=int(adjusted_time_budget))

    # get data transforms
    if "nls" in str(aco_env):
        gen_dists = gen_distance_matrix_cvrp_nls
        with_nls = True
    else:
        gen_dists = gen_distance_matrix_cvrp
        with_nls = False
    test_list = transform_data_deepaco(problem, data, device, tester_cfg["k_sparse"], gen_dists, with_nls)
    # print('test_list', test_list)
    # some preprocessing on the fly
    _t_aco = [0] + tester_cfg["t_aco"]
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    if problem == 'tsp':
        # from models.DeepACO.DeepACO.tsp.test import test
        avg_aco_best, duration, results_all, sols_all = test_tsp(aco_env, test_list, model,
                                                                 tester_cfg,
                                                                 t_aco_diff,
                                                                 device,
                                                                 tester_cfg["k_sparse"])

        logger.info(f"Internal DeepACO average TSP-cost: {avg_aco_best}")

    elif problem == 'cvrp':
        # device hard coded in cvrp/test.py to be str "cpu" --> but will ignore here and run on GPU
        # from models.DeepACO.DeepACO.tsp.test import test
        # results_all
        # print('tester_cfg', tester_cfg)
        avg_aco_best, duration, results_all, sols_all = test_cvrp(aco_env, test_list, model,
                                                                  tester_cfg,
                                                                  t_aco_diff,
                                                                  device=device)  # device="cpu"
        logger.info(f"Internal DeepACO average CVRP-cost: {avg_aco_best}")
    else:
        avg_aco_best, duration, results_all, sols_all = None, None, None, None

    return {}, make_RPSolution(problem, results_all, duration, sols_all, data, with_nls)


@torch.no_grad()
def test_tsp(env, dataset, model, tester_cfg, t_aco, device, k_sparse=None):
    n_ants = tester_cfg.n_ants
    time_limit = tester_cfg.time_limit if not tester_cfg.ignore_time_limit else None
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    results_per_inst, sols_per_inst = [], []
    start = time.time()
    for pyg_data, distances in dataset:
        results, res_per_it, sols_per_it = infer_instance_tsp(env, model, pyg_data, distances, n_ants,
                                                              t_aco_diff, time_limit, device, k_sparse)
        results_per_inst.append(res_per_it[0])
        sols_per_inst.append(sols_per_it[0])
        sum_results += results
    end = time.time()

    return sum_results / len(dataset), end - start, results_per_inst, sols_per_inst


@torch.no_grad()
def test_cvrp(env, dataset, model, tester_cfg, t_aco, device):
    n_ants = tester_cfg.n_ants
    time_limit = tester_cfg.time_limit if not tester_cfg.ignore_time_limit else None
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    results_per_inst, sols_per_inst = [], []
    start = time.time()
    if "nls" not in str(env):
        for demands, distances, _, _ in dataset:
            results, res_per_it, sols_per_it = infer_instance_cvrp(env, model, demands.unsqueeze(0), distances, n_ants,
                                                                   t_aco_diff, time_limit, device=device)
            results_per_inst.append(res_per_it[0])
            sols_per_inst.append(sols_per_it[0])
            sum_results += results
    else:
        for demands, distances, locs, capa in dataset:
            demands = demands / capa
            results, res_per_it, sols_per_it = infer_instance_cvrp(env, model, demands.unsqueeze(0), distances, n_ants,
                                                                   t_aco_diff, time_limit, device=device,
                                                                   positions=locs)
            results_per_inst.append(res_per_it[0])
            sols_per_inst.append(sols_per_it[0])
            sum_results += results
    end = time.time()

    return sum_results / len(dataset), end - start, results_per_inst, sols_per_inst


@torch.no_grad()
def infer_instance_tsp(env, model, pyg_data, distances, n_ants, t_aco_diff, time_limit, device, k_sparse=None):
    print('distances', distances)
    if model and "nls" in str(env):
        # logger.info("evaluate DeepACO with NLS ...")
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        aco = env(
            n_ants=n_ants,
            heuristic=heu_mat.cpu(),
            distances=distances.cpu(),
            device='cpu',
            local_search='nls',
        )
    elif not model and "nls" in str(env):
        aco = env(
            n_ants=n_ants,
            distances=distances.cpu(),
            device='cpu',
            local_search='nls',
        )
    elif model:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        aco = env(
            n_ants=n_ants,
            heuristic=heu_mat,
            distances=distances,
            device=device
        )
    else:
        aco = env(
            n_ants=n_ants,
            distances=distances,
            device=device
        )
        if k_sparse:
            aco.sparsify(k_sparse)

    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    results_all_it = []
    sols_all_it = []
    for i, t in enumerate(t_aco_diff):
        best_cost, results_per_it, sols_per_it = aco.run(t, time_limit=time_limit)
        # print('len(sols_per_it)', len(sols_per_it))
        # print('sols_per_it', sols_per_it)
        results[i] = best_cost
        results_all_it.append(results_per_it)
        sols_all_it.append(sols_per_it)
    assert len(results) == len(results_all_it) == len(t_aco_diff)
    return results, results_all_it, sols_all_it


def infer_instance_cvrp(env, model, demands, distances, n_ants, t_aco_diff, time_limit, device, positions=None):
    if model and "nls" in str(env):
        from models.DeepACO.DeepACO.cvrp_nls.utils import gen_pyg_data
        from models.DeepACO.DeepACO.cvrp_nls.aco import get_subroutes
        # logger.info("evaluate DeepACO with NLS ...")
        model.eval()
        pyg_data = gen_pyg_data(demands.squeeze(0), distances, device)
        heu_vec = model(pyg_data)
        # heu_mat = heu_vec.reshape((n_node + 1, n_node + 1)) + EPS
        heu_mat = model.reshape(pyg_data, heu_vec) + 0.001 #  EPS
        # need to gen positions --> dataset
        aco = env(
            n_ants=n_ants,
            heuristic=heu_mat.cpu(),
            demand=demands.squeeze(0).cpu(),
            distances=distances.cpu(),
            device='cpu',
            swapstar=True,
            positions=positions.cpu(),
            inference=True,
        )
        results = torch.zeros(size=(len(t_aco_diff),), device=device)
        # results = torch.zeros(size=(len(t_aco_diff),), device=device)
        results_all_it, sols_all_it = [], []
        for i, t in enumerate(t_aco_diff):
            best_cost, results_per_it, sols_per_it = aco.run(t, inference=True, time_limit=time_limit)
            # print('sols_p:]', sols_per_it[-1])
            results[i] = best_cost
            path = get_subroutes(aco.shortest_path)
            valid, results[i], final_route = validate_route(distances, demands.squeeze(0), path)
            # print('final_route==sols_per_it[-1]', [(f==s).all() for f,s in zip(final_route, sols_per_it[-1])])
            results_all_it.append(results_per_it)
            sols_all_it.append(sols_per_it)
            if valid is False:
                print("invalid solution.")
        return results, results_all_it, sols_all_it
    elif model:
        from models.DeepACO.DeepACO.cvrp.utils import gen_pyg_data
        n_node = demands.shape[1] - 1
        model.eval()
        pyg_data = gen_pyg_data(demands.squeeze(0), distances, device)
        heu_vec = model(pyg_data)
        heu_mat = heu_vec.reshape((n_node + 1, n_node + 1)) + EPS
        aco = env(
            distances=distances,
            demand=demands.squeeze(0),
            n_ants=n_ants,
            heuristic=heu_mat,
            device=device
        )
        results = torch.zeros(size=(len(t_aco_diff),), device=device)
        results_all_it = []
        sols_all_it = []
        for i, t in enumerate(t_aco_diff):
            best_cost, results_per_it, sols_per_it = aco.run(t)
            results[i] = best_cost
            results_all_it.append(results_per_it)
            sols_all_it.append(sols_per_it)
        return results, results_all_it, sols_all_it
    else:
        aco = env(
            distances=distances,
            demand=demands,
            n_ants=n_ants,
            device=device
        )

        results = torch.zeros(size=(len(t_aco_diff),), device=device)
        results_all_it = []
        sols_all_it = []
        for i, t in enumerate(t_aco_diff):
            best_cost, results_per_it, sols_per_it = aco.run(t)
            results[i] = best_cost
            results_all_it.append(results_per_it)
            sols_all_it.append(sols_per_it)
        return results, results_all_it, sols_all_it


def transform_data_deepaco(problem: str, data: List, device, k_sparse=None, gen_dists=None, with_nls: bool = False):
    # DeepACO needs torch.float32 tensors

    if problem.lower() == 'tsp':
        if with_nls:
            from models.DeepACO.DeepACO.tsp_nls.utils import gen_pyg_data as gen_pyg_data_tsp
            dat_tensor_list = make_tensor_lists(data, 'tsp', device)
            return [gen_pyg_data_tsp(tensor_list_item, k_sparse, start_node=0) for tensor_list_item in
                    dat_tensor_list]
        else:
            from models.DeepACO.DeepACO.tsp.utils import gen_pyg_data as gen_pyg_data_tsp
            dat_tensor_list = make_tensor_lists(data, 'tsp', device)
            return [gen_pyg_data_tsp(tensor_list_item, k_sparse) for tensor_list_item in
                    dat_tensor_list]

    elif problem.lower() == 'cvrp':
        return make_tensor_lists(data, 'cvrp', device, gen_dists)
    else:
        raise NotImplementedError


def make_tensor_lists(dat: List, problem: str, device=None, gen_dists=None,
                      offset: int = 0) -> Union[List, torch.tensor]:
    if problem.lower() == 'tsp':
        return [torch.FloatTensor(inst.coords).to(device) for inst in (dat[offset:offset + len(dat)])]
    elif problem.lower() == 'cvrp':
        dat_cvrp = [make_cvrp_instance(args, device, gen_dists) for args in dat[offset:offset + len(dat)]]
        return dat_cvrp
    else:
        raise NotImplementedError


def make_cvrp_instance(instance: CVRPInstance, device="cpu", gen_dists=None):
    depot = torch.tensor(instance.coords[0])
    loc = torch.tensor(instance.coords[1:])
    all_demand = torch.tensor(instance.node_features[:, instance.constraint_idx[0]]).to(torch.float32)
    capa_orig = torch.tensor(instance.original_capacity)
    # print('capa_orig', capa_orig)
    all_locs = torch.cat((depot.unsqueeze(0), loc), dim=0).to(torch.float64)
    distances = gen_dists(all_locs)
    distances = distances.to(torch.float32)
    data_tuple = (all_demand.to(device), distances.to(device), all_locs.to(device), capa_orig.to(device))
    return data_tuple


def make_RPSolution(problem, times_costs, total_time, sols_tensor_list, data, with_nls) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sols_lists, times, costs = [], [], []
    if problem.lower() == 'tsp':
        for time_cost, inst_sols in zip(times_costs, sols_tensor_list):
            sols_list = [incumbent_sol.tolist() for incumbent_sol in inst_sols]
            sols_lists.append(sols_list)
            times.append(time_cost[:, 0].tolist())
            costs.append(time_cost[:, 1].tolist())
    else:
        for time_cost, inst_sols in zip(times_costs, sols_tensor_list):
            if not with_nls:
                sols_list = _get_sep_tours(problem.lower(), inst_sols)
            else:
                sols_list = []
                for running_sol in inst_sols:
                    sols_list.append([tour.tolist() for tour in running_sol])
            sols_lists.append(sols_list)
            times.append(time_cost[:, 0].tolist())
            costs.append(time_cost[:, 1].tolist())
    # print('len(times_costs)', len(times_costs))
    # print('len(sols_tensor_list)', len(sols_tensor_list))
    # print('sols_tensor_list[0]', sols_tensor_list[0])
    # print('sols_lists[0]', sols_lists[0])
    # print('times_costs[0]', times_costs[0])
    # print('times[0]', times[0])
    # print('costs[0]', costs[0])
    rp_sols = []
    for sol, c, t, inst in zip(sols_lists, costs, times, data):
        s = RPSolution(
            solution=sol[-1],
            cost=c[-1],
            run_time=t[-1],
            running_sols=sol,
            running_costs=c,
            running_times=t,
            num_vehicles=len(sol[-1]) if problem.upper() == 'CVRP' else 1,
            problem=problem,
            instance=inst,
            method_internal_cost=c[-1]
        )
        rp_sols.append(s)
    return rp_sols


def _get_sep_tours(problem: str, sols: torch.Tensor) -> List[List]:
    """get solution (res) as List[List]"""
    # parse solution
    # print('problem', problem)
    if problem.lower() == 'tsp':
        # if problem is tsp - only have single tour
        # for sol_ in sols:
        return [sol_.tolist() for sol_ in sols]

    elif problem.lower() == 'cvrp':
        sols_ = []
        # for sol_batch in sols:
        #     print('sol_batch', sol_batch)
        # print('sols', sols)
        for sol in sols:
            # print('sol', sol)
            sol_lst = sol_to_list(sol, depot_idx=0)
            sols_.append(sol_lst)
        # print(f'sols_: {sols_}')
        return sols_


# Transform solution returned from DeepACO to List[List]
def sol_to_list(sol: Union[torch.tensor, np.ndarray], depot_idx: int = 0) -> List[List]:
    sol_lst, lst = [], [0]
    for e in sol:
        if e == depot_idx:
            if len(lst) > 1:
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


def validate_route(distance: torch.Tensor, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, float]:
    length = 0.0
    valid = True
    visited = {0}
    # print('routes: ', routes)
    for r in routes:
        d = demands[r].sum().item()
        # print('d', d)
        if d > 1.000001:
            valid = False
        length += distance[r[:-1], r[1:]].sum().item()
        for i in r:
            i = i.item()
            if i < 0 or i >= distance.size(0):
                valid = False
            else:
                visited.add(i)
    if len(visited) != distance.size(0):
        valid = False
    return valid, length, routes
