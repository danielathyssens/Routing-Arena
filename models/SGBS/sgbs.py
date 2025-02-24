#
import numpy
import os
import sys

import logging
import warnings
import time
import itertools as it
from typing import Union, Optional, Dict, List, Tuple, Any, Type

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from formats import TSPInstance, CVRPInstance, RPSolution

# TSP
from models.SGBS.SGBS.tsp.SGBS.E_TSPEnv import TSPEnv
from models.SGBS.SGBS.tsp.SGBS.E_TSPModel import TSPModel
# + EAS
from models.SGBS.SGBS.tsp.SGBS_EAS.E_TSPEnv import TSPEnv as TSPEnv_EAS
from models.SGBS.SGBS.tsp.SGBS_EAS.E_TSPModel import TSPModel as TSPModel_EAS
# CVRP
from models.SGBS.SGBS.cvrp.SGBS.E_CVRPEnv import CVRPEnv
from models.SGBS.SGBS.cvrp.SGBS.CVRPModel import CVRPModel
# + EAS
from models.SGBS.SGBS.cvrp.SGBS_EAS.E_CVRPEnv import CVRPEnv as CVRPEnv_EAS
from models.SGBS.SGBS.cvrp.SGBS_EAS.E_CVRPModel import CVRPModel as CVRPEnv_EAS

# from models.SGBS.SGBS.cvrp.SGBS.CVRPTrainer import CVRPTrainer
# from models.SGBS.SGBS.TSP.SGBS.TSPTrainer import TSPTrainer
from models.SGBS.SGBS.cvrp.SGBS.CVRPTester import CVRPTester
from models.SGBS.SGBS.cvrp.SGBS_EAS.CVRPTester import CVRPTester as CVRPTester_EAS
from models.SGBS.SGBS.tsp.SGBS.TSPTester import TSPTester
from models.SGBS.SGBS.tsp.SGBS_EAS.TSPTester import TSPTester as TSPTester_EAS

logger = logging.getLogger(__name__)

# Machine Environment Config
DEBUG_MODE = False
# USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


def train_model(Trainer: None,  # Type[Union[TSPTrainer, CVRPTrainer]]
                env: Union[TSPEnv, CVRPEnv],
                model: Union[TSPModel, CVRPModel],
                optimizer: Optimizer,
                scheduler: Scheduler,
                device: torch.device = torch.device("cpu"),
                train_cfg: Optional[Dict] = None):
    warnings.warn("SGBS uses pretrained POMO model checkpoints... No SGBS training implemented!")
    raise NotImplementedError


def eval_model(Tester: Type[Union[TSPTester, CVRPTester, TSPTester_EAS, CVRPTester_EAS]],
               policy: str,
               env: Union[TSPEnv, CVRPEnv],
               model: Union[TSPModel, CVRPModel],
               data: List[CVRPInstance],
               tester_cfg: Optional[Dict],
               device=torch.device("cpu"),
               time_limit=None,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # comment out because here we want to test on single instances (not in batches of 400 for example)
    # augmentation is still happening
    # if tester_cfg['augmentation_enable']:
    #     tester_cfg['test_batch_size'] = tester_cfg['aug_batch_size']
    # print("tester_cfg['test_batch_size']", tester_cfg['test_batch_size'])
    # print('policy', policy)
    problem = 'cvrp' if Tester in [CVRPTester, CVRPTester_EAS] else 'tsp'
    USE_CUDA = True if device != torch.device("cpu") else False
    if policy == "SGBS":
        if len(set([inst.graph_size for inst in data])) == 1:
            logger.info(f"All graph sizes are the same: {data[0].graph_size}")
            # all instances in test set are the same size (n):
            tester_cfg['test_episodes'] = len(data)
            if env.env_params['problem_size'] is None:
                # means that problem_size (aka graph_size) in test set is set to None, b/c instances have different size
                env.problem_size = data[0].graph_size - 1 if problem == "cvrp" else data[0].graph_size
                print('env.problem_size', env.problem_size)
            if env.env_params['pomo_size'] is None:
                # means that pomo_size is set to graph_size of instances in test set, but instances have different size
                tester_cfg['pomo_size'] = data[0].graph_size - 1 if problem == "cvrp" else data[0].graph_size
                env.pomo_size = data[0].graph_size - 1
                print('env.pomo_size', env.pomo_size)
                # env.pomo_size = data[0].graph_size - 1
            # print('data[0].node_features[:5, -1]', data[0].node_features[:5, -1])
            logger.info(f'tester_cfg before RUN: {tester_cfg}')
            tester = Tester(env=env,
                            model=model,
                            tester_params=tester_cfg,
                            USE_CUDA=USE_CUDA)
            sols, runtimes, costs = tester.run(data=data)  # , costs_aug
        else:
            # Test Instances (mixed size):
            tester_cfg['test_episodes'] = 1  # adapt test_episodes to 1 for each of the instances in single run
            sols, runtimes, costs, costs_aug = [], [], [], []
            for instance in data:
                if env.env_params['problem_size'] is None or env.env_params['problem_size'] != instance.graph_size - 1:
                    env.problem_size = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
                if env.env_params['pomo_size'] is None or env.env_params['problem_size'] != instance.graph_size - 1:
                    tester_cfg['pomo_size'] = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
                    env.pomo_size = instance.graph_size - 1  if problem == "cvrp" else instance.graph_size
                tester = Tester(env=env,
                                model=model,
                                tester_params=tester_cfg,
                                USE_CUDA=USE_CUDA)
                time_budget = time_limit if time_limit is not None else instance.time_limit
                sol, runtime, cost = tester.run(data=[instance],
                                                time_budget=time_budget)  # , cost_aug
                # print('sol', sol)
                # print('len(sol)', len(sol))
                sols.append(sol[0])
                runtimes.extend(runtime)
                costs.extend(cost)
                # costs_aug.extend(cost_aug)
        s_parsed = _get_sep_tours(problem, sols)
        solutions = make_RPSolution(problem, costs, runtimes, s_parsed, data)
        return {}, solutions
    elif policy == "SGBS_EAS":
        # tester_cfg['test_episodes'] = len(data)
        # tester_cfg['num_episodes'] = len(data)
        tester_cfg['test_episodes'] = 1  # adapt test_episodes to 1 for each of the instances in single run
        # tester = Tester(env=env,
        #                 model=model,
        #                 run_params=tester_cfg,
        #                 USE_CUDA=USE_CUDA)
        #                         test_data=data,
        sols, runtimes, costs, running_costs, running_sols, running_sols, running_rts = [], [], [], [], [], [], []
        iterations_vs_time = []
        # sols, runtimes, costs, running_sols, running_rts, running_costs = tester.run()  # , cost_aug
        for instance in data:
            print('INSTANCE ID', instance.instance_id)
            if env.env_params['problem_size'] is None or env.env_params['problem_size'] != instance.graph_size - 1:
                # means that problem_size (aka graph_size) in test set is set to None, b/c instances have different size
                print('instance.graph_size', instance.graph_size)
                print('instance.graph_size - 1', instance.graph_size - 1)
                if instance.graph_size == 99:
                    instance = instance.update(graph_size=100)
                print('instance.graph_size', instance.graph_size)
                env.env_params['problem_size'] = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
                env.problem_size = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
            if env.env_params['pomo_size'] is None or env.env_params['pomo_size'] != instance.graph_size - 1:
                # means that pomo_size is set to graph_size of instances in test set, but instances have different size
                # print('instance.graph_size', instance.graph_size)
                # print('instance.graph_size - 1', instance.graph_size - 1)
                env.env_params['pomo_size'] = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
                env.pomo_size = instance.graph_size - 1 if problem == "cvrp" else instance.graph_size
            # if instance.graph_size > tester_cfg['solution_max_length']:
            # print('instance.graph_size', instance.graph_size)
            # print("tester_cfg['solution_max_length']", tester_cfg['solution_max_length'])
            tester_cfg['solution_max_length'] = (instance.graph_size - 1)*2 if problem == "cvrp" else instance.graph_size
            logger.info(f'tester_cfg before RUN: {tester_cfg}')
            tester = Tester(env=env,
                            model=model,
                            run_params=tester_cfg,
                            USE_CUDA=USE_CUDA)
            time_budget = time_limit if time_limit is not None else instance.time_limit
            sol, runtime, cost, running_sol, running_rt, running_cost, loop_counts = tester.run(data=[instance],
                                                                                                time_budget=time_budget)
            # print('sol', sol)
            # print('loop_counts', loop_counts)
            sols.append(sol[0])
            runtimes.extend(runtime)
            costs.extend(cost)
            running_sols.append(running_sol)
            running_rts.append(running_rt)
            running_costs.append(running_cost)
            iterations_vs_time.append(loop_counts)
        # print('sols', sols)
        # print('running_sols', running_sols)
        # print('running_rts', running_rts)
        # print('running_costs', running_costs)
        print('iterations_vs_time', iterations_vs_time)
        s_parsed = _get_sep_tours(problem, sols)
        # print('s_parsed', s_parsed)
        s_parsed_running = [_get_sep_tours(problem, running_sol) for running_sol in running_sols]
        # print('s_parsed_running', s_parsed_running)
        # print('len(s_parsed_running', len(s_parsed_running))

        # print('len (running_costs', len(running_costs))
        # print('running_rts', running_rts)
        solutions = make_RPSolution(problem, costs, runtimes, s_parsed, data, s_parsed_running, running_rts,
                                    running_costs, iterations_time=iterations_vs_time)
        return {}, solutions
    else:
        raise NotImplementedError


def _get_sep_tours(problem: str, sols: Union[torch.Tensor, List]) -> List[List]:
    """get solution (res) as List[List]"""
    # parse solution
    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        parsed_sols=[]

        for sol_ in sols:
            # print('sol_', sol_)
            # return sol_.tolist()
            if torch.is_tensor(sol_):
                # print('sol_ is tensor')
                parsed_sols.append(sol_.tolist())
                # print('parsed_sols in loop:', parsed_sols)
            elif isinstance(sol_, list):
                # print('sol_ is running sol for one instance')
                # print('sol_', sol_)
                parsed_sols.append([soll.tolist() for soll in sol_])
        return parsed_sols

    elif problem.lower() == 'cvrp':
        sols_ = []
        # for sol_batch in sols:
        #     print('sol_batch', sol_batch)
        # print('sols', sols)
        for sol in sols:
            sol_lst = sol_to_list(sol, depot_idx=0)
            sols_.append(sol_lst)
        # print(f'sols_ after parse: {sols_}')
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


def make_RPSolution(problem, rews, times, s_parsed, data,
                    running_s=None, running_t=None, running_c=None, iterations_time=None) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    # sol_list = [_get_sep_tours(problem, sol_) for sol_ in sols]
    print('iterations_time', iterations_time)
    if running_t == running_c == running_s is None:
        return [
            RPSolution(
                solution=sol,
                cost=r,
                num_vehicles=len(sol) if problem.upper() == 'CVRP' else len([sol]),
                run_time=t,  # float(t[:-1]),
                problem=problem,
                instance=inst,
                method_internal_cost=r,
                iterations_time=it_t
            )
            for sol, r, t, inst, it_t in zip(s_parsed, rews, times, data, iterations_time)
        ]
    else:
        # print('running_c', running_c)
        running_c_upd = []
        for inst_runn_c in running_c:
            running_c_upd.append([r_c.cpu().item() if isinstance(r_c, torch.Tensor) else r_c for r_c in inst_runn_c])
        return [
            RPSolution(
                solution=sol,
                cost=r,
                num_vehicles=len(sol) if problem.upper() == 'CVRP' else len([sol]),
                run_time=t,  # float(t[:-1]),
                problem=problem,
                instance=inst,
                method_internal_cost=r.cpu().item() if isinstance(r, torch.Tensor) else r,
                running_sols=runn_s,
                running_costs=runn_c,
                running_times=runn_t,
                iterations_time=it_t
            )
            for sol, r, t, inst, runn_s, runn_t, runn_c, it_t in
            zip(s_parsed, rews, times, data, running_s, running_t, running_c_upd, iterations_time)
        ]


def make_cvrp_instance(instance: CVRPInstance):
    depot = torch.tensor(instance.coords[0])
    loc = torch.tensor(instance.coords[1:])
    demand = torch.tensor(instance.node_features[1:, instance.constraint_idx[0]])
    return depot, loc, demand


def prep_data(problem: str, dat: Union[List[TSPInstance], List[CVRPInstance]], offset=0):
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    if problem.lower() == "tsp":
        return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    elif problem.lower() == "cvrp":
        return [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]
    else:
        raise NotImplementedError


# state and env from POMO adapted to work without imported problem parameters
################# TSP #######################

class TSP_GROUP_STATE:
    def __init__(self, group_size, data, PROBLEM_SIZE, device):
        # data.shape = (batch, group, 2)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.PROBLEM_SIZE = PROBLEM_SIZE

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = torch.from_numpy(np.zeros((self.batch_s, self.group_s, 0))).long().to(device)
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.ninf_mask = Tensor(np.zeros((self.batch_s, group_size, PROBLEM_SIZE + 1))).to(device, dtype=torch.float)
        # shape = (batch, group, PROBLEM_SIZE)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf


class TSP_GROUP_ENVIRONMENT:
    def __init__(self, data, PROBLEM_SIZE, device):
        # seq.shape = (batch, TSP_SIZE, 2)
        self.data = data
        self.batch_s = data.size(0)
        self.group_s = None
        self.group_state = None
        self.PROBLEM_SIZE = PROBLEM_SIZE
        self.device = device

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = TSP_GROUP_STATE(group_size=group_size,
                                           data=self.data,
                                           PROBLEM_SIZE=self.PROBLEM_SIZE,
                                           device=self.device)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = (self.group_state.selected_count == TSP_SIZE)
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(self.batch_s, -1, TSP_SIZE, 2)
        # shape = (batch, group, TSP_SIZE, 2)
        seq_expanded = self.data[:, None, :, :].expand(self.batch_s, self.group_s, TSP_SIZE, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape = (batch, group, TSP_SIZE, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # size = (batch, group, TSP_SIZE)

        group_travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return group_travel_distances


################# CVRP #############################

class CVRP_GROUP_STATE:
    def __init__(self, group_size, data, PROBLEM_SIZE, device):
        # data.shape = (batch, problem+1, 3)

        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.PROBLEM_SIZE = PROBLEM_SIZE

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = torch.from_numpy(np.zeros((self.batch_s, self.group_s, 0))).long().to(device)
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.at_the_depot = None
        # shape = (batch, group)
        self.loaded = torch.from_numpy(np.ones((self.batch_s, self.group_s))).to(device, dtype=torch.float)
        # shape = (batch, group)
        self.visited_ninf_flag = torch.from_numpy(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE + 1))).to(device,
                                                                                                               dtype=torch.float)
        # shape = (batch, group, problem+1)
        self.ninf_mask = torch.from_numpy(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE + 1))).to(device,
                                                                                                       dtype=torch.float)
        # shape = (batch, group, problem+1)
        self.finished = torch.from_numpy(np.zeros((self.batch_s, self.group_s))).bool().to(device)
        # shape = (batch, group)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        self.at_the_depot = (selected_idx_mat == 0)
        demand_list = self.data[:, None, :, 2].expand(self.batch_s, self.group_s, -1)
        # shape = (batch, group, problem+1)
        gathering_index = selected_idx_mat[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape = (batch, group)
        self.loaded -= selected_demand
        self.loaded[self.at_the_depot] = 1  # refill loaded at the depot
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.visited_ninf_flag[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf
        self.finished = self.finished + (self.visited_ninf_flag == -np.inf).all(dim=2)
        # shape = (batch, group)

        # Status Edit
        ####################################
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # allow car to visit depot anytime
        round_error_epsilon = 0.000001
        demand_too_large = self.loaded[:, :, None] + round_error_epsilon < demand_list
        # shape = (batch, group, problem+1)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.ninf_mask[demand_too_large] = -np.inf

        self.ninf_mask[self.finished[:, :, None].expand(self.batch_s, self.group_s, self.PROBLEM_SIZE + 1)] = 0
        # do not mask finished episode


class CVRP_GROUP_ENVIRONMENT:

    def __init__(self, depot_xy, node_xy, node_demand, PROBLEM_SIZE, device):
        # depot_xy.shape = (batch, 1, 2)
        # node_xy.shape = (batch, problem, 2)
        # node_demand.shape = (batch, problem, 1)

        self.batch_s = depot_xy.size(0)
        self.group_s = None
        self.group_state = None
        self.PROBLEM_SIZE = PROBLEM_SIZE
        self.device = device

        all_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape = (batch, problem+1, 2)
        depot_demand = torch.from_numpy(np.zeros((self.batch_s, 1, 1))).to(device, dtype=torch.float)
        all_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape = (batch, problem+1, 1)
        self.data = torch.cat((all_node_xy, all_node_demand), dim=2)
        # shape = (batch, problem+1, 3)

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = CVRP_GROUP_STATE(group_size=group_size,
                                            data=self.data,
                                            PROBLEM_SIZE=self.PROBLEM_SIZE,
                                            device=self.device)

        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.finished.all()  # state.finished.shape = (batch, group)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_travel_distance(self):
        all_node_xy = self.data[:, None, :, 0:2].expand(self.batch_s, self.group_s, -1, 2)
        # shape = (batch, group, problem+1, 2)
        gathering_index = self.group_state.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape = (batch, group, selected_count, 2)
        ordered_seq = all_node_xy.gather(dim=2, index=gathering_index)
        # shape = (batch, group, selected_count, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # size = (batch, group, selected_count)

        travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return travel_distances


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
