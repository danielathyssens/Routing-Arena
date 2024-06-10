#
import os
import json
import logging
import time
import itertools as it
from typing import Optional, Dict, Union, List, NamedTuple, Tuple, Any
from omegaconf import DictConfig

import numpy as np
import torch
from tensorboard_logger import Logger as TbLogger
from torch.utils.data import Dataset, DataLoader
from formats import TSPInstance, CVRPInstance, RPSolution
from models.NeuOpt.NeuOpt.agent.ppo import PPO
from models.NeuOpt.NeuOpt.problems.problem_cvrp import CVRP
from models.NeuOpt.NeuOpt.problems.problem_tsp import TSP
from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset

logger = logging.getLogger(__name__)


class NeuOptDataset(Dataset):
    def __init__(self,
                 data: Union[List[CVRPInstance], List[TSPInstance]] = None,
                 graph_size: int = None,
                 dummy_rate: Optional[float] = None,
                 offset=0,
                 is_train=False
                 ):
        super(NeuOptDataset, self).__init__()

        if data is not None:
            print('data[0].graph_size', data[0].graph_size)
            assert graph_size == data[0].graph_size - 1 if isinstance(data[0], CVRPInstance) else data[0].graph_size
        self.size = int(np.ceil(graph_size * (1 + dummy_rate)))  # the number of real nodes plus dummy nodes in cvrp/tsp
        self.real_size = graph_size  # the number of real nodes in cvrp /tsp
        self.depot_reps = (self.size - self.real_size)
        self.offset = offset
        if not is_train:
            self.data = [self.make_instance(d) for d in data[offset:offset + len(data)]]

    def make_instance(self, instance: [TSPInstance, CVRPInstance]):
        if isinstance(instance, TSPInstance):
            return {'coordinates': torch.FloatTensor(instance.coords)}
        elif isinstance(instance, CVRPInstance):
            # depot = torch.from_numpy(instance.coords[0])  # torch.tensor(instance.coords[0], dtype=torch.float32)
            depot = torch.tensor(instance.coords[0], dtype=torch.float32)
            # loc = torch.from_numpy(instance.coords[1:])  #
            loc = torch.tensor(instance.coords[1:], dtype=torch.float32)
            # demand = torch.from_numpy(instance.node_features[1:, instance.constraint_idx[0]])
            demand = torch.tensor(instance.node_features[1:, instance.constraint_idx[0]], dtype=torch.float32)
            return {
                'coordinates': torch.cat((depot.view(-1, 2).repeat(self.depot_reps, 1), loc), 0),
                'demand': torch.cat((torch.zeros(self.depot_reps), demand), 0)
            }
        else:
            raise NotImplementedError

    def make_dact_instances(self, data: Union[List[TSPInstance], List[CVRPInstance]]):
        return [self.make_instance(instance=inst) for inst in data[self.offset:self.offset + len(data)]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_neuopt_instances(data: Union[List[TSPInstance], List[CVRPInstance]], dummy_rate=None, offset=0):
    return [make_neuopt_instance(instance=inst, dummy_rate=dummy_rate) for inst in data[offset:offset + len(data)]]


def make_neuopt_instance(instance: [TSPInstance, CVRPInstance], dummy_rate=None):
    if isinstance(instance, TSPInstance):
        return {'coordinates': torch.FloatTensor(instance.coords)}
    elif isinstance(instance, CVRPInstance):
        size = int(np.ceil(instance.graph_size * (1 + dummy_rate)))  # the number of real nodes plus dummy nodes in cvrp
        real_size = instance.graph_size  # the number of real nodes in cvrp
        depot_reps = (size - real_size)
        depot = torch.from_numpy(instance.coords[0])  # torch.tensor(instance.coords[0], dtype=torch.float32)
        loc = torch.from_numpy(instance.coords[1:])  # torch.tensor(instance.coords[1:], dtype=torch.float32)
        demand = torch.from_numpy(instance.node_features[1:, instance.constraint_idx[0]])
        return {
            'coordinates': torch.cat((depot.view(-1, 2).repeat(depot_reps, 1), loc), 0),
            'demand': torch.cat((torch.zeros(depot_reps), demand), 0)
        }
    else:
        raise NotImplementedError


def sol_to_list(sol: Union[np.ndarray, List], depot_idx: int = 0) -> List[List]:
    lst, sol_lst = [], []
    for n in sol:
        if n == depot_idx:
            if len(lst) > 0:
                sol_lst.append(lst)
                lst = []
        else:
            lst.append(n)
    if len(lst) > 0:
        sol_lst.append(lst)
    return sol_lst


def parse_solutions(problem, sols):
    # parse solutions
    s_parsed = None
    if problem.NAME.lower() == "tsp":
        sols = np.concatenate(sols, axis=0)
        sols_torch = torch.from_numpy(sols)
        # print('sols', sols_torch)
        # print('type sols', type(sols_torch))
        sols = get_real_seq_neuopt(sols_torch, problem.NAME.lower())   # [0]
        # print('sols after get_real', sols)
        s_parsed = sols.tolist()

    if problem.NAME.lower() == "cvrp":
        num_dep = problem.dummy_size
        sols = np.concatenate(sols, axis=0)
        s_parsed = []
        # sols_torch = torch.from_numpy(sols)
        # print('sols', sols_torch)
        # print('type sols', type(sols_torch))
        # sols = get_real_seq_dact(sols_torch, problem.NAME.lower())  # [0]
        for sol_ in sols:
            src = 0
            tour_lst, lst = [], []
            for i in range(len(sol_)):
                tgt = sol_[src]
                if tgt < num_dep:
                    if len(lst) > 0:
                        tour_lst.append(lst)
                    lst = []
                else:
                    lst.append(tgt)
                src = tgt
            s_parsed.append([[e - (num_dep - 1) for e in l] for l in tour_lst])

    return s_parsed


def get_real_seq_neuopt(solutions: Union[torch.Tensor], problem: str) -> torch.Tensor:
    # print('solutions', solutions)
    # print(solutions.shape)
    batch_size, seq_length = solutions.size()
    # if problem.lower() == "tsp":

    visited_time = torch.zeros((batch_size, seq_length)).to(solutions.device)
    pre = torch.zeros((batch_size), device=solutions.device).long()
    for i in range(seq_length):
        visited_time[torch.arange(batch_size), solutions[torch.arange(batch_size), pre]] = i + 1
        pre = solutions[torch.arange(batch_size), pre]

    visited_time = visited_time % seq_length
    return visited_time.argsort()
    # else:
    #     visited_time = torch.zeros((batch_size, seq_length)).to(solutions.device)
    #     pre = torch.zeros((batch_size), device=solutions.device).long()
    #     for i in range(seq_length):
    #         visited_time[torch.arange(batch_size), solutions[torch.arange(batch_size), pre]] = i + 1
    #         pre = solutions[torch.arange(batch_size), pre]
    #
    #     visited_time = visited_time % seq_length
    #     return visited_time.argsort()

def train_prep(opts, agent):
    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem,
                                                                       opts.graph_size), opts.run_name))
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Save arguments so exact configuration can always be found
    # if not opts.no_saving:
    #     with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
    #         json.dump(vars(opts), f, indent=True)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        print("Resuming after {}".format(epoch_resume))
        agent.opts.epoch_start = epoch_resume + 1

    return agent, tb_logger


def train_model(problem: CVRP,
                agent: PPO,
                validation_data: Union[List[CVRPInstance], List[CVRPInstance]],
                train_dataset: Union[TSPDataset, CVRPDataset],
                opts):
    agent, tb_logger = train_prep(opts, agent)

    val_dataset = NeuOptDataset(data=validation_data, graph_size=opts.graph_size, dummy_rate=opts.env_cfg.dummy_rate)

    # use train_dataset class from runner and sample using transform function
    # tp get transformfunction- init DACTDataset class once with correct parameters
    dact_train_dataset = NeuOptDataset(graph_size=opts.graph_size, dummy_rate=opts.env_cfg.dummy_rate, is_train=True)
    train_dataset.transform_func = dact_train_dataset.make_neuopt_instances

    agent.start_training(problem, val_dataset=val_dataset, train_dataset_class=train_dataset, tb_logger=tb_logger)

    return None


#
def eval_model(data: Union[List[TSPInstance], List[CVRPInstance]],
               problem: Union[TSP, CVRP],
               agent: PPO,
               time_limit: Union[float, int],
               opts: Union[DictConfig, NamedTuple],
               batch_size: int,
               dummy_rate: Optional[float] = None,
               device=torch.device("cpu")
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # eval mode
    if device.type != "cpu":
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # opts = agent.opts
    agent.eval()
    problem.eval()
    logger.info(f'Inference with {opts.num_augments} augments...')
    graph_size = opts.graph_size if opts.graph_size is not None else data[0].graph_size-1
    val_dataset = NeuOptDataset(data, graph_size=graph_size, dummy_rate=dummy_rate)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

    sols, final_internal_costs, times, sol_time_trajs = [], [], [], []
    for i, batch in enumerate(val_dataloader):
        t_start = time.time()
        res = agent.rollout(
            problem,
            opts.T_max,
            opts.num_augments,
            opts.stall,
            batch,
            Time_Budget=time_limit if time_limit is not None else data[i].time_limit,
            # do_sample=True,
            record=True,
            show_bar=True,
            t_start_=t_start
        )
        t = time.time() - t_start
        t_per_inst = t / batch_size
        sols.append(res[-1].cpu().detach().numpy())
        for j in range(batch_size):
            final_internal_costs.append(res[0].cpu().detach().numpy()[j])
            sol_time_trajs.append([res[-2][j+d] for d in range(0,len(res[-2]), batch_size)])
        times.append([t_per_inst] * batch_size)

    final_s_parsed = parse_solutions(problem, sols)
    running_ts, running_sols = [], []
    for traj in sol_time_trajs:
        rn_ts, rn_sl = [], []
        for tr in traj:
            t, sols_ = tr[0], np.array(tr[1])
            sols_parsed = parse_solutions(problem, list([sols_]))
            rn_ts.append(t / batch_size)
            rn_sl.append(sols_parsed[0])
        running_ts.append(rn_ts)
        running_sols.append(rn_sl)

    times = list(it.chain.from_iterable(times))
    solutions = [
        RPSolution(
            solution=sol,   # if opts.problem.upper() == 'CVRP' else sol_to_list(sol),
            # cost=c.tolist(),
            method_internal_cost=c,
            num_vehicles=len(sol) if opts.problem == 'CVRP' else len([sol]),
            run_time=t,
            running_sols=r_sol,
            running_times=r_t,
            problem=opts.problem,
            instance=inst
        )
        for sol, c, t, r_sol, r_t, inst in zip(final_s_parsed, final_internal_costs, times, running_sols, running_ts, data)
    ]

    return {}, solutions
