import logging
import time
import itertools as it
from typing import Dict, Union, List, NamedTuple, Tuple, Any
from omegaconf import DictConfig

import numpy as np
import pprint as pp
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from formats import CVRPInstance, RPSolution
from data import CVRPDataset
from models.template_model.template_model_.model import TemplateModel, train_epoch


logger = logging.getLogger(__name__)


# necessary functions in template_model_ML.py:
# (a) function that transforms CVRPInstances to the data format used by model
# (b) function that transforms the model solutions to the RPSolution format
# (c) function that calls the model’s internal evaluation function and processes the transformations (a) and (b)
# (d) function that transforms CVRPInstances 1 to the data format used by model


# Function (a)
def prep_data(dat: Union[List[CVRPInstance]], offset=0):
    """preprocesses data format for TemplateModel (e.g. from List[NamedTuple] to List[torch.Tensor])"""
    # if isinstance(dat[0], TSPInstance):
    #     return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    if isinstance(dat[0], CVRPInstance):
        return [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]
    else:
        raise NotImplementedError


# Function (b)
def make_RPSolution(problem, sols, costs, times, instances) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sol_list = [_get_sep_tours(problem, instance.graph_size, sol_) for sol_, instance in zip(sols, instances)]
    return [
        RPSolution(
            solution=sol_list[i],
            cost=costs[i].item() if sol_list[i] is not None else None,
            num_vehicles=len(sol_list[i]) if sol_list[i] is not None else None,
            run_time=times[i] if sol_list[i] is not None else 0,
            problem=problem,
            instance=instances[i],
        )
        for i in range(len(sols))
    ]


# Function (c)
def eval_model(model: TemplateModel,
               data_rp: List,
               problem: str,
               batch_size: int,
               device: torch.device,
               opts: Union[DictConfig, NamedTuple]
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:

    # logger.info(f'Inference with {opts.num_augments} augments...')

    data = prep_data(data_rp)
    dataloader = DataLoader(data, batch_size=batch_size)

    sols, times, costs = [], [], []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        t_start = time.time()
        with torch.no_grad():
            _, cost, res = model(batch, beam_size=opts.beam_size, fst=1, return_pi=True)
            t = time.time() - t_start
            t_per_inst = t / batch_size
            costs.append(cost)
            sols.append(res.cpu().numpy().tolist())
            times.append([t_per_inst] * batch_size)

    times = list(it.chain.from_iterable(times))

    return {}, make_RPSolution(problem, sols, costs, times, data_rp)


# Function (d)
def train_model(
        model: TemplateModel,
        problem: str,
        train_dataset: CVRPDataset,
        val_dataset: CVRPDataset,
        device: torch.device,
        baseline_type: str,
        resume: bool,
        ckpt_save_path: str,
        opts: Union[DictConfig, Dict],
        tb_logger=None,
        **sampling_args
):
    # note that we do not need to prepare training data (transformation function is given to "train_dataset" in runner)

    epoch_rewards = []
    optimizer, lr_scheduler = train_prep(model=model,
                                         problem=problem,
                                         device=device,
                                         baseline_type=baseline_type,
                                         resume=resume,
                                         opts=opts,
                                         **sampling_args)

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_data_samples = train_dataset.sample(sample_size=opts.epoch_size)
        step, avg_epoch_reward = train_epoch(
            ...
        )
        epoch_rewards.append((step, avg_epoch_reward))

    return epoch_rewards


# Some additional helper functions
def _get_sep_tours(problem: str, graph_size: int, tours: torch.Tensor) -> Union[List[List], None]:
    """get solution (res) as List[List]"""
    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        return tours.tolist()[0]

    elif problem.lower() == 'cvrp':
        it = iter(tours[0])
        check = [tours[0][0]] if tours[0][0] != 0 else []

        tours_list_k = [[0, next(it)]]
        for ele in it:
            if ele != 0:
                tours_list_k[-1].append(ele)
                check.append(ele)
            else:
                tours_list_k[-1].append(0)
                tours_list_k.append([ele])
        tours_list_k[-1].append(0)

        return tours_list_k


def make_cvrp_instance(args, distribution_args=None):
    depot = args.coords[args.depot_idx[0]]
    loc = args.coords[1:, :]
    demand = args.node_features[1:, args.constraint_idx[0]]
    capacity = args.vehicle_capacity
    grid_size = 1
    if distribution_args is not None:
        depot_types, customer_types, grid_size = distribution_args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # / capacity -> demands already normalized
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


# Transform solution returned as array to List[List]
def sol_to_list(sol: np.ndarray, depot_idx: int = 0) -> List[List]:
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


def train_prep(model: TemplateModel,
               opts: Union[DictConfig, NamedTuple],
               problem: str,
               device: torch.device,
               resume: bool,
               resume_pth: str = None,
               epoch_resume: int = 99,
               **kwargs):
    # Pretty print the run args
    logger.info(f"Train cfg:")
    pp.pprint(opts)

    # Initialize baseline
    ...
    # Initialize optimizer
    optimizer = optim.Adam(
        ...
    )

    if resume:
        # Load data from load_path
        # baseline.epoch_callback(model, epoch_resume)
        ...

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    return optimizer, lr_scheduler
