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
from tensorboard_logger import Logger

from formats import TSPInstance, CVRPInstance, RPSolution
from data import TSPDataset, CVRPDataset
from models.MDAM.MDAM.nets.attention_model import set_decode_type
# from models.MDAM.MDAM.nets.attention_model import AttentionModel
from models.MDAM.MDAM.nets.model_search import AttentionModel
from models.MDAM.MDAM.reinforce_baselines import ExponentialBaseline, RolloutBaseline, NoBaseline, WarmupBaseline
from models.MDAM.MDAM.utils import load_problem, torch_load_cpu, move_to
from models.MDAM.MDAM.train import train_epoch
from models.MDAM.MDAM.problems.vrp.problem_vrp import CVRP
from models.MDAM.MDAM.problems.tsp.problem_tsp import TSP

logger = logging.getLogger(__name__)

def train_model(
        model: AttentionModel,
        problem: str,
        train_dataset: Union[TSPDataset, CVRPDataset],
        val_dataset: Union[TSPDataset, CVRPDataset],
        device: torch.device,
        baseline_type: str,
        resume: bool,
        ckpt_save_path: str,
        opts: Union[DictConfig, Dict],
        tb_logger=None,
        **sampling_args
):
    epoch_rewards = []
    optimizer, baseline, lr_scheduler = train_prep(model=model,
                                                   problem=problem,
                                                   device=device,
                                                   baseline_type=baseline_type,
                                                   resume=resume,
                                                   opts=opts,
                                                   **sampling_args)

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_data_samples = train_dataset.sample(sample_size=opts.epoch_size)
        step, avg_epoch_reward = train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            train_data_samples,
            val_dataset,
            tb_logger,
            ckpt_save_path,
            opts
        )
        epoch_rewards.append((step, avg_epoch_reward))

    return epoch_rewards

def train_prep(model: AttentionModel,
               opts: Union[DictConfig, NamedTuple],
               baseline_type: str,
               problem: str,
               device: torch.device,
               resume: bool,
               resume_pth: str = None,
               epoch_resume: int = 99,
               **kwargs):
    # Pretty print the run args
    logger.info(f"Train cfg:")
    pp.pprint(opts)

    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize baseline
    problem_mdam = load_problem(problem)
    if baseline_type == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif baseline_type == 'rollout':
        baseline = RolloutBaseline(model, problem_mdam, opts, device=device, **kwargs)
    else:
        assert baseline_type is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if resume:
        # Load data from load_path
        # assert opts.load_path is None or resume_pth is None, "Only one of load path and resume can be given"
        # load_path = opts.load_path if opts.load_path is not None else resume
        if resume_pth is not None:
            print('  [*] Loading data from {}'.format(resume_pth))
            load_data = torch_load_cpu(resume_pth)
        else:
            raise FileNotFoundError(f"Path for resuming training: '{resume_pth}' is not specified. Please specify a "
                                    f"valid checkpoint path to resume training for MDAM or set resume to False")

        torch.set_rng_state(load_data['rng_state'])
        if device != "cpu":
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])

        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    return optimizer, baseline, lr_scheduler


def eval_model(model: AttentionModel,
               data_rp: List,
               problem: str,
               batch_size: int,
               device: torch.device,
               opts: Union[DictConfig, NamedTuple]
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # eval mode
    # if device.type != "cpu":
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False

    # opts = agent.opts
    model.eval()

    # logger.info(f'Inference with {opts.num_augments} augments...')

    # prep data for MDAM model --> from RPInstance to MDAM format
    data = prep_data_MDAM(data_rp)
    # data = data[:2]
    dataloader = DataLoader(data, batch_size=batch_size)

    sols, times, costs = [], [], []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)
        t_start = time.time()
        with torch.no_grad():
            set_decode_type(model, "greedy")
            _, cost, res = model(batch, beam_size=opts.beam_size, fst=1, return_pi=True)
            t = time.time() - t_start
            t_per_inst = t / batch_size
            costs.append(cost)
            sols.append(res.cpu().numpy().tolist())
            times.append([t_per_inst] * batch_size)

    # results = torch.cat(results, 0)
    #
    times = list(it.chain.from_iterable(times))

    return {}, make_RPSolution(problem, sols, costs, times, data_rp)


# utilities for eval and Train;

def prep_data_MDAM(dat: Union[List[TSPInstance], List[CVRPInstance]], offset=0):
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    if isinstance(dat[0], TSPInstance):
        return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    elif isinstance(dat[0], CVRPInstance):
        return [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]
    else:
        raise NotImplementedError


def make_RPSolution(problem, sols, costs, times, instances) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sol_list = [_get_sep_tours(problem, instance.graph_size, sol_) for sol_, instance in zip(sols, instances)]
    # print('sol_list', sol_list)

    return [
        RPSolution(
            solution=sol_list[i],
            cost=costs[i].item() if sol_list[i] is not None else None,
            method_internal_cost=costs[i].item() if sol_list[i] is not None else None,
            num_vehicles=len(sol_list[i]) if sol_list[i] is not None else None,
            run_time=times[i] if sol_list[i] is not None else 0,
            problem=problem,
            instance=instances[i],
        )
        for i in range(len(sols))
    ]


def _get_sep_tours(problem: str, graph_size: int, tours: torch.Tensor) -> Union[List[List], None]:
    """get solution (res) as List[List]"""
    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        return tours.tolist()[0]

    elif problem.lower() == 'cvrp':
        # print('original tours', tours)
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

        check.sort()
        if check != list(np.arange(1, graph_size)):
            print('original tours', tours)
            print('check', check)
            print('list(', list(np.arange(1, graph_size)))
            print('invalid solution found')
            print('Not all customers are solved!')
            print('Setting solution to None')
            return None

        return tours_list_k


def make_cvrp_instance(args, distribution_args=None):
    # depot, loc, demand, capacity, *args = args
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


# Transform solution retruned from MDAM to List[List]
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
