
import torch
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List, Union, Optional, Tuple, Dict
import time

from models.AM.AM.utils import load_model, load_problem, move_to, load_args
from models.AM.AM.nets.attention_model import AttentionModel

from formats import TSPInstance, CVRPInstance, RPSolution

def eval_model(problem: str,
               data_rp: Union[List[CVRPInstance], List[TSPInstance]],
               model: AttentionModel,
               eval_batch_size,
               eval_opts,
               device,
               decode_strategy='greedy',
               no_progress_bar=False) -> Tuple[Dict, List[RPSolution]]:

    # data_prep
    data = make_AM_instance(problem, data_rp)
    dataloader = DataLoader(data, batch_size=eval_batch_size)

    # set model to eval
    model.eval()
    model.set_decode_type(
        "greedy" if decode_strategy in ('bs', 'greedy') else "sampling",
        temp=eval_opts.softmax_temperature)

    costs, sols, times = [], [], []
    for instance in tqdm(dataloader, disable=no_progress_bar):
        st_time = time.time()
        # instance, device, eval_opts, width=0
        sol, cost = _eval_instance(model, instance, device, eval_opts)
        times.append(time.time() - st_time)
        costs.append(cost.item())
        sols.append(sol)
    solutions = make_RPSolution(problem, sols, costs, times, data_rp)
    return {}, solutions


def train_model(data: Union[List[CVRPInstance], List[TSPInstance]]):
    pass


def _eval_instance(model, instance, device, eval_opts, width=0):
    instance = move_to(instance, device)
    if eval_opts.width != 0:
        width = eval_opts.width
    # model, _ = load_model(opts.model, device=self.device, is_eval=True, opts=opts)
    # self.set_decode_type("greedy")
    with torch.no_grad():
        if eval_opts.decode_strategy in ('sample', 'greedy'):
            if eval_opts.decode_strategy == 'greedy':
                assert width == 0, "Do not set width when using greedy"
                assert eval_opts.eval_batch_size <= eval_opts.max_calc_batch_size, \
                    "eval_batch_size should be smaller than calc batch size"
                batch_rep = 1
                iter_rep = 1
            elif width * eval_opts.eval_batch_size > eval_opts.max_calc_batch_size:
                assert eval_opts.eval_batch_size == 1
                assert width % eval_opts.max_calc_batch_size == 0
                batch_rep = eval_opts.max_calc_batch_size
                iter_rep = width // eval_opts.max_calc_batch_size
            else:
                batch_rep = width
                iter_rep = 1
            assert batch_rep > 0
            # This returns (batch_size, iter_rep shape)
            sequences, cost = model.sample_many(instance, batch_rep=batch_rep, iter_rep=iter_rep)
            # , ds
            batch_size = len(cost)
            ids = torch.arange(batch_size, dtype=torch.int64, device=cost.device)
        else:
            assert eval_opts.decode_strategy == 'bs'

            cum_log_p, sequences, cost, ids, batch_size = model.beam_search(
                instance, beam_size=width,
                compress_mask=eval_opts.compress_mask,
                max_calc_batch_size=eval_opts.max_calc_batch_size
            )
        return sequences, cost


def make_AM_instance(problem: str,
                     args: Union[CVRPInstance, List[CVRPInstance], TSPInstance, List[TSPInstance]],
                     distribution_args=None, offset=0):

    if problem.upper() == "CVRP":
        # depot, loc, demand, capacity, *args = args
        if not isinstance(args, List):
            depot = args.coords[args.depot_idx[0]]
            loc = args.coords[1:, :]
            demand = args.node_features[1:, args.constraint_idx[0]]
            capacity = args.vehicle_capacity

            grid_size = 1
            if distribution_args is not None:
                depot_types, customer_types, grid_size = distribution_args
            return {
                'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
                'demand': torch.tensor(demand, dtype=torch.float),   # / capacity -> demands already normalized
                'depot': torch.tensor(depot, dtype=torch.float) / grid_size
            }
        else:
            return [make_AM_instance(problem="CVRP", args=args_) for args_ in args[offset:offset + len(args)]]
    else:
        if not isinstance(args, List):
            return torch.FloatTensor(args.coords)
        else:
            return [torch.FloatTensor(row.coords) for row in (args[offset:offset + len(args)])]


def make_RPSolution(problem, sols, costs, times, instances) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sol_list = [get_sep_tours(problem, sol_) for sol_ in sols]

    return [
        RPSolution(
            solution=sol_list[i],
            cost=costs[i],
            num_vehicles=len(sol_list[i]),
            run_time=times[i],
            problem=problem,
            instance=instances[i],
        )
        for i in range(len(sols))
    ]


def get_sep_tours(problem, tours):

    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        return tours.tolist()[0]

    elif problem.lower() == 'cvrp':
        # print('tours: ', tours)
        it = iter(tours[0])
        tours_list_k = [[0, next(it).item()]]
        for ele in it:
            if ele != 0:
                tours_list_k[-1].append(ele.item())
                # print(tours_list_k[-1])
            else:
                tours_list_k[-1].append(0)
                tours_list_k.append([ele.item()])
            # print(tours_list_k)
        tours_list_k[-1].append(0)
        # tours_list_k = tours_list_k[:-1]
        # tours_list_k[-1] = tours_list_k[-1][:-1]
        # print(f'tours_list_k: {tours_list_k}')
        return tours_list_k
