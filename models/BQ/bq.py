from typing import List, Tuple, Dict, Union, NamedTuple, Any
from omegaconf import DictConfig
from argparse import Namespace
import logging
import torch
import time
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate

import os.path
import numpy as np
from scipy.spatial.distance import pdist, squareform
# from # models.BQ.bq_nco.data.solvers.
from concorde.tsp import TSPSolver

SCALE = 1e6

from formats import RPSolution, CVRPInstance, TSPInstance
from models.BQ.bq_nco.model.model import BQModel
# from models.BQ.bq_nco.learning.tsp.data_iterator import DataIterator  --> copy here
from models.BQ.bq_nco.learning.tsp.traj_learner import TrajectoryLearner as TrajectoryLearner_tsp
from models.BQ.bq_nco.learning.cvrp.traj_learner import TrajectoryLearner as TrajectoryLearner_cvrp
from models.BQ.bq_nco.data.prepare_cvrp_dataset import reorder
from models.BQ.bq_nco.utils.chekpointer import CheckPointer
from models.BQ.bq_nco.utils.misc import set_seed, print_model_params_info, maybe_cuda_ddp

logger = logging.getLogger(__name__)


def eval_model(net: BQModel,
               checkpointer,
               module,
               data_rp: List,
               problem: str,
               batch_size: int,
               device: torch.device,
               opts: Union[DictConfig, NamedTuple],
               normalized_data: bool,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # eval mode
    # if device.type != "cpu":
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False

    # opts = agent.opts
    # model.eval()

    # logger.info(f'Inference with {opts.num_augments} augments...')

    # prep data for BQ model --> from RPInstance to BQ format
    data = prep_data_BQ(data_rp, what="test", normalized_data=normalized_data, test_only=True)
    # data = data[:2]
    data_iterator = DataIterator(data, args=opts, problem=problem.lower())

    TrajectoryLearner = TrajectoryLearner_tsp if problem.lower() == "tsp" else TrajectoryLearner_cvrp

    traj_learner = TrajectoryLearner(opts, net, module, device, data_iterator, checkpointer=checkpointer)
    # (for eval, no need for optimizer, watcher, checkpointer)

    start_time = time.time()
    res, tours, costs = traj_learner.val_test()
    total_inference_time = time.time() - start_time
    print(f"Total Inference time {total_inference_time:.3f}s")
    # print('res', res)
    # print('tours', tours)
    # print('costs', costs)

    # sols, times, costs = [], [], []
    # for batch in tqdm(dataloader, disable=opts.no_progress_bar):
    #     batch = move_to(batch, device)
    #     t_start = time.time()
    #     with torch.no_grad():
    #         set_decode_type(model, "greedy")
    #         _, cost, res = model(batch, beam_size=opts.beam_size, fst=1, return_pi=True)
    #         t = time.time() - t_start
    #         t_per_inst = t / batch_size
    #         costs.append(cost)
    #         sols.append(res.cpu().numpy().tolist())
    #         times.append([t_per_inst] * batch_size)

    # results = torch.cat(results, 0)
    #
    times = [total_inference_time / len(data_rp)] * len(data_rp)
    # print('times', times)

    return {}, make_RPSolution(problem, tours, costs, times, data_rp)


def train_model(net: BQModel,
                module,
                data_rp: Union[List, dict],
                # data_test: Union[List, dict],
                problem: str,
                device: torch.device,
                opts: Union[DictConfig, NamedTuple],
                normalized_data: bool = True,
                normalize_demands: bool = False,
                pretrained_model: str = None,
                ) -> Dict[str, Any]:
    # from models.BQ.bq_nco.utils.exp import setup_exp
    # args, net, module, device, problem="tsp", is_test=False):
    optimizer, checkpointer, other = setup_exp_(opts, net, module, device,
                                                problem=problem.lower(),
                                                pretrained_model=pretrained_model,
                                                is_test=False)

    # Set or re-set iteration counters and other variables from checkpoint reload
    epoch_done = 0 if other is None else other['epoch_done']
    best_current_val_metric = float('inf') if other is None else other['best_current_val_metric']
    print('epoch_done', epoch_done)
    print('best_current_val_metric', best_current_val_metric)
    # prep data for BQ model --> from RPInstance to BQ format
    # for train - reorder has to be True
    train_portion = int(opts.train_frac * len(data_rp))  # ["solutions"]
    test_val_portion = train_portion + int((len(data_rp) - train_portion) / 2)
    print('train_portion', train_portion)
    print('test_val_portion', test_val_portion)
    data_train = prep_data_BQ(data_rp[:train_portion],  # [:200],
                              what="train", reorder_=True,
                              normalized_data=normalized_data,
                              normalize_demands=normalize_demands)
    # for val - reorder has to be False
    data_val = prep_data_BQ(data_rp[train_portion:test_val_portion],  # [200:250],
                            what="val", reorder_=False, normalized_data=normalized_data,
                            normalize_demands=normalize_demands)
    # print('len(data_test)', len(data_test))
    data_test = prep_data_BQ(data_rp[test_val_portion:],  # [250:300]
                             what="test", reorder_=False,
                             normalized_data=normalized_data,
                             normalize_demands=normalize_demands)
    # (self, test_dataset: Union[np.array, dict],
    #                  train_val_set: Union[np.array, str, tuple],
    #                  args: DictConfig, problem: str = "tsp",
    #                  what: str = 'test'):
    data_iterator = DataIterator(data_test, (data_train, data_val),
                                 args=opts, problem=problem.lower(), what="train")

    TrajectoryLearner = TrajectoryLearner_tsp if problem.lower() == "tsp" else TrajectoryLearner_cvrp

    traj_learner = TrajectoryLearner(opts, net, module, device, data_iterator,
                                     optimizer=optimizer,
                                     checkpointer=checkpointer)
    # (for eval, no need for optimizer, watcher, checkpointer)

    start_time = time.time()
    epochs_done, last_val_metrics = traj_learner.train()
    total_train_time = time.time() - start_time
    print(f"Total Train time {total_train_time:.3f}s")
    print("Saving final model...")
    traj_learner.save_model(label='current', complete=True)

    return {'epochs_done': epoch_done,
            'total_train_time': total_train_time,
            'validation_metrics': last_val_metrics}


# utilities for eval and Train;

def prep_data_BQ(data: Union[List[TSPInstance], List[CVRPInstance], List[RPSolution]],
                 what: str, normalized_data: bool, normalize_demands=None, reorder_=False, offset=0, test_only=False):
    # solutions: List[RPSolution] = None):
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    if isinstance(data[0], RPSolution):
        dat = [d.instance for d in data]
        # solutions = data
    else:
        dat = data
    if isinstance(dat[0], TSPInstance):
        print('what', what)
        print('reorder_', reorder_)
        coords, tours, tour_lens = list(), list(), list()

        all_instance_coords = np.stack([inst.coords for inst in dat], axis=0)
        # print('all_instance_coords.shape', all_instance_coords.shape)

        for i, instance_coords in enumerate(all_instance_coords):
            solver = TSPSolver.from_data(instance_coords[:, 0] * SCALE, instance_coords[:, 1] * SCALE, norm="EUC_2D")
            solution = solver.solve()  # data[i].solution
            solution_closed_tour = list(solution[0]) + [0]  # solution + [0]

            if reorder_:
                coords_reordered = instance_coords[np.array(solution_closed_tour)]
                coords.append(coords_reordered)

            else:
                instance_coords = instance_coords.tolist()
                instance_coords.append(instance_coords[0])

                # compute tour length
                adj_matrix = squareform(pdist(instance_coords, metric='euclidean'))
                tour_len = sum([adj_matrix[solution_closed_tour[i], solution_closed_tour[i + 1]]
                                for i in range(len(solution_closed_tour) - 1)])
                tour_lens.append(tour_len)
                coords.append(instance_coords)

        #  'tour_lens': tour_lens
        if reorder_:
            return {'coords': np.array(coords), 'reorder': reorder}
        else:
            tour_lens = np.stack(tour_lens) if not test_only else None
            # print('capacities[0]', capacities[0])
            # print('coords[0]', coords[0][:5])
            # print('demands[0]', demands[0])
            # print('tour_lens[0]', tour_lens[0])
            return {'coords': coords,
                    'tour_lens': tour_lens,
                    'reorder': False}

    elif isinstance(dat[0], CVRPInstance):
        print('reorder_', reorder_)
        # cvrp_dat = [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]
        all_coords, all_demands, all_remaining_capacities, all_capacities = [], [], [], []
        all_tour_lens, all_via_depots = [], []
        for i, args in enumerate(dat[offset:offset + len(dat)]):
            # add first node to the end
            coords = args.coords.tolist()
            coords.append(coords[0])
            demands = args.node_features[:, args.constraint_idx[0]]
            if not normalize_demands and (demands < 1).all():
                # demands = args.node_features[:, args.constraint_idx[0]]
                # print('demands', demands)
                demands = demands * args.original_capacity
                # print('demands 1    ', demands)
                demands = demands.tolist()
            else:
                demands = args.node_features[:, args.constraint_idx[0]].tolist()
            demands.append(demands[0])
            coords = np.array(coords)
            coords = np.clip(coords, a_min=0.0, a_max=1.0)
            demands = np.array(demands).astype(int)
            if what in ["train", "val", "test"] and not test_only:  # "test"
                adj_matrix = squareform(pdist(coords, metric='euclidean'))
                tours = transform_tours_in_train(data[i].solution)
                tour_len = sum([adj_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)])
            else:
                tours = None
                tour_len = None
            if reorder_:
                coords, demands, remaining_capacities, via_depots = reorder(coords,
                                                                            demands,
                                                                            args.original_capacity,
                                                                            tours)
                #print
                # None, None, None, None)
                # reorder(coords, demands, args.capacity, tours))
            else:
                remaining_capacities, via_depots = None, None

            all_coords.append(coords)
            all_demands.append(demands)
            all_remaining_capacities.append(remaining_capacities)
            all_via_depots.append(via_depots)
            all_tour_lens.append(tour_len)
            all_capacities.append(args.original_capacity)

        logger.info(f'Number of {what} instances: {len(all_coords)}')
        capacities = np.stack(all_capacities)
        coords = np.stack(all_coords)
        demands = np.stack(all_demands)

        if reorder_:
            remaining_capacities = np.stack(all_remaining_capacities)
            via_depots = np.stack(all_via_depots)
            # print('capacities[0]', capacities[0])
            # print('coords[0]', coords[0][:5])
            # print('demands[0]', demands[0])
            # print('remaining_capacities[0]', remaining_capacities[0])
            # print('via_depots[0]', via_depots[0])
            return {'capacities': capacities,
                    'coords': coords,
                    'demands': demands,
                    'remaining_capacities': remaining_capacities,
                    'via_depots': via_depots,
                    'reorder': True}
        else:
            tour_lens = np.stack(all_tour_lens) if not test_only else None
            # print('capacities[0]', capacities[0])
            # print('coords[0]', coords[0][:5])
            # print('demands[0]', demands[0])
            # print('tour_lens[0]', tour_lens[0])
            return {'capacities': capacities,
                    'coords': coords,
                    'demands': demands,
                    'tour_lens': tour_lens,
                    'reorder': False}
    else:
        raise NotImplementedError


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


def make_RPSolution(problem, sols, costs, times, instances) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sol_list = [_get_sep_tours(problem, instance.graph_size, sol_) for sol_, instance in zip(sols, instances)]

    return [
        RPSolution(
            solution=sol_list[i],
            cost=costs[i].item() if sol_list[i] is not None else None,
            method_internal_cost=costs[i].item() if sol_list[i] is not None else None,
            num_vehicles=len(sol_list[i]) if sol_list[i] is not None else None,
            run_time=times[i] if sol_list[i] is not None else None,
            problem=problem,
            instance=instances[i],
        )
        for i in range(len(sols))
    ]


def _get_sep_tours(problem: str, graph_size: int, tours: torch.Tensor) -> Union[List[List], None]:
    """get solution (res) as List[List]"""
    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        return tours.tolist()

    elif problem.lower() == 'cvrp':
        it = iter(tours[0][:-1])  # delete last item which is node id N+1 (e.g. 101)
        check = [tours[0][0]] if tours[0][0] != 0 else []

        tours_list_k = [[0, next(it)]] if tours[0][0] != 0 else [[next(it)]]
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


def transform_tours_in_train(solution_tours: list):
    return np.array([targ for targ_ in solution_tours for targ in targ_[:-1]] + [0])


class DataIterator:

    def __init__(self, test_dataset: Union[np.array, dict],
                 train_val_set: Union[np.array, str, tuple] = None,
                 args: DictConfig = None, problem: str = "tsp",
                 what: str = 'test'):
        load_dataset = load_dataset_tsp if problem == "tsp" else load_dataset_cvrp
        if what in ["train", "val"]:
            if isinstance(train_val_set, tuple):
                # if args.train_dataset is not None
                train_dataset, val_dataset = train_val_set
            else:
                train_dataset, val_dataset = None, None
            # we have a training
            self.train_trajectories = load_dataset(train_dataset,
                                                   args.train_batch_size,
                                                   True, "train")

            # if args.val_dataset is not None:
            # print('val_dataset[0]', val_dataset[0])
            self.val_trajectories = load_dataset(val_dataset,
                                                 args.val_batch_size,
                                                 False, "val")

        self.test_trajectories = load_dataset(test_dataset,
                                              args.test_batch_size,
                                              False, "test")


def load_dataset_tsp(data, batch_size, shuffle=False, what="test"):  # changed --> filename to data
    # data = np.load(filename)
    from models.BQ.bq_nco.learning.tsp.dataloading.dataset import collate_func_with_sample_suffix, DataSet
    if what == "train":
        assert data["reorder"]

    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample_suffix if what == "train" else None

    dataset = DataLoader(DataSet(data["coords"], tour_lens=tour_lens), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset


# --> changed input
def load_dataset_cvrp(data, batch_size, shuffle=False, what="test"):
    # data = np.load(filename)
    from models.BQ.bq_nco.learning.cvrp.dataloading.dataset import collate_func_with_sample, DataSet
    if what == "train":
        assert data["reorder"]
    # print('WHAT', what)
    node_coords = data["coords"]
    demands = data["demands"]
    capacities = data["capacities"]
    # print('len(node_coords[0])', len(node_coords[0]))
    # print('node_coords[0]', node_coords[0])
    # print('demands[0] in load', demands[0])
    # print('len(demands[0])', len(demands[0]))
    # print('capacities[0]', capacities[0])

    # in training dataset we have via_depots and remaining capacities but not tour lens
    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
    remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
    via_depots = data["via_depots"] if "via_depots" in data.keys() else None
    # if what == "train":
    #     print('remaining_capacities', remaining_capacities[0])
    #     print('via_depots[0]', via_depots[0])
    collate_fn = collate_func_with_sample if what == "train" else None

    dataset = DataLoader(DataSet(node_coords, demands, capacities,
                                 remaining_capacities=remaining_capacities,
                                 tour_lens=tour_lens,
                                 via_depots=via_depots), batch_size=batch_size,
                         drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
    return dataset


# copied from bq-nco/utils/exp --> minor changes
# net, module, device,
def setup_exp_(args, net, module, device, problem="tsp", pretrained_model=None, is_test=False):
    print(args)
    set_seed(args.seed)

    print("Using", torch.cuda.device_count(), "GPU(s)")

    for d in [args.output_dir, os.path.join(args.output_dir, "models")]:
        if not os.path.exists(d):
            os.makedirs(d)

    if problem == "tsp":
        node_input_dim = 2
    elif problem == "kp":
        node_input_dim = 3
    elif problem == "cvrp" or problem == "op":
        node_input_dim = 4

    # net = BQModel(node_input_dim,
    # model_args.dim_emb,
    # model_args.dim_ff,
    # model_args.activation_ff,
    # model_args.nb_layers_encoder,
    # model_args.nb_heads,
    # model_args.activation_attention,
    # model_args.dropout, model_args.batchnorm, problem)

    # net, module, device = maybe_cuda_ddp(net)
    # print_model_params_info(module)

    optimizer = None
    if not is_test:
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr) if not args.test_only else None

    print('args.pretrained_model', args.pretrained_model)
    print('pretrained_model', pretrained_model)
    # if args.pretrained_model not in ["", None]:
    if pretrained_model is not None:
        path = pretrained_model # args.pretrained_model
        model_dir, name = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
        checkpointer = CheckPointer(name=name, save_dir=model_dir)
        _, other = checkpointer.load(module, optimizer, label='best', map_location=device)
    else:
        model_dir, name = os.path.join(args.output_dir, "models"), f'{int(time.time() * 1000.0)}'
        checkpointer = CheckPointer(name=name, save_dir=model_dir)
        other = None
        print('model_dir', model_dir)

    return optimizer, checkpointer, other

# def collate_func_with_sample_suffix(l_dataset_items):
#     """
#     assemble minibatch out of dataset examples.
#     For instances of TOUR-tsp of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
#     this function also takes care of sampling a SUB-problem (PATH-tsp) of size 3 to N+1
#     """
#     nb_nodes = len(l_dataset_items[0].nodes_coord)
#     subproblem_size = np.random.randint(4, nb_nodes + 1)
#     begin_idx = nb_nodes + 1 - subproblem_size
#     l_dataset_items_new = prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size)
#     return default_collate(l_dataset_items_new)
#
#
# def prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size):
#     l_dataset_items_new = []
#     for d in l_dataset_items:
#         d_new = {}
#         for k, v in d.items():
#             if type(v) == np.float64:
#                 v_ = 0.
#             elif len(v.shape) == 1 or k == 'nodes_coord':
#                 v_ = v[begin_idx:begin_idx+subproblem_size, ...]
#             else:
#                 v_ = v[begin_idx:begin_idx+subproblem_size, begin_idx:begin_idx+subproblem_size]
#             d_new.update({k+'_s': v_})
#         l_dataset_items_new.append({**d, **d_new})
#     return l_dataset_items_new
#
#
# def sample_subproblem(nb_nodes):
#     subproblem_size = np.random.randint(4, nb_nodes + 1)  # between _ included and nb_nodes + 1 excluded
#     begin_idx = np.random.randint(nb_nodes - subproblem_size + 1)
#     return begin_idx, subproblem_size
#
#
#
# class DotDict(dict):
#     def __init__(self, **kwds):
#         self.update(kwds)
#         self.__dict__ = self
#
#
# class DataSet(Dataset):
#
#     def __init__(self, node_coords, tour_lens=None):
#         self.node_coords = node_coords
#         self.tour_lens = tour_lens
#
#     def __len__(self):
#         return len(self.node_coords)
#
#     def __getitem__(self, item):
#         node_coords = self.node_coords[item]
#         dist_matrix = squareform(pdist(node_coords, metric='euclidean'))
#
#         # From list to tensors as a DotDict
#         item_dict = DotDict()
#         item_dict.dist_matrices = torch.Tensor(dist_matrix)
#         item_dict.nodes_coord = torch.Tensor(node_coords)
#         if self.tour_lens is not None:
#             item_dict.tour_len = self.tour_lens[item]
#         else:
#             item_dict.tour_len = torch.Tensor([])
#
#         return item_dict

# def train_model(
#         model: BQNet,
#         problem: str,
#         train_dataset: Union[TSPDataset, CVRPDataset],
#         val_dataset: Union[TSPDataset, CVRPDataset],
#         device: torch.device,
#         baseline_type: str,
#         resume: bool,
#         ckpt_save_path: str,
#         opts: Union[DictConfig, Dict],
#         tb_logger=None,
#         **sampling_args
# ):
#     epoch_rewards = []
#     optimizer, baseline, lr_scheduler = train_prep(model=model,
#                                                    problem=problem,
#                                                    device=device,
#                                                    baseline_type=baseline_type,
#                                                    resume=resume,
#                                                    opts=opts,
#                                                    **sampling_args)
#
#     for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
#         train_data_samples = train_dataset.sample(sample_size=opts.epoch_size)
#         step, avg_epoch_reward = train_epoch(
#             model,
#             optimizer,
#             baseline,
#             lr_scheduler,
#             epoch,
#             train_data_samples,
#             val_dataset,
#             tb_logger,
#             ckpt_save_path,
#             opts
#         )
#         epoch_rewards.append((step, avg_epoch_reward))
#
#     return epoch_rewards
#
# def train_prep(model: AttentionModel,
#                opts: Union[DictConfig, NamedTuple],
#                baseline_type: str,
#                problem: str,
#                device: torch.device,
#                resume: bool,
#                resume_pth: str = None,
#                epoch_resume: int = 99,
#                **kwargs):
#     # Pretty print the run args
#     logger.info(f"Train cfg:")
#     pp.pprint(opts)
#
#     torch.backends.cuda.matmul.allow_tf32 = True
#
#     # Initialize baseline
#     problem_mdam = load_problem(problem)
#     if baseline_type == 'exponential':
#         baseline = ExponentialBaseline(opts.exp_beta)
#     elif baseline_type == 'rollout':
#         baseline = RolloutBaseline(model, problem_mdam, opts, device=device, **kwargs)
#     else:
#         assert baseline_type is None, "Unknown baseline: {}".format(opts.baseline)
#         baseline = NoBaseline()
#
#     if opts.bl_warmup_epochs > 0:
#         baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
#
#     # Initialize optimizer
#     optimizer = optim.Adam(
#         [{'params': model.parameters(), 'lr': opts.lr_model}]
#         + (
#             [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
#             if len(baseline.get_learnable_parameters()) > 0
#             else []
#         )
#     )
#
#     if resume:
#         # Load data from load_path
#         # assert opts.load_path is None or resume_pth is None, "Only one of load path and resume can be given"
#         # load_path = opts.load_path if opts.load_path is not None else resume
#         if resume_pth is not None:
#             print('  [*] Loading data from {}'.format(resume_pth))
#             load_data = torch_load_cpu(resume_pth)
#         else:
#             raise FileNotFoundError(f"Path for resuming training: '{resume_pth}' is not specified. Please specify a "
#                                     f"valid checkpoint path to resume training for MDAM or set resume to False")
#
#         torch.set_rng_state(load_data['rng_state'])
#         if device != "cpu":
#             torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
#         # Set the random states
#         # Dumping of state was done before epoch callback, so do that now (model is loaded)
#         baseline.epoch_callback(model, epoch_resume)
#
#         # Load baseline from data, make sure script is called with same type of baseline
#         if 'baseline' in load_data:
#             baseline.load_state_dict(load_data['baseline'])
#
#         # Load optimizer state
#         if 'optimizer' in load_data:
#             optimizer.load_state_dict(load_data['optimizer'])
#             for state in optimizer.state.values():
#                 for k, v in state.items():
#                     # if isinstance(v, torch.Tensor):
#                     if torch.is_tensor(v):
#                         state[k] = v.to(device)
#
#     # Initialize learning rate scheduler, decay by lr_decay once per epoch!
#     lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
#
#     return optimizer, baseline, lr_scheduler
