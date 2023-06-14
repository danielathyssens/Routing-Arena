import logging
import time
import os
import warnings
import json

import psutil
import itertools
from typing import Dict, Union, List, NamedTuple, Tuple, Any
from omegaconf import DictConfig
# import psutil

import numpy as np
import pprint as pp
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar
from sklearn.utils.class_weight import compute_class_weight
# from tensorboard_logger import Logger

from formats import TSPInstance, CVRPInstance, RPSolution

from models.DPDP.DPDP.utils.data_utils import load_dataset, load_heatmaps
from models.DPDP.DPDP.export_heatmap_ import generate_heatmap
from models.DPDP.DPDP.utils.functions import move_to, get_durations, compute_batch_costs, accurate_cdist
from models.DPDP.DPDP.problems import load_problem
from models.DPDP.DPDP.config import get_config
from models.DPDP.DPDP.dp import BatchGraph, StreamingTopK, SimpleBatchTopK, run_dp
from models.DPDP.DPDP.models.gcn_model_vrp import ResidualGatedGCNModelVRP
from models.DPDP.DPDP.models.gcn_model import ResidualGatedGCNModel
from models.DPDP.DPDP.models.sparse_wrapper import wrap_sparse
from models.DPDP.DPDP.models.prep_wrapper import PrepWrapResidualGatedGCNModel
# TRAIN IMPORTS
# from models.DPDP.DPDP.main_vrp import test, metrics_to_str, mean_tour_len_edges
#from models.DPDP.DPDP.problems.vrp.vrp_reader import VRPReader

# Fix according to https://discuss.pytorch.org/t/
# a-call-to-torch-cuda-is-available-makes-an-unrelated-multi-processing-computation-crash/4075/4
mp = torch.multiprocessing.get_context('spawn')

logger = logging.getLogger(__name__)


def train_model(model: object,
                config_pth: str,
                val_data_rp: List,
                problem: str,
                opts: Union[DictConfig, NamedTuple],
                device: torch.device,
                resume: bool,
                resume_pth: str = None,
                epoch_resume: int = 99,
                epoch_start: int = 0,
                n_epochs: int = 100):
    """Training wrapper for DPDP.
       Orig Function Call: python main_vrp.py --config configs/vrp_nazari100.json"""

    # Pretty print the run args
    pp.pprint(opts)

    # get config
    config = load_cfg_model(problem, cfg_path=config_pth)

    if torch.cuda.is_available():
        print("CUDA available, using {} GPUs".format(torch.cuda.device_count()))
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        print("CUDA not available")
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    # transform validation data
    # val_data = prep_data_DPDP(problem, val_data_rp)

    # Instantiate the network
    net = nn.DataParallel(ResidualGatedGCNModelVRP(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()
    print(net)
    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)

    # Create log directory
    log_dir = f"./logs/{config.expt_name}/"
    os.makedirs(log_dir, exist_ok=True)
    json.dump(config, open(f"{log_dir}/config.json", "w"), indent=4)
    writer = SummaryWriter(log_dir)  # Define Tensorboard writer

    # Training parameters
    num_nodes = config.num_nodes
    num_neighbors = config.num_neighbors
    max_epochs = config.max_epochs
    val_every = config.val_every
    test_every = config.test_every
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    accumulation_steps = config.accumulation_steps
    learning_rate = config.learning_rate
    decay_rate = config.decay_rate
    val_loss_old = 1e6  # For decaying LR based on validation loss
    best_pred_tour_len = 1e6  # For saving checkpoints
    best_val_loss = 1e6  # For saving checkpoints

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(optimizer)
    dataset = VRPReader(
        config.num_nodes, config.num_neighbors, config.batch_size,
        config.train_filepath, config.train_filepath_solution  # note training solutions (targets) need to be given
    )

    if 'resume_from_dir' in config:
        if torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(config.resume_from_dir, "last_train_checkpoint.tar"))
        else:
            checkpoint = torch.load(os.path.join(config.resume_from_dir, "last_train_checkpoint.tar"),
                                    map_location='cpu')
        # Load network state
        net.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load other training parameters
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        # Note: the learning_rate was set in load_state_dict,
        # this is just to have the local variable for logging
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        epoch = -1
    epoch_bar = master_bar(range(epoch + 1, max_epochs))
    for epoch in epoch_bar:
        # Log to Tensorboard
        writer.add_scalar('learning_rate', learning_rate, epoch)

        # Train
        train_time, train_loss, train_err_edges, train_err_tour, train_err_tsp, train_pred_tour_len, train_gt_tour_len = train_one_epoch(
            net, optimizer, config, epoch_bar, dataset=dataset)
        epoch_bar.write(
            't: ' + metrics_to_str(epoch, train_time, learning_rate, train_loss, train_err_edges, train_err_tour,
                                   train_err_tsp, train_pred_tour_len, train_gt_tour_len))
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('pred_tour_len/train_pred_tour_len', train_pred_tour_len, epoch)
        writer.add_scalar('optimality_gap/train_opt_gap', train_pred_tour_len / train_gt_tour_len - 1, epoch)

        if epoch % val_every == 0 or epoch == max_epochs - 1:
            # Validate
            val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len = test(net,
                                                                                                                    config,
                                                                                                                    epoch_bar,
                                                                                                                    mode='val')
            epoch_bar.write(
                'v: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_err_edges, val_err_tour,
                                       val_err_tsp, val_pred_tour_len, val_gt_tour_len))
            writer.add_scalar('loss/val_loss', val_loss, epoch)
            writer.add_scalar('pred_tour_len/val_pred_tour_len', val_pred_tour_len, epoch)
            writer.add_scalar('optimality_gap/val_opt_gap', val_pred_tour_len / val_gt_tour_len - 1, epoch)

            # Save checkpoint
            if val_pred_tour_len < best_pred_tour_len:
                best_pred_tour_len = val_pred_tour_len  # Update best val predicted tour length
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, log_dir + "best_val_tourlen_checkpoint.tar")

            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update best val loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, log_dir + "best_val_loss_checkpoint.tar")

            # Update learning rate
            if val_loss > 0.99 * val_loss_old:
                learning_rate /= decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            val_loss_old = val_loss  # Update old validation loss

        if epoch % test_every == 0 or epoch == max_epochs - 1:
            # Test
            test_time, test_loss, test_err_edges, test_err_tour, test_err_tsp, test_pred_tour_len, test_gt_tour_len = test(
                net, config, epoch_bar, mode='test')
            epoch_bar.write(
                'T: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_err_edges, test_err_tour,
                                       test_err_tsp, test_pred_tour_len, test_gt_tour_len))
            writer.add_scalar('loss/test_loss', test_loss, epoch)
            writer.add_scalar('pred_tour_len/test_pred_tour_len', test_pred_tour_len, epoch)
            writer.add_scalar('optimality_gap/test_opt_gap', test_pred_tour_len / test_gt_tour_len - 1, epoch)

        # Save training checkpoint at the end of epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, log_dir + "last_train_checkpoint.tar")

        # Save checkpoint after every 250 epochs
        if epoch != 0 and (epoch % 250 == 0 or epoch == max_epochs - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, log_dir + f"checkpoint_epoch{epoch}.tar")

    return net


#
def eval_model(ckpt_pth: str,
               heatmap_pth: str,
               data_rp: List,
               problem_str: str,
               batch_size: int,
               beam_size: int,
               device: torch.device,
               normalization: bool,
               data_dist: str,
               opts: Union[DictConfig, NamedTuple]
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # logger.info(f'Inference with {opts.num_augments} augments...')
    data = prep_data_DPDP(problem_str, data_rp, distr=data_dist, normalization=normalization)

    # load model-specific problem object
    problem = load_problem(problem_str)

    # load model for heatmaps
    model, model_config = load_cfg_model(problem=problem.NAME, checkpoint_path=ckpt_pth)

    # multi-process or not
    device_count = torch.cuda.device_count() if (device != torch.device("cpu")) else 1
    num_processes = opts.num_processes * device_count

    # UPDATE SYSTEM INFO For logging
    opts.system_info = {
        'used_device_count': device_count,
        'used_num_processes': num_processes,
        'devices': ["cpu"] if not (device != torch.device("cpu")) else [torch.cuda.get_device_name(i) for i in
                                                                        range(device_count)],
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (2 ** 30)
    }
    if num_processes > 1:
        assert device != torch.device("cpu"), "Can only do multiprocessing with cuda"
        assert len(data) % num_processes == 0, f"Dataset size {len(data)} must be divisible by {device_count}" \
                                               f" devices x {opts.num_processes} processes = {num_processes} "

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(problem, data, model, model_config, heatmap_pth, beam_size, batch_size, device, opts, i,
                  i % device_count if (device != "cpu") else None, num_processes) for i in
                 range(num_processes)]
            )))

    else:
        dataset_dpdp = pack_heatmaps(problem, heatmap_pth, data, opts)  # heatmaps are empty and generated per batch
        results = _eval_dataset(problem, dataset_dpdp, model, model_config, batch_size, beam_size, opts, device,
                                no_progress_bar=opts.no_progress_bar)

    costs, durations, tours = print_statistics(results, batch_size, opts)
    times = durations  # list(itertools.chain.from_iterable(durations))
    return {}, make_RPSolution(problem_str, tours, costs, times, data_rp)


def _eval_dataset(problem, dataset, net_model, model_cfg, batch_size, beam_size, opts, device, no_progress_bar=False):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # ckpt_pth = "logs/vrp_nazari100/best_val_loss_checkpoint.tar"
    results = []
    for batch in tqdm(dataloader, disable=no_progress_bar):
        # generate heatmap for batch
        heatmaps, time_heatmap = generate_heatmap(problem=problem.NAME, net=net_model, config=model_cfg,
                                                  instance=batch, batch_size=1)
        batch = move_to(batch, device)
        heatmaps = move_to(torch.tensor(heatmaps), device)
        start = time.time()
        with torch.no_grad():

            if opts.decode_strategy[:4] in ('dpbs', 'dpdp'):
                # if decode = dpbs or dpdp, make sure dataset is not normalized!
                assert opts.heatmap_threshold is None or opts.knn is None, "Cannot have both"
                assert problem.NAME in ('cvrp', 'tsp', 'tsptw')
                # Deep policy beam search or deep policy dynamic programming = new style implementation

                batch_size = len(batch) if problem.NAME == 'tsp' else len(batch['loc'])
                try:
                    sequences, costs, batch_size = evaluate_dp(
                        problem.NAME == 'cvrp', problem.NAME == 'tsptw', batch, heatmaps=heatmaps,
                        beam_size=beam_size, collapse=opts.decode_strategy[:4] == 'dpdp',
                        score_function=opts.score_function,
                        heatmap_threshold=opts.heatmap_threshold, knn=opts.knn,
                        use_weak_version=opts.decode_strategy[-1] == '-',
                        verbose=opts.verbose
                    )
                except RuntimeError as e:
                    if 'out of memory' in str(e) and opts.skip_oom:
                        print('| WARNING: ran out of memory, skipping batch')
                        sequences = [None] * batch_size
                        costs = [None] * batch_size
                    else:
                        raise e
                # print(' (time.time() - start)',  (time.time() - start))
                # print('time_heatmap', time_heatmap)
                duration = (time.time() - start) + time_heatmap
                costs = compute_batch_costs(problem, batch, sequences, device=device, check_costs=costs)
        assert len(sequences) == batch_size

        # print(sequences, costs)
        for seq, cost in zip(sequences, costs):
            if problem.NAME in ("tsp", "tsptw"):
                if seq is not None:  # tsptw can be infeasible or TSP failed with sparse graph
                    seq = seq.tolist()  # No need to trim as all are same length
            elif problem.NAME == "cvrp":
                if seq is not None:  # Can be failed with sparse graph
                    seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            else:
                assert False, "Unkown problem: {}".format(problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))
    assert len(results) == len(dataset)
    return results


# utilities for Eval and Train;
def prep_data_DPDP(problem: str, dat: Union[List[TSPInstance], List[CVRPInstance]], distr: str = None,
                   normalization: bool = True, offset: int = 0):
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    if problem.lower() == "tsp":
        return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    elif problem.lower() == "cvrp":
        return [make_cvrp_instance(args, normalization, distr) for args in dat[offset:offset + len(dat)]]
    else:
        raise NotImplementedError


def make_RPSolution(problem, sols, costs, times, instances) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    sol_list = [_get_sep_tours(problem, sol_) for sol_ in sols]
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


def _get_sep_tours(problem: str, tours: torch.Tensor) -> List[List]:
    """get solution (res) as List[List]"""
    if problem.lower() == 'tsp':
        # if problem is TSP - only have single tour
        return tours.tolist()[0]

    elif problem.lower() == 'cvrp':
        # print(tours)
        it = iter(tours)
        tours_list_k = [[0, next(it)]]
        for ele in it:
            if ele != 0:
                tours_list_k[-1].append(ele)
            else:
                tours_list_k[-1].append(0)
                tours_list_k.append([ele])
        tours_list_k[-1].append(0)
        return tours_list_k


def make_cvrp_instance(args, normalization, distribution=None, distribution_args=None):
    distribution = 'nazari' if distribution is None else distribution
    # depot, loc, demand, capacity, *args = args
    depot = args.coords[args.depot_idx[0]]
    loc = args.coords[1:, :]
    demand = args.node_features[1:, args.constraint_idx[0]]
    capacity = args.vehicle_capacity
    original_capacity = args.original_capacity
    grid_size = 1000.0 if not normalization else 1.0
    # print('torch.tensor(demand, dtype=torch.float) / capacity', torch.tensor(demand, dtype=torch.float) / capacity)
    if args.instance_id in [25]:
        print('distribution', distribution)
        print('depot', depot)
        print('demand', demand)
        print('original_capacity', original_capacity)
    if distribution_args is not None:
        depot_types, customer_types, grid_size = distribution_args
    if distribution == 'nazari':
        if normalization:
            return {
                'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
                'demand': torch.tensor(demand, dtype=torch.float) / capacity,
                'depot': torch.tensor(depot, dtype=torch.float) / grid_size
            }
        else:
            return {  # Let dtypes be inferred
                'loc': torch.FloatTensor(loc),
                'demand': torch.tensor(demand),
                'depot': torch.FloatTensor(depot),
                'capacity': original_capacity,
                'grid_size': grid_size
            }
    else:
        if normalization:
            return {
                'loc': torch.tensor(loc, dtype=torch.float) / grid_size.float(),
                'demand': torch.tensor(demand, dtype=torch.float) / capacity.float(),
                'depot': torch.tensor(depot, dtype=torch.float) / grid_size.float()
            }
        else:
            return {
                'loc': torch.FloatTensor(loc),
                'demand': torch.tensor(demand), #.int()
                'depot': torch.FloatTensor(depot),
                'capacity': original_capacity,
                'grid_size': grid_size
            }


def evaluate_dp(is_vrp, has_tw, batch, heatmaps, beam_size, collapse, score_function,
                heatmap_threshold, knn, use_weak_version, verbose):
    coords = torch.cat((batch['depot'][:, None], batch['loc']), 1).float() if is_vrp or has_tw else batch
    demands = batch['demand'] if is_vrp else None
    vehicle_capacities = batch['capacity'] if is_vrp else None
    timew = batch['timew'] if has_tw else None
    dist = accurate_cdist(coords, coords)
    quant_c_dt = torch.int32
    if has_tw:
        dist = dist.round()
        assert (dist.max(-1).values.sum(-1) < torch.iinfo(torch.int).max).all()
        assert (timew < torch.iinfo(torch.int).max).all()
        dist = dist.int()
        timew = timew.int()
        quant_c_dt = None  # Don't use quantization since we're using ints already
        batch['dist'] = dist  # For final distance computation

    graph = BatchGraph.get_graph(
        dist, score_function=score_function, heatmap=heatmaps, heatmap_threshold=heatmap_threshold, knn=knn,
        quantize_cost_dtype=quant_c_dt, demand=demands, vehicle_capacity=vehicle_capacities, timew=timew,
        start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    )
    assert graph.batch_size == len(coords)
    add_potentials = graph.edge_weight is not None
    assert add_potentials == ("potential" in score_function.split("_"))

    if False:
        # This implementation is simpler but slower
        candidate_queue = SimpleBatchTopK(beam_size)
    else:
        candidate_queue = StreamingTopK(
            beam_size,
            dtype=graph.score.dtype if graph.score is not None else graph.cost.dtype,
            verbose=verbose,
            payload_dtypes=(torch.int32, torch.int16),  # parent = max 1e9, action = max 2e3 (for VRP with 1000 nodes)
            device=coords.device,
            alloc_size_factor=10. if beam_size * graph.batch_size <= int(1e6) else 2.,
            # up to 1M we can easily allocate 10x so 10MB
            kthvalue_method='sort',  # Other methods may increase performance but are experimental / buggy
            batch_size=graph.batch_size
        )
    mincost_dp_qt, solution = run_dp(
        graph, candidate_queue, return_solution=True, collapse=collapse, use_weak_version=use_weak_version,
        beam_device=coords.device, bound_first=True,  # Always bound first #is_vrp or beam_size >= int(1e7),
        sort_beam_by='group_idx', trace_device='cpu',
        verbose=verbose, add_potentials=add_potentials
    )
    assert len(mincost_dp_qt) == graph.batch_size
    assert len(solution) == graph.batch_size
    solutions_np = [sol.cpu().numpy() if sol is not None else None for sol in solution]
    cost = graph.dequantize_cost(mincost_dp_qt)
    return solutions_np, cost, graph.batch_size


def load_cfg_model(problem: str, checkpoint_path: str = None, cfg_path: str = None, no_prepwrap=False):
    if checkpoint_path is None:  # train
        assert os.path.isfile(cfg_path), "Make sure cfg file for training exists"
        # config_path = "configs/default.json"
        config = get_config(cfg_path)
        print("Loaded {}:\n{}".format(cfg_path, config))
        return config

    else:  # eval
        assert os.path.isfile(checkpoint_path), "Make sure checkpoint file exists"
        # checkpoint_path = args.checkpoint
        log_dir = os.path.split(checkpoint_path)[0]
        config_path = os.path.join(log_dir, "config.json")

        config = get_config(config_path)
        print("Loaded {}:\n{}".format(config_path, config))

        # heatmap_filename = args.output_filename

        # if heatmap_filename is None:
        #     dataset_name = os.path.splitext(os.path.split(args.instances)[-1])[0]
        #     heatmap_dir = os.path.join("results", args.problem, dataset_name, "heatmaps")
        #     heatmap_filename = os.path.join(heatmap_dir, f"heatmaps_{config.expt_name}.pkl")
        # else:
        #     heatmap_dir = os.path.split(heatmap_filename)[0]

        # assert not os.path.isfile(heatmap_filename) or args.f, "Use -f to overwrite existing results"

        if torch.cuda.is_available():
            print("CUDA available, using GPU")
            dtypeFloat = torch.cuda.FloatTensor
            dtypeLong = torch.cuda.LongTensor
            torch.cuda.manual_seed(1)
        else:
            print("CUDA not available")
            dtypeFloat = torch.FloatTensor
            dtypeLong = torch.LongTensor
            torch.manual_seed(1)

        do_prepwrap = not no_prepwrap

        # Instantiate the network
        model_class = ResidualGatedGCNModelVRP if problem == 'cvrp' else ResidualGatedGCNModel
        model = model_class(config, dtypeFloat, dtypeLong)
        if problem in ('tsp', 'tsptw'):
            if 'sparse' in config and config.sparse is not None:
                model = wrap_sparse(model, config.sparse)

            if do_prepwrap:
                assert config.num_neighbors == -1, "PrepWrap only works for fully connected"
                model = PrepWrapResidualGatedGCNModel(model)
        net = nn.DataParallel(model)
        if torch.cuda.is_available():
            net.cuda()

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Load network state
        if problem in ('tsp', 'tsptw'):
            try:
                net.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError:
                # Backwards compatibility
                # Old checkpoints don't contain the PrepWrapModel, so load directly into the nested model
                # (but need to wrap DataParallel)
                nn.DataParallel(model.model).load_state_dict(checkpoint['model_state_dict'])
        else:
            # print("checkpoint['model_state_dict']", checkpoint['model_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])

        print("Loaded checkpoint with epoch", checkpoint['epoch'], 'val_loss', checkpoint['val_loss'])

        return net, config


def load_heatmap(filename: str,
                 instance: Union[TSPInstance, CVRPInstance],
                 problem: str = 'vrp',
                 ckpt_pth: str = 'logs/vrp_nazari100/best_val_loss_checkpoint.tar',
                 symmetric: bool = True):
    if filename is None:
        return None
    heatmaps, *_ = load_dataset(filename)
    if (heatmaps >= 0).all():
        print("Warning: heatmaps where not stored in logaritmic space, conversion may be lossy!")
        heatmaps = np.log(heatmaps)
    return heatmaps if not symmetric else np.maximum(heatmaps, np.transpose(heatmaps, (0, 2, 1)))


class HeatmapDataset(Dataset):

    def __init__(self, dataset=None, heatmaps=None):
        super(HeatmapDataset, self).__init__()

        self.dataset = dataset
        self.heatmaps = heatmaps
        assert (len(self.dataset) == len(
            self.heatmaps)), f"Found {len(self.dataset)} instances but {len(self.heatmaps)} heatmaps"

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'heatmap': self.heatmaps[item]
        }

    def __len__(self):
        return len(self.dataset)


def unpack_heatmaps(batch):
    if isinstance(batch, dict) and 'heatmap' in batch and 'data' in batch:
        return batch['data'], batch['heatmap']
    return batch, None


def pack_heatmaps(prob, heatmap_pth, dataset, opts, offset=None):
    if heatmap_pth is None:
        warnings.warn(f"No heatmaps are loaded for the dataset path. Generate heatmaps first!")
        return dataset
    offset = offset or opts.offset
    # For TSPTW, use undirected heatmap since problem is undirected because of time windows
    return HeatmapDataset(dataset, load_heatmaps(heatmap_pth,
                                                 symmetric=prob.NAME != 'tsptw')[offset:offset + len(dataset)])


def eval_dataset_mp(args):
    # (dataset_path, beam_size, opts, i, device_num, num_processes) = args
    (problem, data_, model, model_cfg, heatmap_pth, beam_size, batch_size, device, opts, i, device_num, num_processes) = args
    # problem = load_problem(opts.problem)
    val_size = len(data_) // num_processes
    # make_dataset_kwargs = {'normalize': False} if opts.decode_strategy[:4] in ('dpbs', 'dpdp') and problem.NAME == 'cvrp' else {}
    # dataset = problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i, **make_dataset_kwargs)
    # dataset_ = pack_heatmaps(problem, heatmap_pth, data_, opts, offset=opts.offset + val_size * i)
    # device = torch.device("cuda:{}".format(device_num) if device_num is not None else 'cpu')
    dataset_dpdp = pack_heatmaps(problem, heatmap_pth, data_, opts)  # heatmaps are empty and generated per batch
    return _eval_dataset(problem, dataset_dpdp, model, model_cfg, batch_size, beam_size, opts, device,
                            no_progress_bar=opts.no_progress_bar)

    # return _eval_dataset(problem, dataset_, beam_size, batch_size, opts, device,
    #                      no_progress_bar=opts.no_progress_bar or i > 0)  # Disable for other processes

# TRAIN FUNCTIONS
def train_one_epoch(net, optimizer, config, master_bar, dataset=None):
    # Set training mode
    net.train()

    # Assign parameters
    num_nodes = config.num_nodes
    num_neighbors = config.num_neighbors
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    accumulation_steps = config.accumulation_steps
    train_filepath = config.train_filepath
    train_target_filepath = config.train_filepath_solution

    if dataset is None:
        dataset = VRPReader(num_nodes, num_neighbors, batch_size, train_filepath, train_target_filepath,
                            do_shuffle=True)
    else:
        dataset.shuffle()
    if batches_per_epoch != -1:
        batches_per_epoch = min(batches_per_epoch, dataset.max_iter)
    else:
        batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)

    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    # running_err_edges = 0.0
    # running_err_tour = 0.0
    # running_err_tsp = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0

    start_epoch = time.time()
    for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        # y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        # Backward pass
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute error metrics and mean tour lengths
        # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size * loss.data.item() * accumulation_steps  # Re-scale loss
        # running_err_edges += batch_size* err_edges
        # running_err_tour += batch_size* err_tour
        # running_err_tsp += batch_size* err_tsp
        running_pred_tour_len += batch_size * pred_tour_len
        running_gt_tour_len += batch_size * gt_tour_len
        running_nb_batch += 1

        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss / running_nb_data,
            pred_tour_len=running_pred_tour_len / running_nb_data,
            gt_tour_len=running_gt_tour_len / running_nb_data))
        master_bar.child.comment = result

    # Compute statistics for full epoch
    loss = running_loss / running_nb_data
    err_edges = 0  # running_err_edges/ running_nb_data
    err_tour = 0  # running_err_tour/ running_nb_data
    err_tsp = 0  # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len / running_nb_data
    gt_tour_len = running_gt_tour_len / running_nb_data

    return time.time() - start_epoch, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len


# PRINT UTILS
def print_statistics(results, bs, opts):
    num_processes = opts.system_info['used_num_processes']
    device_count = opts.system_info['used_device_count']
    batch_size = bs
    assert num_processes % device_count == 0
    num_processes_per_device = num_processes // device_count

    results_stat = [(cost, tour, duration) for (cost, tour, duration) in results if tour is not None]
    if len(results_stat) < len(results):
        failed = [i + opts.offset for i, (cost, tour, duration) in enumerate(results) if tour is None]
        print("*" * 100)
        print("FAILED {} of {} instances, only showing statistics for {} solved instances!".format(
            len(results) - len(results_stat), len(results), len(results_stat)))
        print("Instances failed (showing max 10): ", failed[:10])
        print("*" * 100)
        # results = results_stat
    costs, tours, durations = zip(*results_stat)  # Not really costs since they should be negative
    print("Costs (showing max 10): ", costs[:10])
    if len(tours) == 1:
        print("Tour", tours[0])
    print("Average cost: {:.3f} +- {:.3f}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))

    avg_serial_duration, avg_parallel_duration, total_duration_parallel, total_duration_single_device, effective_batch_size = get_durations(
        durations, batch_size, num_processes, device_count
    )

    print("Average serial duration (per process per device): {:.3f}".format(avg_serial_duration))
    if batch_size > 1:
        print("Average parallel duration (per process per device), effective batch size {:.2f}): {:.3f}".format(
            effective_batch_size, avg_parallel_duration))
    if device_count > 1:
        print(
            "Calculated total duration for {} instances with {} processes x {} devices (= {} proc) in parallel: {}".format(
                len(durations), num_processes_per_device, device_count, num_processes, total_duration_parallel))
    # On 1 device it takes k times longer than on k devices
    print("Calculated total duration for {} instances with {} processes on 1 device in parallel: {}".format(
        len(durations), num_processes_per_device, total_duration_single_device))
    print("Number of GPUs used:", device_count)
    return costs, durations, tours
