import tqdm
import time
import numpy as np
import math
import torch
# from torch.utils.data import DataLoader
from models.AM.utils import load_model
from models.AM.utils import move_to


def run_eval_am(dat_x_inst, device, model, width, opts):
    inst_am = move_to(dat_x_inst, device)
    st_am = time.time()
    outs_am = eval_dataset(model, inst_am, width, opts)
    duration = time.time() - st_am
    # res_lst = appnd(res_lst, model, outs_am, duration_am)

    return res_lst, duration


def prelim(opts, data_to_eval, device):
    model, _ = load_model(opts.model)
    model.to(device)
    model.eval()
    model.set_decode_type("greedy" if opts.decode_strategy in ('bs', 'greedy') else
                             "sampling", temp=opts.softmax_temp)
    widths = opts.width if opts.width is not None else [0]
    # if opts.multiprocessing: (left out for now)
    # else:
    dataset = model.problem.make_dataset(filename=data_to_eval,
                                         num_samples=opts.val_size,
                                         offset=opts.offset)

    return model, widths, dataset


def eval_dataset(model, batch, width, opts):
    with torch.no_grad():
        if opts.decode_strategy in ('sample', 'greedy'):
            if opts.decode_strategy == 'greedy':
                assert width == 0, "Do not set width when using greedy"
                assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                    "eval_batch_size should be smaller than calc batch size"
                batch_rep = 1
                iter_rep = 1
            elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                assert opts.eval_batch_size == 1
                assert width % opts.max_calc_batch_size == 0
                batch_rep = opts.max_calc_batch_size
                iter_rep = width // opts.max_calc_batch_size
            else:
                batch_rep = width
                iter_rep = 1
            assert batch_rep > 0
            # This returns (batch_size, iter_rep shape)
            sequences, costs, ds = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            batch_size = len(costs)
            ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
        else:
            assert opts.decode_strategy == 'bs'

            cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                batch, beam_size=width,
                compress_mask=opts.compress_mask,
                max_calc_batch_size=opts.max_calc_batch_size
            )

    if sequences is None:
        sequences = [None] * batch_size
        costs = [math.inf] * batch_size
    else:
        sequences, costs = get_best(
            sequences.cpu().numpy(), costs.cpu().numpy(),
            ids.cpu().numpy() if ids is not None else None,
            batch_size
        )

    return [sequences, costs, ds]


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def appnd(listt, model, outs, duration):
    for seq, cost, d in zip(outs[0], outs[1], outs[2]):
        if model.problem.NAME == "tsp":
            seq = seq.tolist()  # No need to trim as all are same length
        elif model.problem.NAME in ("cvrp", "sdvrp"):
            seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            # d = d.tolist()
        elif model.problem.NAME in ("op", "pctsp"):
            seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
        else:
            assert False, "Unkown problem: {}".format(model.problem.NAME)
        # Note VRP only
        listt.append((cost, seq, d, duration))

        return listt
