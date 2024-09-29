import torch
import math
import os
import sys

# Functions for Uchoa Data Generation (reference: DPDP (Kool et al. (2021)):
GRID_SIZE = 1000


def create_BKS(data_file_path: str, model_name: str):
    X_res = []
    BKS_registry = {}
    for file in os.listdir(data_file_path):
        # print('file: ', file)
        if file[-3:] == 'sol':
            cost, routes = read_X_sol_file(data_file_path + "/" + file)
            X_res.append((file[:-4], cost, routes))
            BKS_registry[file[:-4]] = (cost, routes, model_name)
    return BKS_registry

# DATA PREPROCESSING/IMPORT UTILS
def read_X_sol_file(sol_filename):
    num_vehicles = 0
    routes = []
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            l = line.strip().split(" ")
            # print(l)
            if 'Route' in l:
                num_vehicles += 1
                routes.append([int(loc) for loc in l[2:]])
            elif 'Cost' in l:
                cost = int(l[-1])

    return cost, routes



# DATA GENERATION UTILS
def generate_depot_coordinates(batch_size, depot_type=None, device=None):
    # Depot Position
    # 0 = central (500, 500), 1 = eccentric (0, 0), 2 = random
    depot_types = (torch.rand(batch_size, device=device) * 3).int()
    if depot_type is not None:  # mix
        # Central, Eccentric, Random
        codes = {'C': 0, 'E': 1, 'R': 2}
        depot_types[:] = codes[depot_type.upper()]

    depot_locations = torch.rand(batch_size, 2, device=device) * GRID_SIZE
    depot_locations[depot_types == 0] = GRID_SIZE / 2
    depot_locations[depot_types == 1] = 0
    return depot_locations, depot_types


def generate_clustered(num_seeds, num_samples, max_seeds=None, device=None):
    if max_seeds is None:
        max_seeds = num_seeds.max().item()
    batch_size = num_seeds.size(0)
    batch_rng = torch.arange(batch_size, dtype=torch.long, device=device)
    seed_coords = (torch.rand(batch_size, max_seeds, 2, device=device) * GRID_SIZE)
    # We make a little extra since some may fall off the grid
    n_try = num_samples * 2
    while True:
        loc_seed_ind = (torch.rand(batch_size, n_try) * num_seeds[:, None].float()).long()
        loc_seeds = seed_coords[batch_rng[:, None], loc_seed_ind]
        alpha = torch.rand(batch_size, n_try, device=device) * 2 * math.pi
        d = -40 * torch.rand(batch_size, n_try, device=device).log()
        coords = torch.stack((torch.sin(alpha), torch.cos(alpha)), -1) * d[:, :, None] + loc_seeds
        coords.size()
        feas = ((coords >= 0) & (coords <= GRID_SIZE)).sum(-1) == 2
        feas_topk, ind_topk = feas.byte().topk(num_samples, dim=-1)
        if feas_topk.all():
            break
        n_try *= 2  # Increase if this fails
    return coords[batch_rng[:, None], ind_topk]


def generate_customer_coordinates(batch_size, graph_size, min_seeds=3, max_seeds=8, customer_type=None, device=None):
    # Customer position
    # 0 = random, 1 = clustered, 2 = random clustered 50/50
    # We always do this so we always pull the same number of random numbers
    customer_types = (torch.rand(batch_size, device=device) * 3).int()
    if customer_type is not None:  # Mix
        # Random, Clustered, Random-Clustered (half half)
        codes = {'R': 0, 'C': 1, 'RC': 2}
        customer_types[:] = codes[customer_type.upper()]

    # Sample number of seeds uniform (inclusive)
    num_seeds = (torch.rand(batch_size, device=device) * ((max_seeds - min_seeds) + 1)).int() + min_seeds

    # We sample random and clustered coordinates for all instances, this way, the instances in the 'mix' case
    # Will be exactly the same as the instances in one of the tree 'not mixed' cases and we can reuse evaluations
    rand_coords = torch.rand(batch_size, graph_size, 2, device=device) * GRID_SIZE
    clustered_coords = generate_clustered(num_seeds, graph_size, max_seeds=max_seeds, device=device)

    # Clustered
    rand_coords[customer_types == 1] = clustered_coords[customer_types == 1]
    # Half clustered
    rand_coords[customer_types == 2, :(graph_size // 2)] = clustered_coords[customer_types == 2, :(graph_size // 2)]

    return rand_coords, customer_types


def generate_demands(coords, device=None):
    batch_size, graph_size, _ = coords.size()
    # Demand distribution
    # 0 = unitary (1)
    # 1 = small values, large variance (1-10)
    # 2 = small values, small variance (5-10)
    # 3 = large values, large variance (1-100)
    # 4 = large values, large variance (50-100)
    # 5 = depending on quadrant top left and bottom right (even quadrants) (1-50), others (51-100) so add 50
    # 6 = many small, few large most (70 to 95 %, unclear so take uniform) from (1-10), rest from (50-100)
    lb = torch.tensor([1, 1, 5, 1, 50, 1, 1], dtype=torch.long, device=device)
    ub = torch.tensor([1, 10, 10, 100, 100, 50, 10], dtype=torch.long, device=device)
    customer_positions = (torch.rand(batch_size, device=device) * 7).long()
    lb_ = lb[customer_positions, None]
    ub_ = ub[customer_positions, None]
    # Make sure we always sample the same number of random numbers
    rand_1 = torch.rand(batch_size, graph_size, device=device)
    rand_2 = torch.rand(batch_size, graph_size, device=device)
    rand_3 = torch.rand(batch_size, device=device)
    demands = (rand_1 * (ub_ - lb_ + 1).float()).long() + lb_
    # either both smaller than grid_size // 2 results in 2 inequalities satisfied, or both larger 0
    # in all cases it is 1 (odd quadrant) and we should add 50

    demands[customer_positions == 5] += ((coords[customer_positions == 5] < GRID_SIZE // 2).long().sum(
        -1) == 1).long() * 50
    # slightly different than in the paper we do not exactly pick a value between 70 and 95 % to have a large value
    # but based on the threshold we let each individual location have a large demand with this probability
    demands_small = demands[customer_positions == 6]
    demands[customer_positions == 6] = torch.where(
        rand_2[customer_positions == 6] > (rand_3 * 0.25 + 0.70)[customer_positions == 6, None],
        demands_small,
        (rand_1[customer_positions == 6] * (100 - 50 + 1)).long() + 50
    )
    return demands

# changed from Kool et al. (random state)
def sample_triangular(sz, a, b, c, rnd_state=None, device=None):
    # See https://en.wikipedia.org/wiki/Triangular_distribution#Generating_triangular-distributed_random_variates
    # print('device', device)
    a, b, c = (torch.tensor(v, dtype=torch.float, device=device) for v in (a, b, c))
    # U = torch.rand(sz, device=device)
    U = torch.from_numpy(rnd_state.random(sz)).to(device)
    Fc = (c - a) / (b - a)
    return torch.where(
        U < Fc,
        a + torch.sqrt(U * (b - a) * (c - a)),
        b - torch.sqrt((1 - U) * (b - a) * (b - c))
    )


# from L2O-Meta
def format_ds_save_path(directory, args=None, affix=None, fname=''):
    """Format the path for saving datasets"""
    directory = os.path.normpath(os.path.expanduser(directory))

    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    if affix is not None:
        fname = str(affix) + fname
    if fname != '':
        fpath = os.path.join(directory, fname)
    else:
        fpath = directory
    if fpath[-3:] not in ['.pt', 'dat', 'pkl']:
        fpath += '.pt'

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        a = input()
        if a != 'y':
            print('Could not write to file. Terminating program...')
            sys.exit()

    return fpath

