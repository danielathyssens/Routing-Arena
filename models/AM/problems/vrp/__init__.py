
--boundary_.oOo._ZoGrXgAJYnckvrkSkIzp8TYFHppDQJKM
Content-Length: 17330
Content-Type: application/octet-stream
X-File-MD5: 47ca56169335459da26abc3a17c8433c
X-File-Mtime: 1648144214
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/tsp/tsp_baseline.py

import argparse
import numpy as np
import os
import time
from datetime import timedelta
from scipy.spatial import distance_matrix
from ...utils import run_all_in_pool
from ...utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output, CalledProcessError
from ...problems.vrp.vrp_baseline import get_lkh_executable
import torch
from tqdm import tqdm
import re


def solve_gurobi(directory, name, loc, disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have gurobi installed to run this script
    from ...problems.tsp.tsp_gurobi import solve_euclidian_tsp as solve_euclidian_tsp_gurobi

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()

            cost, tour = solve_euclidian_tsp_gurobi(loc, threads=1, timeout=timeout, gap=gap)
            duration = time.time() - start  # Measure clock time
            save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        total_cost = calc_tsp_length(loc, tour)
        assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def solve_concorde_log(executable, directory, name, loc, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.tsp".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.concorde.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    # if True:
    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                try:
                    # Concorde is weird, will leave traces of solution in current directory so call from target dir
                    check_call([executable, '-s', '1234', '-x', '-o',
                                os.path.abspath(tour_filename), os.path.abspath(problem_filename)],
                               stdout=f, stderr=f, cwd=directory)
                except CalledProcessError as e:
                    # Somehow Concorde returns 255
                    assert e.returncode == 255
                duration = time.time() - start

            tour = read_concorde_tour(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def solve_lkh_log(executable, directory, name, loc, runs=1, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.lkh{}.cvrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsplib(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_concorde_tour(filename):
    with open(filename, 'r') as f:
        n = None
        tour = []
        for line in f:
            if n is None:
                n = int(line)
            else:
                tour.extend([int(node) for node in line.rstrip().split(" ")])
    assert len(tour) == n, "Unexpected tour length"
    return tour


def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour.tolist()


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
    )


def run_insertion(loc, method):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    tour = []  # np.empty((0, ), dtype=int)
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is r