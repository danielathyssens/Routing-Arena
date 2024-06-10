
--boundary_.oOo._EDo3xtuRCSqh5n9ApTcUkIhSvBcwe8P9
Content-Length: 230
Content-Type: application/octet-stream
X-File-MD5: bf3b213e04458d74cd073d8852c3ca00
X-File-Mtime: 1634914512
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/__init__.py

from models.AM.problems.tsp.problem_tsp import TSP
from models.AM.problems.vrp.problem_vrp import CVRP, SDVRP
from models.AM.problems.op.problem_op import OP
from models.AM.problems.pctsp.problem_pctsp import PCTSPDet, PCTSPStoch

--boundary_.oOo._EDo3xtuRCSqh5n9ApTcUkIhSvBcwe8P9
Content-Length: 8
Content-Type: application/octet-stream
X-File-MD5: 1bf32eada91639e81d4e828bbf2beb03
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/.gitignore

compass/
--boundary_.oOo._EDo3xtuRCSqh5n9ApTcUkIhSvBcwe8P9
Content-Length: 346
Content-Type: application/octet-stream
X-File-MD5: b2b3fa0236e1ee42a09bfd548107d289
X-File-Mtime: 1645797821
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/install_compass.sh

#!/usr/bin/problem bash
git clone https://github.com/bcamath-ds/compass
cd compass
sudo apt-get install libtool m4
sudo apt-get install libgsl0-dev libatlas-base-dev libbfd-dev libiberty-dev
sudo apt-get install libssl-dev
sudo apt-get install autoconf automake
autoheader
libtoolize
aclocal
automake --add-missing
autoconf
./configure
make
cd ..
--boundary_.oOo._EDo3xtuRCSqh5n9ApTcUkIhSvBcwe8P9
Content-Length: 4369
Content-Type: application/octet-stream
X-File-MD5: ecd98705f1182d52ed12b059409d466e
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/op_gurobi.py

#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

from gurobipy import *


def solve_euclidian_op(depot, loc, prize, max_length, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan op problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    points = [depot] + loc
    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if tour is not None:
                # add subtour elimination constraint for every pair of cities in tour
                # model.cbLazy(quicksum(model._vars[i, j]
                #                       for i, j in itertools.combinations(tour, 2))
                #              <= len(tour) - 1)

                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour)))

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges, exclude_depot=True):
        unvisited = list(range(n))
        #cycle = range(n + 1)  # initial length has 1 more city
        cycle = None
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            # If we do not yet have a cycle or this is the shorter cycle, keep this cycle
            # Unless it contains the depot while we do not want the depot
            if (
                (cycle is None or len(cycle) > len(thiscycle))
                    and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
            ):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # Depot vars can be 2
    for i,j in vars.keys():
        if i == 0 or j == 0:
            vars[i,j].vtype = GRB.INTEGER
            vars[i,j].ub = 2

    prize_dict = {
        i + 1: -p  # We need to maximize so negate
        for i, p in enumerate(prize)
    }
    delta = m.addVars(range(1, n), obj=prize_dict, vtype=GRB.BINARY, name='delta')

    # Add degree-2 constraint (2 * delta for nodes which are not the depot)
    m.addConstrs(vars.sum(i,'*') == (2 if i == 0 else 2 * delta[i]) for i in range(n))

    # Length of tour constraint
    m.addConstr(quicksum(var * dist[i, j] for (i, j), var in vars.items() if j < i) <= max_length)

    # Optimize model

    m._vars = vars
    m._dvars = delta
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected, exclude_depot=False)
    assert tour[0] == 0, "Tour should start with depot"

    return m.objVal, tour
--boundary_.oOo._EDo3xtuRCSqh5n9ApTcUkIhSvBcwe8P9
Content-Length: 16915
Content-Type: application/octet-stream
X-File-MD5: a4a61397ca3b9241dc1d810f9fab9932
X-File-Mtime: 1642773340
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/op_baseline.py

import argparse
import os
import numpy as np
from ...utils import run_all_in_pool
from ...utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output
import tempfile
import time
from datetime import timedelta
from ...problems.op.opga.opevo import run_alg as run_opga_alg
from tqdm import tqdm
import re

MAX_LENGTH_TOL = 1e-5


# Run install_compass.sh to install
def solve_compass(executable, depot, loc, demand, capacity):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.oplib")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_oplib(problem_filename, depot, loc, demand, capacity)
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_compass_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_oplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_compass_log(executable, directory, name, depot, loc, prize, max_length, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.oplib".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.compass.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_oplib(problem_filename, depot, loc, prize, max_length, name=name)

            with open(log_filename, 'w') as f:
         