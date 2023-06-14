#!/bin/env python

# Mostly from https://github.com/yorak/VeRyPy

from __future__ import print_function
from __future__ import division
from builtins import range
from timeit import default_timer
from typing import List, Union

from tqdm import tqdm
import numpy as np
from scipy.spatial import distance_matrix as calc_distance_matrix
from timeit import default_timer as timer
from itertools import groupby
from formats import TSPInstance, CVRPInstance, RPSolution
from verypy.classic_heuristics.parallel_savings import parallel_savings_init, clarke_wright_savings_function
from verypy.classic_heuristics.gaskell_savings import gaskell_lambda_savings_function, gaskell_pi_savings_function
from verypy.util import sol2routes

C_EPS = 1e-10
S_EPS = 1e-10


def sol2routes(sol):
    """Convert  solution to a list of routes (each a list of customers leaving 
    and returning to a depot (node 0). Removes empty routes. WARNING: this also 
    removes other concecutive duplicate nodes, not just 0,0!"""
    if not sol or len(sol) <= 2: return []
    return [[0] + list(r) + [0] for x, r in groupby(sol, lambda z: z == 0) if not x]


def routes2sol(routes):
    """Concatenates a list of routes to a solution. Routes may or may not have
    visits to the depot (node 0), but the procedure will make sure that 
    the solution leaves from the depot, returns to the depot, and that the 
    routes are separated by a visit to the depot."""
    if not routes:
        return None

    sol = [0]
    for r in routes:
        if r:
            if r[0] == 0:
                sol += r[1:]
            else:
                sol += r
            if sol[-1] != 0:
                sol += [0]
    return sol


def clarke_wright_savings_function(D):
    N = len(D)
    n = N - 1
    savings = [None] * int((n * n - n) / 2)
    idx = 0
    for i in range(1, N):
        for j in range(i + 1, N):
            s = D[i, 0] + D[0, j] - D[i, j]
            savings[idx] = (s, -D[i, j], i, j)
            idx += 1
    savings.sort(reverse=True)
    return savings


def gaskell_lambda_savings_function(D):
    n = len(D) - 1
    savings = [None] * int((n * n - n) / 2)
    idx = 0
    d_avg = np.average(D[0:])
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s_AB = D[i, 0] + D[0, j] - D[i, j]
            lambda_AB = s_AB * (d_avg + abs(D[0, i] - D[0, j]) - D[i, j])
            savings[idx] = (lambda_AB, -D[i, j], i, j)
            idx += 1
    savings.sort(reverse=True)

    return savings


def gaskell_pi_savings_function(D):
    n = len(D) - 1
    savings = [None] * int((n * n - n) / 2)
    idx = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            pi_AB = D[i, 0] + D[0, j] - 2 * D[i, j]
            savings[idx] = (pi_AB, -D[i, j], i, j)
            idx += 1
    savings.sort(reverse=True)
    return savings


def parallel_savings_init(D, d, C, L=None, minimize_K=False,
                          savings_callback=clarke_wright_savings_function):
    """
    Implementation of the basic savings algorithm / construction heuristic for
    capaciated vehicle routing problems with symmetric distances (see, e.g.
    Clarke-Wright (1964)). This is the parallel route version, aka. best
    feasible merge version, that builds all of the routes in the solution in
    parallel making always the best possible merge (according to varied savings
    criteria, see below).
    
    * D is a numpy ndarray (or equvalent) of the full 2D distance matrix.
    * d is a list of demands. d[0] should be 0.0 as it is the depot.
    * C is the capacity constraint limit for the identical vehicles.
    * L is the optional constraint for the maximum route length/duration/cost.
    
    * minimize_K sets the primary optimization objective. If set to True, it is
       the minimum number of routes. If set to False (default) the algorithm 
       optimizes for the mimimum solution/routing cost. In savings algorithms 
       this is done by ignoring a merge that would increase the total distance.
       WARNING: This only works when the solution from the savings algorithm is
       final. With postoptimimization this non-improving move might have still
       led to improved solution.
   
    * optional savings_callback is a function of the signature:
        sorted([(s_11,x_11,i_1,j_1)...(s_ij,x_ij,i,j)...(s_nn,x_nn,n,n) ]) =
            savings_callback(D)
      where the returned (sorted!) list contains savings (that is, how much 
       solution cost approximately improves if a route merge with an edge
       (i,j) is made). This should be calculated for each i \in {1..n},
       j \in {i+1..n}, where n is the number of customers. The x is a secondary
       sorting criterion but otherwise ignored by the savings heuristic.
      The default is to use the Clarke Wright savings criterion.
        
    See clarke_wright_savings.py, gaskell_savings.py, yellow_savings.py etc.
    to find specific savings variants. They all use this implementation to do 
    the basic savings procedure and they differ only by the savings
    calculation. There is also the sequental_savings.py, which builds the 
    routes one by one.
    
    Clarke, G. and Wright, J. (1964). Scheduling of vehicles from a central
     depot to a number of delivery points. Operations Research, 12, 568-81.
    """
    N = len(D)
    ignore_negative_savings = not minimize_K

    ## 1. make route for each customer
    routes = [[i] for i in range(1, N)]
    route_demands = d[1:] if C else [0] * N
    if L: route_costs = [D[0, i] + D[i, 0] for i in range(1, N)]

    ## 2. compute initial savings 
    savings = savings_callback(D)

    # zero based node indexing!
    endnode_to_route = [0] + list(range(0, N - 1))

    ## 3. merge
    # Get potential merges best savings first (second element is secondary
    #  sorting criterion, and it it ignored)
    for best_saving, _, i, j in savings:

        if ignore_negative_savings:
            cw_saving = D[i, 0] + D[0, j] - D[i, j]
            if cw_saving < 0.0:
                break

        left_route = endnode_to_route[i]
        right_route = endnode_to_route[j]

        # the node is already an internal part of a longer segment
        if ((left_route is None) or
                (right_route is None) or
                (left_route == right_route)):
            continue

        # check capacity constraint validity
        if C:
            merged_demand = route_demands[left_route] + route_demands[right_route]
            if merged_demand - C_EPS > C:
                continue
        # if there are route cost constraint, check its validity        
        if L:
            merged_cost = route_costs[left_route] - D[0, i] + \
                          route_costs[right_route] - D[0, j] + \
                          D[i, j]
            if merged_cost - S_EPS > L:
                continue

        # update bookkeeping only on the recieving (left) route
        if C: route_demands[left_route] = merged_demand
        if L: route_costs[left_route] = merged_cost

        # merging is done based on the joined endpoints, reverse the 
        #  merged routes as necessary
        if routes[left_route][0] == i:
            routes[left_route].reverse()
        if routes[right_route][-1] == j:
            routes[right_route].reverse()

        # the nodes that become midroute points cannot be merged
        if len(routes[left_route]) > 1:
            endnode_to_route[routes[left_route][-1]] = None
        if len(routes[right_route]) > 1:
            endnode_to_route[routes[right_route][0]] = None

        # all future references to right_route are to merged route
        endnode_to_route[routes[right_route][-1]] = left_route

        # merge with list concatenation
        routes[left_route].extend(routes[right_route])
        routes[right_route] = None
    return routes


def eval_savings(data, savings_function, is_normalized):
    options = {
        'clarke_wright': clarke_wright_savings_function,
        'gaskell_lambda': gaskell_lambda_savings_function,
        'gaskell_pi': gaskell_pi_savings_function
    }
    assert savings_function in options.keys(), print('not a valid savings function')
    savings_func = options[savings_function]

    solutions = []
    times = []
    objs = []
    for i, instance in enumerate(data):
        # the implementation assumes one depot and at index 0. The CVRP instance has a depot index variable as a list.
        # We simply retrieve the first element of that list and treat it as the depot. In case this  index is not 0, we swap the nodes.
        # This will break if the instance has multiple depots
        assert len(instance['depot_idx']) == 1, print('This savings implementation does not deal with multiple depots')
        depot_index = instance['depot_idx'][0]
        instance['node_features'][[0, depot_index]] = instance['node_features'][[depot_index, 0]]
        # instance['node_features'][:, 2:4]
        distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
        demands = instance['node_features'][:, 4].tolist()
        vehicle_capacity = instance.vehicle_capacity if is_normalized else instance.original_capacity

        t1 = timer()
        solution = parallel_savings_init(D=distances, d=demands, C=vehicle_capacity, savings_callback=savings_func)
        solution = sol2routes(routes2sol(solution))
        t2 = timer()
        total_time = t2 - t1

        sol = RPSolution(
            solution=solution,
            cost=cost(solution, distances),
            run_time=total_time,
            problem="CVRP",
            instance=instance
        )

        solutions.append(sol)
        times.append(total_time)
        objs.append(cost(solution, distances))

    runtimes = np.array(times)
    objs = np.array(objs)

    results_ = {
        "objs": objs,
        "runtimes": runtimes,
    }
    return results_, solutions


# version veripy API
def eval_savings_(
        instance: CVRPInstance,
        min_k: bool = False,
        savings_function: str = 'clarke_wright',
        **kwargs
):
    SAVINGS_FN = {
        'clarke_wright': clarke_wright_savings_function,
        'gaskell_lambda': gaskell_lambda_savings_function,
        'gaskell_pi': gaskell_pi_savings_function
    }
    savings_func = SAVINGS_FN[savings_function]

    demands = instance['node_features'][:, 4].tolist()  # instance.demands.copy()
    vehicle_capacity = instance['vehicle_capacity']

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = parallel_savings_init(
        D=distances, d=demands, C=vehicle_capacity,
        savings_callback=savings_func,
        minimize_K=min_k,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total, distances


def run_savings(test_ds: List[CVRPInstance],
                savings_func: str = 'clarke_wright',
                min_k: bool = False,
                disable_progress_bar: bool = False):
    # sols = []
    solutions, times, objs = [], [], []
    for instance in tqdm(test_ds, disable=disable_progress_bar):
        assign, rt, dist = eval_savings_(
            instance=instance,
            min_k=min_k,
            savings_function=savings_func
        )
        # sols.append({
        #     "instance": instance,
        #     "assignment": assign,
        #     "run_time": rt,
        # })
        solutions.append(assign)
        times.append(rt)
        objs.append(cost(assign, dist))

    return [
        RPSolution(
            cost=obj,
            solution=sol,
            run_time=time,
            instance=inst,
            problem="CVRP"
        )
        for obj, sol, time, inst in zip(objs, solutions, times, test_ds)
    ]


def cost(routes, D):
    """calculate the cost of a solution"""
    cost = 0
    for route in routes:
        cost += sum([D[route[i], route[i + 1]] for i in range(len(route) - 1)])
    return cost
