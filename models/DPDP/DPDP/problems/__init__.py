from models.DPDP.DPDP.problems.tsp.problem_tsp import TSP # models.DPDP.DPDP.
from models.DPDP.DPDP.problems.vrp.problem_vrp import CVRP
from models.DPDP.DPDP.problems.tsptw.problem_tsptw import TSPTW

def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'tsptw': TSPTW
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem