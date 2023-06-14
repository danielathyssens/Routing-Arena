import numpy as np
import torch
from formats import CVRPInstance, TSPInstance, RPSolution
from models.NeuroLS.lib.routing.formats import RPInstance
from models.NeuroLS.lib.routing.formats import RPSolution as RPSol_neurols
from typing import Optional, Dict, Union, List
from operator import attrgetter
import logging

logger = logging.getLogger(__name__)

CVRP_DEFAULTS = {'uniform': {  # max num vehicles and integer capacity per problem size
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}, 'uchoa':
    {  # max num vehicles and integer capacity per problem size (UCHOA)
        20: [8, 30],
        50: [16, 40],
        100: [80, 50],
        200: [130, 50],
        500: [210, 50],
    }, None:
    {  # max num vehicles and integer capacity per problem size
        20: [8, 30],
        50: [16, 40],
        100: [32, 50],
        200: [48, 50],
        500: [64, 50],
    },
}


def to_RPInstance(instance_list: Union[List[TSPInstance], List[CVRPInstance]]) -> List[RPInstance]:
    # logger.info(f"transforming data to RPInstance")
    # print('instance_list[0].coords[:5]', instance_list[0].coords[:5])
    # print('instance_list[0].node_features[:5,-1]', instance_list[0].node_features[:5,-1])
    # print('instance.vehicle_capacity', instance_list[0].vehicle_capacity)
    if isinstance(instance_list[0], CVRPInstance):
        return [
            RPInstance(
                coords=instance.coords.astype('float32'),
                node_features=instance.node_features.astype('float32'),
                graph_size=instance.graph_size,
                depot_idx=[0],
                constraint_idx=[-1],  # demand is at last position of node features,
                vehicle_capacity=instance.vehicle_capacity if isinstance(instance, CVRPInstance) else None,
                max_num_vehicles=instance.max_num_vehicles if instance.max_num_vehicles is not None
                else instance.graph_size - 1  # else
                # CVRP_DEFAULTS[instance.type][instance.graph_size - 1][0] if isinstance(instance, CVRPInstance) else None
            )
            for i, instance in enumerate(instance_list)
        ]

    elif isinstance(instance_list[0], TSPInstance):
        return [
            RPInstance(
                coords=instance.coords,
                node_features=instance.node_features,
                graph_size=instance.graph_size,
                depot_idx=[0],
                constraint_idx=[-1],  # demand is at last position of node features,
            )
            for i, instance in enumerate(instance_list)
        ]
    else:
        raise NotImplementedError(f"Instance type needs to be CVRPInstance or TSPInstance.")


def to_RPSolution(solutions: List[RPSol_neurols],
                  running_solutions: Optional[List[List[List]]],
                  running_times: Optional[List[List]],
                  original_instances: List[CVRPInstance]) -> List[RPSolution]:
    """Parse model solutions (RPSol_neurols) back to RPSolution for consistent evaluation.
       The solution, instance, etc. for the first test instance (id=0) is always appended last in NeuroLS"""
    # correct instance ids in solutions
    sols_correct = []
    count=0
    instance_ids_in_order = [instance.instance_id for instance in original_instances]
    inst_dct = {inst_id: instance
                for inst_id, instance in zip(instance_ids_in_order, original_instances)
                }
    for sol in solutions:
        true_id = [instance.instance_id for instance in original_instances if np.array_equal(sol.instance.coords.astype('float32'),
                                                                                             instance.coords.astype('float32'))][0]
        print('true_id', true_id)
        print('sol.instance.instance_id', sol.instance.instance_id)
        print('instance_ids_in_order[count]', instance_ids_in_order[count])
        print('inst_dct[true_id].instance_id', inst_dct[true_id].instance_id)
        sols_correct.append(sol.update(instance=inst_dct[true_id]))
        count=0
    # print('sols_correct', sols_correct)
    assert sorted([sol.instance.instance_id for sol in solutions]) == list(np.arange(len(original_instances))), \
        f"Instance IDs problem!"
    RP_sols_unsorted = [
        RPSolution(
            solution=sol.solution if sol is not None else None,
            running_sols=running_solutions[i] if running_solutions[i] is not None else None,
            cost=sol.cost if sol is not None else None,
            num_vehicles=sol.num_vehicles if sol is not None else None,
            run_time=sol.run_time if sol is not None else None,
            running_times=running_times[i] if running_times[i] is not None else None,
            problem=sol.problem,
            instance=sol.instance,
            instance_id=sol.instance.instance_id  # needed here to sort RPSolutions before eval
        )
        for i, sol in enumerate(sols_correct)
    ]
    # print('RP_sols_unsorted', RP_sols_unsorted)
    RP_sols = sorted(RP_sols_unsorted, key=attrgetter('instance_id'))
    # print('RP_sols', RP_sols)
    return RP_sols
