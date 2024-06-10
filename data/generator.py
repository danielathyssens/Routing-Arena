import os
import random
import warnings
import logging
import torch
import numpy as np
from warnings import warn
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any
from omegaconf import DictConfig

from data.sampler import DataSampler
from formats import RPInstance, TSPInstance, CVRPInstance, CVRPTWInstance

# functions for uchoa instance generation
from data.data_utils import (
    generate_depot_coordinates,
    generate_customer_coordinates,
    generate_demands,
    sample_triangular
)

logger = logging.getLogger(__name__)

CVRP_DEFAULTS = {  # num vehicles and integer capacity per problem size
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}

# vehicle capacities for instances with TW (from Solomon)
TW_CAPACITIES = {
    10: 250.,
    20: 500.,
    50: 750.,
    100: 1000.
}
# standard maximum fleet size
# STD_K = {
#     10: 6,
#     20: 12,
#     50: 24,
#     100: 36,
# }


INSTANCE_TYPE = {
    "tsp": TSPInstance,
    "cvrp": CVRPInstance,
    "cvrptw": CVRPTWInstance,
}

UCHOA_TYPE = {
    "depot": {0: "C", 1: "E", 2: "R"},
    "customer": {0: "R", 1: "C", 2: "RC"},
}

BASE_SIZES = {
    20: 2000,
    50: 5000,
    100: 10000
}


# code from L2O-Meta
class RPGenerator:
    """Wraps data generation, loading and saving functionalities for routing problems."""
    RPS = ['tsp', 'cvrp', 'cvrptw']

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 float_prec: np.dtype = np.float32,
                 generator_args: Union[dict, DictConfig] = None):
        self._seed = seed if seed is not None else 1234
        print('self._seed', self._seed)
        self.generator_args = generator_args
        self.store_subsampled_data = True  # self.generator_args["store_subsamples"]
        self._rnd = np.random.default_rng(seed)  # np.random.RandomState(seed)  #
        print('self._rnd', self._rnd)
        self.verbose = verbose
        self.float_prec = float_prec
        print('self.generator_args', self.generator_args)
        self.sampler = DataSampler(verbose=verbose, random_state=self._rnd, **generator_args)
        try:
            self.single_large_inst = torch.load(generator_args.single_large_instance)
        except AttributeError:
            self.single_large_inst = generator_args.single_large_instance

    def generate_custom_data(self, problem_lst: List[str],
                             sampling_args_lst: List[dict],
                             usage_type: str,
                             for_RL4CO: bool = True,
                             save_path: str = "data/custom_data/"):

        for prob in problem_lst:
            for sample_args in sampling_args_lst:
                size = sample_args["sample_size"]
                g_size = sample_args["graph_size"]
                save_name = "/" + prob + str(g_size) + "_" + str(size) + "_" + str(
                    self._seed) + "_" + usage_type + ".npz"
                x = self.generate_subsamples(prob, base_nodes_size=BASE_SIZES[g_size], graph_size=g_size,
                                             distribution=self.generator_args["coords_sampling_dist"],
                                             sampling_args=sample_args, for_RL4CO=for_RL4CO)
                print("type(x)", type(x))
                # +str(self.generator_args['coords_sampling_dist'])
                np.savez(save_path + save_name, x)

    def generate_subsamples(self,
                            problem: str,
                            graph_size: int,
                            sample_size: int = None,
                            base_nodes_size: int = 10000,
                            distribution: str = 'uniform',
                            normalize: bool = True,
                            sampling_args: Dict = None,
                            for_RL4CO: bool = False):
        """Generate 1 large (min 100000 nodes) RP instance from which to
           create subsampled instances - while ensuring feasibility"""

        sample_size = sampling_args["sample_size"] if sample_size is None else sample_size
        graph_size = sampling_args["graph_size"] if graph_size is None else graph_size
        cap_original = sampling_args['cap']
        # print('cap_original', cap_original)

        if self.single_large_inst is None:
            print("generating single large instance to subsample from...")
            if problem in ["cvrp", "vrp", "cvrptw"]:
                sampling_args['cap'] = sampling_args['cap']  # int(base_nodes_size/graph_size)*sampling_args['cap']
                # logger.info(f"temporarily adjust capacity for {problem} to {sampling_args['cap']}...")
            self.single_large_inst = self.generate(problem,
                                                   sample_size=1,
                                                   graph_size=base_nodes_size,
                                                   normalize=normalize,
                                                   sampling_args=sampling_args,
                                                   feasibility_insurance=False)
            logger.info(f"saving single large instance in logs ...")
            if isinstance(self.single_large_inst, tuple):
                torch.save(self.single_large_inst[0], "single_large_instance.pt")
                self.single_large_inst = self.single_large_inst[0]
            else:
                torch.save(self.single_large_inst, "single_large_instance.pt")
            if problem in ["cvrp", "vrp", "cvrptw", "vrptw"]:
                # if self.generator_args.coords_sampling_dist != "uchoa":
                # reset capacity
                sampling_args['cap'] = int(self.single_large_inst[0].original_capacity)

            print(f"original cap: sampling_args['cap']: {sampling_args['cap']}")
        # sub-sample #sample_size instances from single large instance
        if not for_RL4CO:
            logger.info(f"subsampling from single large instance ...")
            sub_sampled_data = self._sub_sample(problem,
                                                self.single_large_inst[0],
                                                self.sampler.normalize_demands,
                                                sample_size,
                                                graph_size,
                                                sampling_args['n_depots'],
                                                distribution)
            if self.store_subsampled_data:
                logger.info(f"saving sub-sampled instances in logs ...")
                if self.generator_args.coords_sampling_dist == "uchoa":
                    file_name = (f"{problem}{graph_size}_{self.sampler.depot_type}_{self.sampler.customer_type}"
                                 f"_{self.sampler.customer_type}{self._seed}_size{sample_size}.pt")
                else:
                    file_name = (f"{problem}{graph_size}_{self._seed}_{self.generator_args.coords_sampling_dist}"
                                 f"_size{sample_size}.pt")
                torch.save(sub_sampled_data, file_name)
            return sub_sampled_data
        else:
            return self._out_RL4CO(problem, self._sub_sample(problem,
                                                             self.single_large_inst[0],
                                                             self.sampler.normalize_demands,
                                                             sampling_args["sample_size"],
                                                             graph_size,
                                                             sampling_args['n_depots'],
                                                             distribution))

    def generate(self,
                 problem: str,
                 sample_size: int = None,
                 graph_size: int = None,
                 normalize: bool = True,
                 sampling_args: Dict = None,
                 feasibility_insurance: bool = None):
        """Generate data with corresponding RP generator function."""

        if sample_size is None:
            sample_size = sampling_args["sample_size"]
        if graph_size is None:
            graph_size = sampling_args["graph_size"]

        print('sample_size', sample_size)
        print('graph_size', graph_size)
        print('sampling_args', sampling_args)
        print('feasibility_insurance', feasibility_insurance)

        #    try:
        #         generate = getattr(self, f"generate_{problem.lower()}_data")
        #    except AttributeError:
        #         raise ModuleNotFoundError(f"The corresponding generator for the problem <{problem}> does not exist.")
        #       return generate(size=sample_size, graph_size=graph_size, **kwargs)

        if problem.lower() == "tsp":
            return self.generate_tsp_data(sample_size, graph_size, normalize, sampling_args)

        elif problem.lower() == "cvrp":
            return self.generate_cvrp_data(sample_size, graph_size, normalize, sampling_args, feasibility_insurance)

        elif problem.lower() == "cvrptw":
            return self.generate_cvrptw_data(sample_size, graph_size, normalize, sampling_args)

        else:
            raise ModuleNotFoundError(f"The generator for problem type <{problem}> does not exist.")

        # generate uniformly distributed data from Nazari et al.
        # if distribution is None:
        #     warn(f"No general distribution type for data generation specified. Defaulting to uniform distribution.")
        #     return self.generate_nazari(problem, sample_size, graph_size, normalize, **sampling_args)

        # generate uniformly distributed data from Nazari et al.
        # elif distribution in ["nazari", "uniform"]:
        #     if sample_size is not None and graph_size is not None:
        #        return self.generate_nazari(problem, sample_size, graph_size, sampling_args["k"], sampling_args["cap"],
        #                                     sampling_args["max_cap_factor"], sampling_args["n_depots"])
        #     else:
        #         return self.generate_nazari(problem, size=sample_size, **sampling_args)

        # generate Uchoa-like distribution data
        # elif distribution == "uchoa":
        #     return self.generate_uchoa(problem=problem,
        #                                size=sample_size,
        #                                graph_size=graph_size,
        #                                normalize=normalize,
        #                                n_depots=sampling_args['n_depots'],
        #                                **generator_args)

        # elif distribution == "solomon":
        #     raise NotImplementedError

        # elif distribution in ["gm", "gm_unif_mixed"]:
        #     return self.generate_gm_unif(
        #         distribution=distribution,
        #         problem=problem,
        #         weight_dist=generator_args['weights_sampling_dist'],
        #         **sampling_args
        #     )

        # mix all existing distribution Samplers
        # elif distribution == "mixed":
        #     raise NotImplementedError

        # else:
        #     print("Specified Distribution not known - please enter one of the following distributions for sampling: ["
        #           "'uniform', 'uchoa', 'solomon']")

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        if self._seed is None or (seed is not None and self._seed != seed):
            self._seed = seed
            self._rnd = np.random.default_rng(seed)
            self.sampler.seed(seed)

    def generate_tsp_data(self,
                          sample_size: int = 1000,
                          graph_size: int = 100,
                          normalize: bool = True,
                          sampling_args: Dict = None):

        # coords = self.sampler.sample_tsp(n=graph_size)
        coords = np.stack([self.sampler.sample_tsp(n=graph_size) for _ in range(sample_size)])

        # use dummy depot node as start node in tsp tour, therefore need to reduce graph size by 1
        node_features = self._create_nodes(sample_size, graph_size - 1, n_depots=1, features=[coords])

        # type cast
        coords = coords.astype(self.float_prec)
        node_features = node_features.astype(self.float_prec)
        type_name = self.generator_args['coords_sampling_dist']

        return [
            TSPInstance(
                coords=coords[i],
                node_features=node_features[i],
                graph_size=graph_size,
                instance_id=i,
                type=type_name
            )
            for i in range(sample_size)
        ], None

    def generate_cvrp_data(self,
                           sample_size: int,
                           graph_size: int,
                           normalize: bool = True,
                           sampling_args: Dict = None,
                           feasibility_insurance: bool = None,
                           is_single_large_inst: bool = False):

        # sample_size, graph_size, k, cap, max_cap_factor, n_depots = sampling_args

        coords_, weights_, capa_, types_ = [], [], [], []
        print('sample_size', sample_size)
        print('feasibility_insurance', feasibility_insurance)
        if graph_size != sampling_args["graph_size"]:
            sampling_args["graph_size"] = graph_size
        if feasibility_insurance is not None:
            sampling_args["feasibility_insurance"] = feasibility_insurance
        for _ in range(sample_size):
            coords, c_probs, weights, original_capa, q_type, c_type, d_type = self.sampler.sample_cvrp(**sampling_args)
            coords_.append(coords)
            weights_.append(weights)
            capa_.append(original_capa)
            if self.generator_args.coords_sampling_dist == "uchoa":
                c_type_ = UCHOA_TYPE["customer"][c_type]
                d_type_ = UCHOA_TYPE["depot"][d_type]
            else:
                c_type_, d_type_ = c_type, d_type
            types_.append((c_type_, d_type, q_type))

        coords = np.stack(coords_)
        demands = np.stack(weights_)
        print('coords.shape', coords.shape)
        print('demands.shape', demands.shape)
        # print('demands[:,:5]', demands[:,:5])
        print('self.sampler.normalize_demands', self.sampler.normalize_demands)
        # demands = np.stack([
        #     self.sampler.sample_weights(n=graph_size + n_depots, k=k, cap=cap, max_cap_factor=max_cap_factor)
        #     for _ in range(sample_size)
        # ])
        node_features = self._create_nodes(sample_size, graph_size, n_depots=sampling_args["n_depots"],
                                           features=[coords, demands])

        # type cast
        coords = coords.astype(self.float_prec)
        node_features = node_features.astype(self.float_prec)
        if self.generator_args["coords_sampling_dist"] == "uchoa":
            if self.sampler.customer_type == self.sampler.demand_type == self.sampler.depot_type is not None:
                type_name = ("uchoa" + "_" + self.sampler.customer_type + "_" +
                             self.sampler.depot_type + "_" + str(self.sampler.demand_type))
            else:
                type_name = "uchoa_mixed"
        else:
            type_name = self.generator_args["coords_sampling_dist"] + "_" + self.generator_args["weights_sampling_dist"]
        return [
            CVRPInstance(
                coords=coords[i],
                node_features=node_features[i],
                graph_size=graph_size + sampling_args["n_depots"],
                constraint_idx=[-1],  # demand is at last position of node features
                vehicle_capacity=1.0,  # demands are normalized
                original_capacity=capa_[i],
                max_num_vehicles=int(sampling_args["k"]),
                depot_type=types_[i][1],  # depot stays uniform or uchoa-type depots
                coords_dist=self.sampler.coords_sampling_dist + "_" + str(types_[i][0]),
                demands_dist=types_[i][2],
                instance_id=i,
                type=type_name,
                sample_prob=c_probs if is_single_large_inst else None
            )
            for i in range(sample_size)
        ], self.sampler.normalize_demands

    def generate_cvrptw_data(self,
                             sample_size: int = 1000,
                             graph_size: int = 100,
                             normalize: bool = True,
                             sampling_args: Dict = None):

        coords_, weights_, t_windows_, service_time_, service_horizon_ = [], [], [], [], []
        for _ in range(sample_size):
            coords, weights, tw, service_time, service_horizon = self.sampler.sample_cvrptw(**sampling_args)
            print('coords.shape in fwrd loop - generate_cvrptw', coords.shape)
            print('weights.shape in fwrd loop - generate_cvrptw', weights.shape)
            print('tw.shape in fwrd loop - generate_cvrptw', tw.shape)
            coords_.append(np.squeeze(coords))
            weights_.append(np.squeeze(weights))
            t_windows_.append(np.squeeze(tw))
            service_time_.append(np.squeeze(service_time))
            service_horizon_.append(service_horizon)

        coords = np.stack(coords_)
        print('coords.shape in generate_cvrptw', coords.shape)
        demands = np.stack(weights_)
        print('demands.shape in generate_cvrptw', demands.shape)
        t_windows = np.stack(t_windows_)
        print('t_windows.shape in generate_cvrptw', t_windows.shape)
        service_time = np.stack(service_time_)
        print('service_time.shape in generate_cvrptw', service_time.shape)
        print('service_time in generate_cvrptw', service_time[0])
        # demands = np.stack([
        #     self.sampler.sample_weights(n=graph_size + n_depots, k=k, cap=cap, max_cap_factor=max_cap_factor)
        #     for _ in range(sample_size)
        # ])
        node_features = self._create_nodes(sample_size, graph_size, n_depots=sampling_args["n_depots"],
                                           features=[coords, demands, t_windows])

        type_name = self.generator_args["coords_sampling_dist"] + "_" + self.generator_args["weights_sampling_dist"]
        return [
            CVRPTWInstance(
                coords=coords[i],
                demands=demands[i],
                node_features=node_features[i],
                tw=t_windows[i],
                depot_tw=t_windows[i, 0],
                node_tw=t_windows[i, 1:],
                service_time=service_time[i],
                org_service_horizon=service_horizon_[i],
                graph_size=graph_size + sampling_args["n_depots"],
                constraint_idx=[-1],  # tw is at last position of node features
                vehicle_capacity=1.0,  # demands are normalized
                original_capacity=int(sampling_args["cap"]),
                max_num_vehicles=int(sampling_args["k"]),
                depot_type=self.sampler.depot_type,  # depot stays uniform or uchoa-type depots
                coords_dist=self.sampler.coords_sampling_dist,
                demands_dist=self.sampler.weights_sampling_dist,
                instance_id=i,
                type=type_name
            )
            for i in range(sample_size)
        ]

    def generate_gm_unif(self,
                         distribution,
                         problem,
                         sample_size,
                         graph_size,
                         k,
                         cap,
                         n_depots,
                         max_cap_factor,
                         weight_dist):
        # generator_args already passed to Sampler init (for uniform fraction, n_components, mu_sampling_dist, ...)

        if problem.lower() == 'tsp':
            coords = np.stack([
                self.sampler.sample_coords(n=graph_size + n_depots) for _ in range(sample_size)
            ])

            # use dummy depot node as start node in tsp tour, therefore need to reduce graph size by 1
            node_features = self._create_nodes(sample_size, graph_size - 1, n_depots=1, features=[coords])

            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)
            type_name = distribution + "_" + weight_dist

            return [
                TSPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size,
                    instance_id=i,
                    type=type_name
                )
                for i in range(sample_size)
            ]

        if problem.lower() == 'cvrp':
            coords = np.stack([
                self.sampler.sample_coords(n=graph_size + n_depots) for _ in range(sample_size)
            ])
            print('coords.shape 2', coords.shape)
            print('max_cap_factor', max_cap_factor)
            print('distribution', distribution)
            if max_cap_factor is None and weight_dist in ["gamma", "uniform"]:
                warnings.warn(f"No 'max_cap_factor' specified for ['gamma','uniform'] weight distributions."
                              f" Setting 'max_cap_factor' to default of 1.5")
                max_cap_factor = 1.5
            demands = np.stack([
                self.sampler.sample_weights(n=graph_size + n_depots, k=k, cap=cap, max_cap_factor=max_cap_factor)
                for _ in range(sample_size)
            ])
            node_features = self._create_nodes(sample_size, graph_size, n_depots=n_depots, features=[coords, demands])

            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)
            type_name = distribution + "_" + weight_dist
            return [
                CVRPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size + n_depots,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=1.0,  # demands are normalized
                    original_capacity=int(cap),
                    max_num_vehicles=k,
                    depot_type="uniform",  # depot stays uniform
                    coords_dist=self.sampler.coords_sampling_dist,
                    demands_dist=self.sampler.weights_sampling_dist,
                    instance_id=i,
                    type=type_name
                )
                for i in range(sample_size)
            ]

    def generate_nazari(self,
                        problem,
                        size: int,
                        graph_size: int,
                        normalize: bool = True,
                        k: Optional[int] = None,
                        cap: Optional[float] = None,
                        max_cap_factor: Optional[float] = None,
                        n_depots: int = 1,
                        **kwargs) -> Union[List[TSPInstance], List[CVRPInstance]]:
        """Generate uniform-random distributed data for either tsp, CVRP or CVRPTW

        Args:
            problem (str): problem for which to generate data
            size (int): size of dataset (number of problem instances)
            graph_size (int): size of problem instance graph (number of nodes)
            ### Additional for CVRP:
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            n_depots: number of depots (default = 1)

        Returns:
            RPDataset
        """

        if problem == "tsp":
            # From Kool et al. (2019)
            # Sample points randomly in [0, 1] square
            # tsp_data = [torch.FloatTensor(graph_size, 2).uniform_(0, 1) for i in range(size)]
            coords = np.stack([
                self.sampler.sample_coords(n=graph_size, **kwargs) for _ in range(size)
            ])

            # use dummy depot node as start node in tsp tour, therefore need to reduce graph size by 1
            node_features = self._create_nodes(size, graph_size - 1, n_depots=1, features=[coords])

            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)

            return [
                TSPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size,
                    instance_id=i
                )
                for i in range(size)
            ]
            # return tsp_data

        elif problem == "cvrp":

            if k is None:
                k = CVRP_DEFAULTS[graph_size][0]
            if cap is None:
                cap = CVRP_DEFAULTS[graph_size][1]

            coords = np.stack([
                self.sampler.sample_coords(n=graph_size + n_depots, **kwargs) for _ in range(size)
            ])
            demands = np.stack([
                self.sampler.sample_weights(n=graph_size + n_depots, k=k, cap=cap, max_cap_factor=max_cap_factor)
                for _ in range(size)
            ])
            node_features = self._create_nodes(size, graph_size, n_depots=n_depots, features=[coords, demands])

            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)

            return [
                CVRPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size + n_depots,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=1.0,  # demands are normalized
                    original_capacity=int(cap),
                    max_num_vehicles=k,
                    depot_type="uniform",
                    coords_dist=self.sampler.coords_sampling_dist,
                    demands_dist=self.sampler.weights_sampling_dist,
                    instance_id=i,
                    type="uniform"
                )
                for i in range(size)
            ]

    def generate_uchoa(self,
                       problem,
                       size: int,
                       graph_size: int,
                       normalize: bool = True,
                       n_depots: int = 1,
                       coords_sampling_dist: str = "uchoa",
                       depot_type: str = None,
                       customer_type: str = None,
                       demand_type: str = None) -> List[CVRPInstance]:

        #                        k: int,
        #                        cap: Optional[float] = None,
        #                        max_cap_factor: Optional[float] = None,

        """Generate Uchoa distributed data (currently only) for CVRP

        Args:
            problem (str): problem for which to generate data
            size (int): size of dataset (number of problem instances)
            graph_size (int): size of problem instance graph (number of nodes)
            normalize (bool): whether uchoa data should be normalized
            n_depots (int): amount of depots to be used
            coords_sampling_dist (str): needs to be "uchoa"
            depot_type (str): which type of depot position (R: Random, E: Eccentric, C: Central)
            customer_type (str): which type of depot position (R: Random, RC: RandomClustered, C: Clustered)

        Returns:
            RPDataset
        """

        assert coords_sampling_dist == "uchoa"
        if depot_type is not None:
            logger.info(f"Provided additional kwargs: depot_type = {depot_type}")
        if customer_type is not None:
            logger.info(f"Provided additional kwargs: customer_type = {customer_type}")
        if demand_type is not None:
            logger.info(f"Provided additional kwargs: demand_type = {demand_type}")


        elif problem != 'cvrp':
            raise ModuleNotFoundError(f"The Uchoa-distribution is currently not implemented for <{problem}> ")

        if self.verbose:
            print(f" Generating uchoa-distributed data with depot type {depot_type} and customer type {customer_type}")

        if problem == "cvrp":
            coords_scaled, c_types, d_types, grid_size = self.sampler.sample_coords_uchoa(n=graph_size,
                                                                                          num_samples=size,
                                                                                          depot_type=depot_type,
                                                                                          customer_type=customer_type)
            # re-scale coordinates to be betw. 0 and 1 (originally betw. 10 and 1000)
            coords = coords_scaled / grid_size

            if self.verbose:
                if depot_type and customer_type is not None:
                    print(f"Sampled {size} {problem} problems with graph of size {graph_size} and depot, customer type:"
                          f" {depot_type},{customer_type}")
                elif depot_type is None and customer_type is not None:
                    print(f"Sampled {size} {problem} problems with graph of size {graph_size} and random depot types "
                          f"and customer type: {customer_type}")
                elif customer_type is None and depot_type is not None:
                    print(f"Sampled {size} {problem} problems with graph of size {graph_size} and random customer types"
                          f"and depot type: {depot_type}")
                else:
                    print(f"Sampled {size} {problem} problems with graph of size {graph_size} and random depot, and "
                          f"customer types")
                print(f"Example of scaled coords for Instance 0: {coords_scaled[0, :5]}")
                print(f"Example of RE-SCALED coords for Instance 0: {coords[0, :5]}")

            demands, capacities, demand_types = self.sampler.sample_weights_uchoa(coords_scaled, demand_type)
            if self.verbose:
                print(f"Sampled {graph_size} demands with demand type mixed.")
                print(f"Capacities are not uniform and not normalized. Will be stored in CVRPInstance as "
                      f"'original capacity'")

            # print(f"demand_types", demand_types)
            # print(f"demands[:5]", demands[:5])
            # print(f"capacities", capacities)

            # replace depot demand to be 0
            demands[:, 0] = 0.0
            # demands = np.concatenate((np.array([0] * size).reshape(size, 1), demands),
            #                         axis=-1)  # add 0 demand for depot
            if normalize:
                # print(f"UNORMALIZED demands {demands[:2,:5]}")
                # print(f"capacities {capacities[:5]}")
                demands = demands.astype(float) / capacities[:, None].astype(float)
                # print(f"Normalized demands 1 {demands[0, :5]}")
                # demands = np.round(demands, 3)
                if self.verbose:
                    print(f"Normalized demands 2 {demands[:2, :5]}")

            node_features = self._create_nodes(size, graph_size, features=[coords, demands])
            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)

            return [
                CVRPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size + n_depots,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=1.0 if normalize else capacities[i],  # demands are normalized
                    coords_dist=c_types[i],
                    depot_type=d_types[i],
                    demands_dist=demand_types[i],
                    original_capacity=capacities[i],
                    original_locations=coords_scaled[i],
                    instance_id=i,
                    type="uchoa"
                )
                for i in range(size)
            ]

        elif problem == "tsp":
            raise NotImplementedError

        elif problem == "cvrptw":
            raise NotImplementedError

    @staticmethod
    def _out_RL4CO(problem,
                   instance_list: Union[List[TSPInstance], List[CVRPInstance], List[CVRPTWInstance]]):
        if problem.lower() == "tsp":
            # #{"locs": np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)}
            coords_list = [instance.coords for instance in instance_list]
            return {
                "locs": np.stack(coords_list).astype(np.float32)
            }
        elif problem.lower() == "cvrp":

            coords_list = [instance.coords[1:, :] for instance in instance_list]
            depot_list = [instance.coords[0, :] for instance in instance_list]
            demand_list = [instance.node_features[:, -1] * instance.original_capacity for instance in instance_list]
            capa_list = [instance.original_capacity for instance in instance_list]
            return {
                "depot": np.stack(depot_list).astype(np.float32),  # Depot location
                "locs": np.stack(coords_list).astype(np.float32),  # Node locations
                "demand": np.stack(demand_list).astype(np.float32),  # Demand, uniform integer 1 ... 9
                "capacity": np.stack(capa_list).astype(np.float32),
            }  # Capacity, same for whole dataset

        elif problem.lower() == "cvrptw":
            raise NotImplementedError
        else:
            warnings.warn(f"No valid problem type selected. Choose [tsp, CVRP, CVRPTW]")

    @staticmethod
    def _sub_sample(problem: str,
                    single_large_instance: Union[TSPInstance, CVRPInstance, CVRPTWInstance],
                    normalize_demands: bool,
                    sample_size: int = 64,
                    graph_size: int = 20,
                    n_depots: int = 1,
                    distribution: str = "uniform",
                    fixed_depot: bool = True,
                    # depot_node_idx: int = None,
                    selection_strategy: str = 'random'):

        # rp_instance = INSTANCE_TYPE[problem]
        # for node, demand in zip(single_large_instance.coords,single_large_instance.node_features[:,-1]):
        # completely random selection
        # print('len(single_large_instance.coords)', len(single_large_instance.coords))
        # print('single_large_instance.coords[0]', single_large_instance.coords[0])
        # print('single_large_instance.node_features[0]', single_large_instance.node_features[0])
        # print('single_large_instance.node_features[1]', single_large_instance.node_features[1])
        # print('single_large_instance.graph_size', single_large_instance.graph_size)
        # print('graph_size', graph_size)
        # print('sample_size', sample_size)
        # setting depot node to have 0 demands ---> TODO: keep fixed depot throughout subsampled instances?
        if not fixed_depot:
            graph_size = graph_size + 1 if problem.lower() in ["cvrp", "cvrptw"] else graph_size
        # else:
        #     single_large_instance = single_large_instance
        sample_nodes_id_all, selected_coords_all, selected_node_features_all = [], [], []
        for i_ in range(sample_size):
            # for j_ in range(graph_size):
            selected_nodes_id = random.sample(range(1, len(single_large_instance.coords)), graph_size)
            selected_coords = single_large_instance.coords[selected_nodes_id]
            # if problem in ["vrp", "cvrp"]:
            selected_node_features = single_large_instance.node_features[selected_nodes_id]
            if not fixed_depot:
                # re-adjust features for depot node --> first sampled instance is depot
                selected_node_features[0][0] = 1.0
                selected_node_features[0][1] = 0.0
                selected_node_features[0][-1] = 0.0
            else:
                fixed_depot_coords = np.expand_dims(single_large_instance.coords[0], axis=0)
                fixed_depot_features = np.expand_dims(single_large_instance.node_features[0], axis=0)
                # concat depot node to coords/features
                # print('selected_coords.shape before', selected_coords.shape)
                # print('selected_node_features.shape before', selected_node_features.shape)
                selected_coords = np.concatenate((fixed_depot_coords, selected_coords), axis=0)
                selected_node_features = np.concatenate((fixed_depot_features,
                                                         selected_node_features), axis=0)
                # print('selected_coords.shape', selected_coords.shape)
                # print('selected_node_features.shape', selected_node_features.shape)
            subsample_capa = single_large_instance.vehicle_capacity if normalize_demands \
                else single_large_instance.original_capacity
            if np.sum(selected_node_features[:, -1]) > (single_large_instance.max_num_vehicles * subsample_capa):
                warn(f"generated instance is infeasible just by demands vs. "
                     f"total available vehicle capacity of "
                     f"specified number of vehicles: {(single_large_instance.max_num_vehicles * subsample_capa)} "
                     f"> {np.sum(selected_node_features[:, -1])}.")
            sample_nodes_id_all.append(selected_nodes_id)
            selected_coords_all.append(selected_coords)
            selected_node_features_all.append(selected_node_features)
        if problem.lower() == "cvrp":
            return [
                CVRPInstance(
                    coords=selected_coords_all[i],
                    node_features=selected_node_features_all[i],
                    graph_size=graph_size + n_depots,
                    constraint_idx=[-1],  # demand is at last position of node features
                    vehicle_capacity=single_large_instance.vehicle_capacity,  # demands are normalized
                    coords_dist=single_large_instance.coords_dist,
                    depot_type=single_large_instance.depot_type,
                    demands_dist=single_large_instance.demands_dist,
                    original_capacity=single_large_instance.original_capacity,
                    original_locations=single_large_instance.original_locations,
                    max_num_vehicles=single_large_instance.max_num_vehicles,
                    instance_id=i,
                    type="subsampled_" + distribution
                )
                for i in range(sample_size)
            ]
        if problem.lower() == "tsp":
            return [
                TSPInstance(
                    coords=selected_coords_all[i],
                    node_features=selected_node_features_all[i],
                    graph_size=graph_size,
                    coords_dist=single_large_instance.coords_dist,
                    depot_type=single_large_instance.depot_type,
                    original_locations=single_large_instance.original_locations,
                    instance_id=i,
                    type="subsampled_" + distribution
                )
                for i in range(sample_size)
            ]
        if problem.lower() == "cvrptw":
            NotImplementedError
            # return [
            #         VRPTWInstance(
            #            coords=selected_coords_all[i],
            #            node_features=selected_node_features_all[i],
            #            graph_size=graph_size + n_depots,
            #            constraint_idx=[-1],  # demand is at last position of node features
            #            vehicle_capacity=single_large_instance.vehicle_capacity,  # demands are normalized
            #            coords_dist=single_large_instance.coords_dist,
            #            depot_type=single_large_instance.depot_type,
            #            demands_dist=single_large_instance.demands_dist,
            #            original_capacity=single_large_instance.original_capacity,
            #            original_locations=single_large_instance.original_locations,
            #            instance_id=i,
            #            type="subsampled_"+distribution
            #                         )
            #        for i in range(sample_size)
            #    ]

    @staticmethod
    def _distance_matrix(coords: np.ndarray, l_norm: Union[int, float] = 2):
        """Calculate distance matrix with specified norm. Default is l2 = Euclidean distance."""
        return np.linalg.norm(coords[:, :, None] - coords[:, None, :], ord=l_norm, axis=0)[:, :, :, None]

    @staticmethod
    def _create_nodes(size: int,
                      graph_size: int,
                      features: List,
                      n_depots: int = 1):
        """Create node id and type vectors and concatenate with other features."""
        return np.dstack((
            np.broadcast_to(np.concatenate((  # add id and node type (depot / customer)
                np.array([1] * n_depots +
                         [0] * graph_size)[:, None],  # depot/customer type 1-hot
                np.array([0] * n_depots +
                         [1] * graph_size)[:, None],  # depot/customer type 1-hot
            ), axis=-1), (size, graph_size + n_depots, 2)),
            *features,
        ))


def sub_sample(problem: str,
               single_large_instance: Union[TSPInstance, CVRPInstance, CVRPTWInstance],
               sample_size: int = 64,
               graph_size: int = 20,
               n_depots: int = 1,
               distribution: str = "uniform",
               fixed_depot: bool = True,
               seed: int = 1111,
               # depot_node_idx: int = None,
               selection_strategy: str = 'random'):
    if not fixed_depot:
        graph_size = graph_size + 1 if problem.lower() in ["cvrp", "cvrptw"] else graph_size
    # else:
    #     single_large_instance = single_large_instance
    sample_nodes_id_all, selected_coords_all, selected_node_features_all = [], [], []
    random.seed(seed)
    for i_ in range(sample_size):
        # for j_ in range(graph_size):
        selected_nodes_id = random.sample(range(1, len(single_large_instance.coords)), graph_size)
        selected_coords = single_large_instance.coords[selected_nodes_id]
        # if problem in ["vrp", "cvrp"]:
        selected_node_features = single_large_instance.node_features[selected_nodes_id]
        if not fixed_depot:
            # re-adjust features for depot node --> first sampled instance is depot
            selected_node_features[0][0] = 1.0
            selected_node_features[0][1] = 0.0
            selected_node_features[0][-1] = 0.0
        else:
            fixed_depot_coords = np.expand_dims(single_large_instance.coords[0], axis=0)
            fixed_depot_features = np.expand_dims(single_large_instance.node_features[0], axis=0)
            # concat depot node to coords/features
            # print('selected_coords.shape before', selected_coords.shape)
            # print('selected_node_features.shape before', selected_node_features.shape)
            selected_coords = np.concatenate((fixed_depot_coords, selected_coords), axis=0)
            selected_node_features = np.concatenate((fixed_depot_features,
                                                     selected_node_features), axis=0)
            # print('selected_coords.shape', selected_coords.shape)
            # print('selected_node_features.shape', selected_node_features.shape)
        if np.sum(selected_node_features[:, -1]) > single_large_instance.max_num_vehicles:
            warn(f"generated instance is infeasible just by demands vs. "
                 f"available vehicle capacity of specified number of vehicles.")
        sample_nodes_id_all.append(selected_nodes_id)
        selected_coords_all.append(selected_coords)
        selected_node_features_all.append(selected_node_features)
    # print('sample_nodes_id_all[0]', sample_nodes_id_all[0])

    if problem.lower() == "cvrp":
        return [
            CVRPInstance(
                coords=selected_coords_all[i],
                node_features=selected_node_features_all[i],
                graph_size=graph_size + n_depots,
                constraint_idx=[-1],  # demand is at last position of node features
                vehicle_capacity=single_large_instance.vehicle_capacity,  # demands are normalized
                coords_dist=single_large_instance.coords_dist,
                depot_type=single_large_instance.depot_type,
                demands_dist=single_large_instance.demands_dist,
                original_capacity=single_large_instance.original_capacity,
                original_locations=single_large_instance.original_locations,
                max_num_vehicles=single_large_instance.max_num_vehicles,
                instance_id=i,
                type="subsampled_" + distribution
            )
            for i in range(sample_size)
        ], sample_nodes_id_all
    if problem.lower() == "tsp":
        return [
            TSPInstance(
                coords=selected_coords_all[i],
                node_features=selected_node_features_all[i],
                graph_size=graph_size,
                coords_dist=single_large_instance.coords_dist,
                depot_type=single_large_instance.depot_type,
                original_locations=single_large_instance.original_locations,
                instance_id=i,
                type="subsampled_" + distribution
            )
            for i in range(sample_size)
        ]
    if problem.lower() == "cvrptw":
        pass
