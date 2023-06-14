import warnings
import logging
import torch
import numpy as np
from warnings import warn
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any

from data.sampler import DataSampler
from formats import RPInstance, TSPInstance, CVRPInstance, VRPTWInstance

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

# code from L2O-Meta
class RPGenerator:
    """Wraps data generation, loading and saving functionalities for routing problems."""
    RPS = ['tsp', 'cvrp', 'cvrptw']

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 float_prec: np.dtype = np.float32,
                 generator_args: dict = None):
        self._seed = seed
        self._rnd = np.random.default_rng(seed)
        self.verbose = verbose
        self.float_prec = float_prec
        self.sampler = DataSampler(verbose=verbose, **generator_args)

    def generate(self,
                 problem: str,
                 sample_size: int = 1000,
                 graph_size: int = 20,
                 distribution: str = 'uniform',
                 normalize: bool = True,
                 sampling_args: Dict = None,
                 generator_args: Dict = None,):
        """Generate data with corresponding RP generator function."""

        # generate uniformly distributed data from Nazari et al.
        if distribution is None:
            warn(f"No general distribution type for data generation specified. Defaulting to uniform distribution.")
            return self.generate_nazari(problem, sample_size, graph_size, normalize, **sampling_args)

        # generate uniformly distributed data from Nazari et al.
        elif distribution in ["nazari", "uniform"]:
            return self.generate_nazari(problem, size=sample_size, **sampling_args)

        # generate Uchoa-like distribution data
        elif distribution == "uchoa":
            return self.generate_uchoa(problem=problem,
                                       size=sample_size,
                                       graph_size=graph_size,
                                       normalize=normalize,
                                       n_depots=sampling_args['n_depots'],
                                       **generator_args)

        elif distribution == "solomon":
            raise NotImplementedError

        elif distribution in ["gm", "gaussian_mixture", "gm_unif_mixed"]:
            return self.generate_gm_unif(
                distribution=distribution,
                problem=problem,
                **sampling_args
            )

        # mix all existing distribution Samplers
        elif distribution == "mixed":
            raise NotImplementedError

        else:
            print("Specified Distribution not known - please enter one of the following distributions for sampling: ["
                  "'uniform', 'uchoa', 'solomon']")

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        if self._seed is None or (seed is not None and self._seed != seed):
            self._seed = seed
            self._rnd = np.random.default_rng(seed)
            self.sampler.seed(seed)

    def generate_gm_unif(self, distribution, problem, sample_size, graph_size, k, cap, n_depots):
        # generator_args already passed to Sampler init (for uniform fraction, n_components, mu_sampling_dist, ...)
        if problem.lower() == 'cvrp':
            coords = np.stack([
                self.sampler.sample_coords(n=graph_size + n_depots) for _ in range(sample_size)
            ])
            print('coords.shape 2', coords.shape)
            demands = np.stack([
                self.sampler.sample_weights(n=graph_size + n_depots, k=k, cap=cap)
                for _ in range(sample_size)
            ])
            node_features = self._create_nodes(sample_size, graph_size, n_depots=n_depots, features=[coords, demands])

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
                        depot_type="uniform",  # depot stays uniform
                        coords_dist=self.sampler.coords_sampling_dist,
                        demands_dist=self.sampler.weights_sampling_dist,
                        instance_id=i,
                        type=distribution
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
        """Generate uniform-random distributed data for either TSP, CVRP or VRPTW

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

            # use dummy depot node as start node in TSP tour, therefore need to reduce graph size by 1
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

        elif problem == "vrptw":
            pass

        else:
            raise ModuleNotFoundError(f"The corresponding generator for the problem <{problem}> does not exist.")

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
                    print(f"Normalized demands 2 {demands[:2,:5]}")

            node_features = self._create_nodes(size, graph_size, features=[coords, demands])

            # type cast
            coords = coords.astype(self.float_prec)
            node_features = node_features.astype(self.float_prec)

            return [
                CVRPInstance(
                    coords=coords[i],
                    node_features=node_features[i],
                    graph_size=graph_size+ n_depots,
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

        elif problem == "vrptw":
            raise NotImplementedError



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







