import torch
import numpy as np
from warnings import warn
from typing import Union, NamedTuple, Optional, Tuple, List, Dict, Any
import os
import sys
import pickle
import logging
import math
from scipy.linalg import block_diag
from data.data_utils import sample_triangular
from omegaconf import OmegaConf, DictConfig, ListConfig

# Uchoa instances:
GRID_SIZE = 1000

# standard Nazari vehicle capacities
CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.,
}
# vehicle capacities for instances with TW (from Solomon)
TW_CAPACITIES = {
    10: 250.,
    20: 500.,
    50: 750.,
    100: 1000.
}
# standard maximum fleet size
STD_K = {
    10: 6,
    20: 12,
    50: 24,
    100: 36,
}

logger = logging.getLogger(__name__)


def parse_from_cfg(x):
    if isinstance(x, DictConfig):
        return dict(x)
    elif isinstance(x, ListConfig):
        return list(x)
    else:
        return x


class DataSampler:
    """Sampler implementing different options to generate data for RPs."""

    def __init__(self,
                 n_components: int = 5,
                 n_dims: int = 2,
                 coords_sampling_dist: str = "uniform",
                 weights_sampling_dist: str = "random_int",
                 depot_type: str = "R",
                 customer_type: str = "R",
                 demand_type: int = 0,
                 # uchoa_distrib_type: str = "uchoa_depotc",
                 covariance_type: str = "diag",
                 mus: Optional[np.ndarray] = None,
                 sigmas: Optional[np.ndarray] = None,
                 mu_sampling_dist: str = "normal",
                 mu_sampling_params: Tuple = (0, 1),
                 sigma_sampling_dist: str = "uniform",
                 sigma_sampling_params: Tuple = (0.1, 0.3),
                 weights_sampling_params: Tuple = (1, 10),
                 uniform_fraction: float = 0.5,
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                 try_ensure_feasibility: bool = True,
                 verbose: bool = False,
                 ):
        """

        Args:
            n_components: number of mixture components
            n_dims: dimension of sampled features, e.g. 2 for Euclidean coordinates
            coords_sampling_dist: type of distribution to sample coordinates, one of ["uniform"]
            covariance_type: type of covariance matrix, one of ['diag', 'full']
            mus: user provided mean values for mixture components
            sigmas: user provided covariance values for mixture components
            mu_sampling_dist: type of distribution to sample initial mus, one of ['uniform', 'normal']
            mu_sampling_params: parameters for mu sampling distribution
            sigma_sampling_dist: type of distribution to sample initial sigmas, one of ['uniform', 'normal']
            sigma_sampling_params: parameters for sigma sampling distribution
            weights_sampling_dist: type of distribution to sample weights,
                                    one of ['random_int', 'uniform', 'gamma']
            weights_sampling_params: parameters for weight sampling distribution
            uniform_fraction: fraction of coordinates to be sampled uniformly for mixed instances
                              or parameter tuple to sample this per instance from a beta distribution
            random_state: seed integer or numpy random (state) generator
            try_ensure_feasibility: flag to try to ensure the feasibility of the generated instances
            verbose: verbosity flag to print additional info and warnings
        """
        self.nc = n_components
        self.f = n_dims
        self.coords_sampling_dist = coords_sampling_dist.lower()
        self.depot_type = depot_type  # uchoa
        self.customer_type = customer_type  # uchoa
        self.demand_type = demand_type  # uchoa
        self.covariance_type = covariance_type
        self.mu_sampling_dist = mu_sampling_dist.lower()
        self.mu_sampling_params = mu_sampling_params
        self.sigma_sampling_dist = sigma_sampling_dist.lower()
        self.sigma_sampling_params = sigma_sampling_params
        self.weights_sampling_dist = weights_sampling_dist.lower()
        self.weights_sampling_params = weights_sampling_params
        self.uniform_fraction = uniform_fraction
        self.try_ensure_feasibility = try_ensure_feasibility
        self.verbose = verbose
        self.normalizers = []

        # set random generator
        if random_state is None or isinstance(random_state, int):
            self.rnd = np.random.default_rng(random_state)
        else:
            self.rnd = random_state

        self._sample_nc, self._nc_params = False, None
        if not isinstance(n_components, int):
            print('n_components', n_components)
            print('type n_components', type(n_components))
            n_components = parse_from_cfg(n_components)
            print('n_components', n_components)
            print('type n_components', type(n_components))
            assert isinstance(n_components, (tuple, list))
            self._sample_nc = True
            self._nc_params = n_components
            self.nc = 1
        self._sample_unf_frac, self._unf_frac_params = False, None
        if not isinstance(uniform_fraction, float):
            uniform_fraction = parse_from_cfg(uniform_fraction)
            assert isinstance(uniform_fraction, (tuple, list))
            self._sample_unf_frac = True
            self._unf_frac_params = uniform_fraction
            self.uniform_fraction = None

        if self.coords_sampling_dist in ["gm", "gaussian_mixture", "gm_unif_mixed"]:
            # sample initial mu and sigma if not provided
            if mus is not None:
                assert (
                        (mus.shape[0] == self.nc and mus.shape[1] == self.f) or
                        (mus.shape[0] == self.nc * self.f)
                )
                self.mu = mus.reshape(self.nc * self.f)
            else:
                self.mu = self._sample_mu(mu_sampling_dist.lower(), mu_sampling_params)
            if sigmas is not None:
                assert not self._sample_nc
                assert (
                        (sigmas.shape[0] == self.nc and sigmas.shape[1] == (
                            self.f if covariance_type == "diag" else self.f ** 2))
                        or (sigmas.shape[0] == (
                    self.nc * self.f if covariance_type == "diag" else self.nc * self.f ** 2))
                )
                self.sigma = self._create_cov(sigmas, cov_type=covariance_type)
            else:
                covariance_type = covariance_type.lower()
                if covariance_type not in ["diag", "full"]:
                    raise ValueError(f"unknown covariance type: <{covariance_type}>")
                self.sigma = self._sample_sigma(sigma_sampling_dist.lower(), sigma_sampling_params, covariance_type)
        else:
            if self.coords_sampling_dist not in ["uniform", "uchoa"]:
                raise ValueError(f"unknown coords_sampling_dist: '{self.coords_sampling_dist}'")

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd = np.random.default_rng(seed)
        else:
            self.rnd = np.random.default_rng(123)

    def resample_gm(self):
        """Resample initial mus and sigmas."""
        self.mu = self._sample_mu(
            self.mu_sampling_dist,
            self.mu_sampling_params
        )
        self.sigma = self._sample_sigma(
            self.sigma_sampling_dist,
            self.sigma_sampling_params,
            self.covariance_type
        )

    def sample(self,
               n: int,
               k: int,
               cap: Optional[float] = None,
               max_cap_factor: Optional[float] = None,
               resample_mixture_components: bool = True,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
            weights: (n, )
        """
        coords = self.sample_coords(n=n, resample_mixture_components=resample_mixture_components, **kwargs)
        weights = self.sample_weights(n=n, k=k, cap=cap, max_cap_factor=max_cap_factor)

        return coords, weights

    def sample_coords(self,
                      n: int,
                      resample_mixture_components: bool = True,
                      **kwargs) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
        """
        if self.coords_sampling_dist == "uniform":
            coords = self._sample_unf_coords(n, **kwargs)
        else:
            if self._sample_nc:
                self.nc = self.sample_rnd_int(*self._nc_params)
                self.resample_gm()
            elif resample_mixture_components:
                self.resample_gm()

            if self.coords_sampling_dist == "gm_unif_mixed":
                print('self.coords_sampling_dist', self.coords_sampling_dist)
                print('self._sample_unf_frac', self._sample_unf_frac)
                print('self.uniform_fraction', self.uniform_fraction)
                if self._sample_unf_frac:
                    # if specified, sample the fraction value from a beta distribution
                    v = self._sample_beta(1, *self._unf_frac_params)
                    self.uniform_fraction = 0.0 if v <= 0.04 else v
                    # print(self.uniform_fraction)
                n_unf = math.floor(n * self.uniform_fraction)
                n_gm = n - n_unf
                logger.info(f'Sampled {n_unf} uniform and {n_gm} gm customers for '
                            f'{self.coords_sampling_dist} distribution')
                unf_coords = self._sample_unf_coords(n_unf, **kwargs)
                n_per_c = math.ceil(n_gm / self.nc)
                gm_coords = self._sample_gm_coords(n_per_c, n_gm, **kwargs)
                coords = np.vstack((unf_coords, gm_coords))
                print('coords.shape', coords.shape)
            else:
                n_per_c = math.ceil(n / self.nc)
                coords = self._sample_gm_coords(n_per_c, n, **kwargs)
            # depot stays uniform!
            coords[0] = self._sample_unf_coords(1, **kwargs)

        return coords.astype(np.float32)

    def sample_weights(self,
                       n: int,
                       k: int,
                       cap: Optional[float] = None,
                       max_cap_factor: Optional[float] = None,
                       ) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle

        Returns:
            weights: (n, )
        """
        n_wo_depot = n - 1
        # sample a weight for each point
        if self.weights_sampling_dist in ["random_int", "random_k_variant"]:
            assert cap is not None, \
                f"weight sampling dist 'random_int' requires <cap> to be specified"

            if self.weights_sampling_dist == "random_int":
                # standard integer sampling adapted from Nazari et al. and Kool et al.
                weights = self.rnd.integers(1, 10, size=(n_wo_depot,))
                normalizer = cap + 1
                if self.try_ensure_feasibility:
                    need_k = int(np.ceil(weights.sum() / cap))
                    normalizer *= ((need_k / k) + 0.1)
            else:
                weights = self.rnd.integers(1, (cap - 1) // 2, size=(n_wo_depot,))
                # normalize weights by total max capacity of vehicles
                _div = max(2, self.sample_rnd_int(k // 4, k))
                if max_cap_factor is not None:
                    normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / _div
                else:
                    normalizer = np.ceil((weights.sum(axis=-1)) * 1.08) / _div
        elif self.weights_sampling_dist in ["uniform", "gamma"]:
            assert max_cap_factor is not None, \
                f"weight sampling dists 'uniform' and 'gamma' require <max_cap_factor> to be specified"
            if self.weights_sampling_dist == "uniform":
                weights = self._sample_uniform(n_wo_depot, *self.weights_sampling_params)
            elif self.weights_sampling_dist == "gamma":
                weights = self._sample_gamma(n_wo_depot, *self.weights_sampling_params)
            else:
                raise ValueError
            weights = weights.reshape(-1)
            if self.verbose:
                if np.any(weights.max(-1) / weights.min(-1) > 10):
                    warn(f"Largest weight is more than 10-times larger than smallest weight.")
            # normalize weights w.r.t. norm capacity of 1.0 per vehicle and specified max_cap_factor
            # using ceiling adds a slight variability in the total sum of weights,
            # such that not all instances are exactly limited to the max_cap_factor
            normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / k
        else:
            raise ValueError(f"unknown weight sampling distribution: {self.weights_sampling_dist}")

        weights = weights / normalizer

        if np.sum(weights) > k:
            if self.verbose:
                warn(f"generated instance is infeasible just by demands vs. "
                     f"total available capacity of specified number of vehicles.")
            if self.try_ensure_feasibility:
                raise RuntimeError

        weights = np.concatenate((np.array([0]), weights), axis=-1)  # add 0 weight for depot
        return weights.astype(np.float32)

    def sample_rnd_int(self, lower: int, upper: int) -> int:
        """Sample a single random integer between lower (inc) and upper (excl)."""
        return self.rnd.integers(lower, upper, 1)[0]

    # from DPDP (Kool et al. 2020)
    def sample_coords_uchoa(self,
                            n: int,
                            num_samples: int,
                            depot_type: [str] = None,
                            customer_type: [str] = None,
                            int_locs: bool = True,
                            min_seeds: int = 3,
                            max_seeds: int = 8) -> Tuple[np.ndarray, List, List, int]:

        """
        Args:
            n: number of samples to draw
            num_samples: number of instances to sample --> batch size
            depot_type: which type of depot centrality (central, eccentric, random)
            customer_type: node distribution
            min_seeds: min nr. of seeds to be sampled
            max_seeds: max nr. of seeds to be sampled

        Returns:
            coords: (n, n_dims)
        """

        if depot_type is None and self.verbose:
            logger.info(f"Sampling uchoa-type data with mixed depot types (central, eccentric, random)")
        else:
            dep_type = None
            if depot_type == 'C':
                dep_type = "central"  # (500, 500)
            elif depot_type == 'E':
                dep_type = "eccentric"  # (0, 0),
            elif depot_type == 'R':
                dep_type = "random"
            if self.verbose:
                logger.info(f"Sampling uchoa-type data with depot type: {dep_type}")

        # Depot Position
        # 0 = central (500, 500), 1 = eccentric (0, 0), 2 = random
        depot_types = (np.random.rand(num_samples) * 3).astype(int)
        # (torch.rand(batch_size, device=device) * 3).int()
        if depot_type is not None:  # else mix
            # Central, Eccentric, Random
            codes = {'C': 0, 'E': 1, 'R': 2}
            depot_types[:] = codes[depot_type.upper()]

        depot_locations = np.random.rand(num_samples, 2) * GRID_SIZE
        depot_locations[depot_types == 0] = GRID_SIZE / 2
        depot_locations[depot_types == 1] = 0

        # Customer position
        # 0 = random, 1 = clustered, 2 = random clustered 50/50
        # We always do this so we always pull the same number of random numbers
        customer_types = (np.random.rand(num_samples) * 3).astype(int)

        if customer_type is not None:  # else Mix
            # Random, Clustered, Random-Clustered (half half)
            codes = {'R': 0, 'C': 1, 'RC': 2}
            customer_types[:] = codes[customer_type.upper()]
        if self.verbose:
            if customer_type is None:
                logger.info(f"Sampling uchoa-type data with mixed customer types "
                            f"(Random, Clustered, Random-Clustered (half half))")
            else:
                logger.info(f"Sampling uchoa-type data with customer type: {customer_type}")

        # Sample number of seeds uniform (inclusive)
        num_seeds = (np.random.rand(num_samples) * ((max_seeds - min_seeds) + 1)).astype(int) + min_seeds

        # We sample random and clustered coordinates for all instances, this way, the instances in the 'mix' case
        # Will be exactly the same as the instances in one of the tree 'not mixed' cases and we can reuse evaluations
        rand_coords = np.random.rand(num_samples, n, 2) * GRID_SIZE
        clustered_coords = self.generate_clustered_uchoa(num_seeds, n, max_seeds=max_seeds)

        # Clustered
        rand_coords[customer_types == 1] = clustered_coords[customer_types == 1]
        # Half clustered
        rand_coords[customer_types == 2, :(n // 2)] = clustered_coords[customer_types == 2, :(n // 2)]

        # stack depot coord and customer coords
        coords = np.stack([np.vstack((depot_locations[i].reshape(1, 2), rand_coords[i])) for i in range(num_samples)])
        coords = coords.astype(int) if int_locs else coords
        return coords, customer_types, depot_types, GRID_SIZE

    # from DPDP (Kol et al. 2020)
    # @staticmethod
    def sample_weights_uchoa(self,
                             coordinates: np.ndarray,
                             demand_type: int = None) -> Tuple[np.ndarray, np.ndarray, List]:

        batch_size, graph_size, _ = coordinates.shape
        # Demand distribution
        # 0 = unitary (1)
        # 1 = small values, large variance (1-10)
        # 2 = small values, small variance (5-10)
        # 3 = large values, large variance (1-100)
        # 4 = large values, large variance (50-100)
        # 5 = depending on quadrant top left and bottom right (even quadrants) (1-50), others (51-100) so add 50
        # 6 = many small, few large most (70 to 95 %, unclear so take uniform) from (1-10), rest from (50-100)
        lb = torch.tensor([1, 1, 5, 1, 50, 1, 1], dtype=torch.long)
        ub = torch.tensor([1, 10, 10, 100, 100, 50, 10], dtype=torch.long)
        if demand_type is not None:
            customer_positions = (torch.ones(batch_size, device="cpu") * demand_type).long()
        else:
            customer_positions = (torch.rand(batch_size, device="cpu") * 7).long()
        # customer_positions = (torch.ones(batch_size)*2).long()
        # for i in range(len(customer_positions)):
        #    # print(dem_type[i])
        #    if customer_positions[i] == 0:
        #        customer_positions[i] = 2
        #    elif customer_positions[i] == 3:
        #        customer_positions[i] = 2
        if self.verbose:
            logger.info(f"demand types are mixed by default; {customer_positions[:5]}")
        lb_ = lb[customer_positions, None]
        ub_ = ub[customer_positions, None]
        # Make sure we always sample the same number of random numbers
        rand_1 = torch.rand(batch_size, graph_size)
        rand_2 = torch.rand(batch_size, graph_size)
        rand_3 = torch.rand(batch_size)
        demands = (rand_1 * (ub_ - lb_ + 1).float()).long() + lb_
        # either both smaller than grid_size // 2 results in 2 inequalities satisfied, or both larger 0
        # in all cases it is 1 (odd quadrant) and we should add 50
        if customer_positions.size() == 1:
            if customer_positions != torch.tensor([5]):
                demands[customer_positions == 5] += ((coordinates[customer_positions == 5] < GRID_SIZE // 2).astype(
                    int).sum(
                    -1) == 1).astype(int) * 50
        # slightly different than in the paper we do not exactly pick a value between 70 and 95 % to have a large value
        # but based on the threshold we let each individual location have a large demand with this probability
        demands_small = demands[customer_positions == 6]
        demands[customer_positions == 6] = torch.where(
            rand_2[customer_positions == 6] > (rand_3 * 0.25 + 0.70)[customer_positions == 6, None],
            demands_small,
            (rand_1[customer_positions == 6] * (100 - 50 + 1)).long() + 50
        )
        r = sample_triangular(batch_size, 3, 6, 25)
        capacity = torch.ceil(r * demands.float().mean(-1)).long()
        # It can happen that demand is larger than capacity, so cap demand
        demand = torch.min(demands, capacity[:, None])

        return demand.cpu().numpy(), capacity.cpu().numpy(), customer_positions.cpu().tolist()

    # from DPDP (Kol et al. 2020)
    @staticmethod
    def generate_clustered_uchoa(num_seeds, graph_size, max_seeds=None):
        if max_seeds is None:
            max_seeds = num_seeds.max()
            # .item()
        num_samples = num_seeds.shape[0]
        batch_rng = torch.arange(num_samples, dtype=torch.long)
        # batch_rng = np.arange(num_samples, dtype=int)
        seed_coords = (torch.rand(num_samples, max_seeds, 2) * GRID_SIZE)
        # We make a little extra since some may fall off the grid
        n_try = graph_size * 2
        while True:
            loc_seed_ind = (torch.rand(num_samples, n_try, device="cpu") * num_seeds[:, None].astype(float)).long()
            # loc_seed_ind = (np.random.rand(num_samples, n_try) * num_seeds[:, None].astype(float)).astype(int)
            loc_seeds = seed_coords[batch_rng[:, None], loc_seed_ind]
            alpha = torch.rand(num_samples, n_try) * 2 * math.pi
            # alpha = np.random.rand(num_samples, n_try) * 2 * math.pi
            d = -40 * torch.rand(num_samples, n_try).log()
            # d = -40 * np.log(np.random.rand(num_samples, n_try))
            coords = torch.stack((torch.sin(alpha), torch.cos(alpha)), -1) * d[:, :, None] + loc_seeds
            coords.size()
            feas = ((coords >= 0) & (coords <= GRID_SIZE)).sum(-1) == 2
            feas_topk, ind_topk = feas.byte().topk(graph_size, dim=-1)
            # feas_topk, ind_topk = np.byte(feas).topk(graph_size, dim=-1)
            if feas_topk.all():
                break
            n_try *= 2  # Increase if this fails
        return np.array(coords.cpu()[batch_rng[:, None], ind_topk])

    def _sample_mu(self, dist: str, params: Tuple):
        size = self.nc * self.f
        if dist == "uniform":
            return self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            return self._sample_normal(size, params[0], params[1])
        elif dist == "ring":
            return self._sample_ring(self.nc, params).reshape(-1)
        elif dist == "io_ring":
            return self._sample_io_ring(self.nc).reshape(-1)
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")

    def _sample_sigma(self, dist: str, params: Tuple, cov_type: str):
        if cov_type == "full":
            size = self.nc * self.f ** 2
        else:
            size = self.nc * self.f
        if dist == "uniform":
            x = self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            x = np.abs(self._sample_normal(size, params[0], params[1]))
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")
        return self._create_cov(x, cov_type=cov_type)

    def _create_cov(self, x, cov_type: str):
        if cov_type == "full":
            # create block diagonal matrix to model covariance only
            # between features of each individual component
            x = x.reshape((self.nc, self.f, self.f))
            return block_diag(*x.tolist())
        else:
            return np.diag(x.reshape(-1))

    def _sample_uniform(self,
                        size: Union[int, Tuple[int, ...]],
                        low: Union[int, np.ndarray] = 0.0,
                        high: Union[int, np.ndarray] = 1.0):
        return self.rnd.uniform(size=size, low=low, high=high)

    def _sample_normal(self,
                       size: Union[int, Tuple[int, ...]],
                       mu: Union[int, np.ndarray],
                       sigma: Union[int, np.ndarray]):
        return self.rnd.normal(size=size, loc=mu, scale=sigma)

    def _sample_gamma(self,
                      size: Union[int, Tuple[int, ...]],
                      alpha: Union[int, np.ndarray],
                      beta: Union[int, np.ndarray]):
        return self.rnd.gamma(size=size, shape=alpha, scale=beta)

    def _sample_beta(self,
                     size: Union[int, Tuple[int, ...]],
                     alpha: Union[int, np.ndarray],
                     beta: Union[int, np.ndarray]):
        return self.rnd.beta(size=size, a=alpha, b=beta)

    def _sample_unf_coords(self, n: int, **kwargs) -> np.ndarray:
        """Sample coords uniform in [0, 1]."""
        return self.rnd.uniform(size=(n, self.f))

    def _sample_gm_coords(self, n_per_c: int, n: Optional[int] = None, **kwargs) -> np.ndarray:
        """Sample coordinates from k Gaussians."""
        coords = self.rnd.multivariate_normal(
            mean=self.mu,
            cov=self.sigma,
            size=n_per_c,
        ).reshape(-1, self.f)  # (k*n, f)
        if n is not None:
            coords = coords[:n]  # if k % n != 0, some of the components have 1 more sample than others
        # normalize coords in [0, 1]
        return self._normalize_coords(coords)

    def _sample_ring(self, size: int, radius_range: Tuple = (0, 1)):
        """inspired by https://stackoverflow.com/a/41912238"""
        # eps = self.rnd.standard_normal(1)[0]
        if size == 1:
            angle = self.rnd.uniform(0, 2 * np.pi, size)
            # eps = self.rnd.uniform(0, np.pi, size)
        else:
            angle = np.linspace(0, 2 * np.pi, size)
        # angle = np.linspace(0+eps, 2*np.pi+eps, size)
        # angle = rnd.uniform(0, 2*np.pi, size)
        # angle += self.rnd.standard_normal(size)*0.05
        angle += self.rnd.uniform(0, np.pi / 3, size)
        d = np.sqrt(self.rnd.uniform(*radius_range, size))
        # d = np.sqrt(rnd.normal(np.mean(radius_range), (radius_range[1]-radius_range[0])/2, size))
        return np.concatenate((
            (d * np.cos(angle))[:, None],
            (d * np.sin(angle))[:, None]
        ), axis=-1)

    def _sample_io_ring(self, size: int):
        """sample an inner and outer ring."""
        # have approx double the number of points in outer ring than inner ring
        num_inner = size // 3
        num_outer = size - num_inner
        inner = self._sample_ring(num_inner, (0.01, 0.2))
        outer = self._sample_ring(num_outer, (0.21, 0.5))
        return np.vstack((inner, outer))

    @staticmethod
    def _normalize_coords(coords: np.ndarray):
        """Applies joint min-max normalization to x and y coordinates."""
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        max_val = coords.max()  # joint max to preserve relative spatial distances
        coords[:, 0] = coords[:, 0] / max_val
        coords[:, 1] = coords[:, 1] / max_val
        return coords

    @staticmethod
    def _create_edges(coords: np.ndarray, l_norm: Union[int, float] = 2):
        """Calculate distance matrix with specified norm. Default is l2 = Euclidean distance."""
        return np.linalg.norm(coords[:, :, None] - coords[:, None, :], ord=l_norm, axis=-1)[:, :, :, None]

    def _sample_tw(self,
                   size: int,
                   graph_size: int,
                   edges: np.ndarray,
                   service_duration: Union[int, float, np.ndarray],
                   service_window, time_factor, tw_expansion,
                   n_depots: int = 1):
        """Sample feasible time windows."""
        # TW start needs to be feasibly reachable directly from depot
        min_t = np.ceil(edges[:, 0,
                        1:] - service_duration + 1)  # TODO: adapt to multiple vehcs with different start depots --> n_start_depots & n_end_depots?
        # TW end needs to be early enough to perform service and return to depot until end of service window
        max_t = service_window - np.ceil(edges[:, 0, 1:] + 1)
        # horizon allows for the feasibility of reaching nodes /
        # returning from nodes within the global tw (service window)
        horizon = np.concatenate((min_t, max_t), axis=-1)
        epsilon = np.maximum(np.abs(self._rnds.standard_normal([size, graph_size])), 1 / time_factor)

        # sample earliest start times a
        a = self._rnds.randint(horizon[:, :, 0], horizon[:, :, 1])
        # calculate latest start times b, which is
        # = a + service_time_expansion x normal random noise, all limited by the horizon
        b = np.minimum(a + tw_expansion * time_factor * epsilon, horizon[:, :, -1]).astype(int)

        # add depot TWs and return
        return (
            # np.concatenate((np.array([0]*size)[:, None], a), axis=-1),
            np.concatenate((np.zeros((size, n_depots)), a), axis=-1),
            np.concatenate((
                np.broadcast_to(np.array([service_window])[:, None], (size, n_depots)),
                b
            ), axis=-1),
        )

    def get_normalizers(self) -> List:
        return self.normalizers
