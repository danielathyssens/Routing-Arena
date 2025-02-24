import torch
import numpy as np
from warnings import warn
from typing import Union, NamedTuple, Optional, Tuple, List, Dict, Any
import os
import sys
import pickle
import logging
import math

from numpy import ndarray, dtype, floating
# from numpy.typing import _32Bit
from scipy.linalg import block_diag
from scipy.spatial import distance
import scipy.stats as stats
from data.data_utils import sample_triangular
from omegaconf import OmegaConf, DictConfig, ListConfig

# from formats import ObjectiveDict

# Uchoa instances:
GRID_SIZE = 1000

# standard Nazari vehicle capacities
CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}
# vehicle capacities for instances with TW (from Solomon)
TW_CAPACITIES = {
    10: 250.,
    20: 500.,
    50: 750.,
    100: 1000.,
    200: 2000
}
# standard maximum fleet size
STD_K = {
    10: 6,
    20: 12,
    50: 24,
    100: 36,
    200: 40
}

logger = logging.getLogger(__name__)

# Solomon instance naming components
GROUPS = ["r", "c", "rc"]
TYPES = ["1", "2"]
TW_FRACS = [0.25, 0.5, 0.75, 1.0]


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
                 twindow_sampling_dist: str = None,
                 solomon_tw_cfg: Dict = None,
                 depot_type: str = None,
                 customer_type: str = None,
                 demand_type: int = None,
                 normalize_demands: bool = None,
                 normalize_tws: bool = True,
                 single_large_instance: str = None,
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
                 beta_exp: float = 10.0,
                 radius: float = 0.3,
                 pm: float = 0.4,
                 angle_range: tuple = (0,2),
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                 try_ensure_feasibility: bool = True,
                 verbose: bool = False,
                 ):
        """

        Args:
            n_components: number of mixture components
            n_dims: dimension of sampled features, e.g. 2 for Euclidean coordinates
            coords_sampling_dist: type of distribution to sample coordinates, one of ["uniform", "gm", "gm_unif_mixed"]
            covariance_type: type of covariance matrix, one of ['diag', 'full']
            mus: user provided mean values for mixture components
            sigmas: user provided covariance values for mixture components
            mu_sampling_dist: type of distribution to sample initial mus, one of ['uniform', 'normal']
            mu_sampling_params: parameters for mu sampling distribution
            sigma_sampling_dist: type of distribution to sample initial sigmas, one of ['uniform', 'normal']
            sigma_sampling_params: parameters for sigma sampling distribution
            weights_sampling_dist: type of distribution to sample weights,
                                    one of ['random_int', 'uniform', 'gamma', 'uchoa']
            normalize_demands: whether to normalize demands by capacity
            weights_sampling_params: parameters for weight sampling distribution
            uniform_fraction: fraction of coordinates to be sampled uniformly for mixed instances
                              or parameter tuple to sample this per instance from a beta distribution
            beta_exp: lambda scale value for exponential distribution in explosion mutation
            pm: anchor for ration mutation of originally uniform distrib. coordinates
            angle_range: angle for ration mutation of originally uniform distrib. coordinates
            random_state: seed integer or numpy random (state) generator
            try_ensure_feasibility: flag to try to ensure the feasibility of the generated instances
            verbose: verbosity flag to print additional info and warnings
        """
        self.nc = n_components
        self.f = n_dims
        self.coords_sampling_dist = coords_sampling_dist.lower()
        self.twindow_sampling_dist = twindow_sampling_dist.lower() if twindow_sampling_dist is not None else None
        self.depot_type = depot_type  # uchoa
        self.customer_type = customer_type  # uchoa
        self.demand_type = demand_type  # uchoa
        print('self.depot_type', self.depot_type)
        print('self.customer_type', self.customer_type)
        print('self.demand_type', self.demand_type)
        self.covariance_type = covariance_type
        self.mu_sampling_dist = mu_sampling_dist.lower()
        self.mu_sampling_params = mu_sampling_params
        self.sigma_sampling_dist = sigma_sampling_dist.lower()
        self.sigma_sampling_params = sigma_sampling_params
        self.weights_sampling_dist = weights_sampling_dist.lower()
        self.weights_sampling_params = weights_sampling_params
        self.uniform_fraction = uniform_fraction
        self.beta_exp = beta_exp
        self.radius = radius
        self.pm = pm
        self.angle_range = angle_range
        self.try_ensure_feasibility = try_ensure_feasibility
        self.verbose = verbose
        self.normalize_demands = normalize_demands
        self.normalize_tws = normalize_tws
        # self.tw_frac = solomon_tw_fraction
        print('self.normalize_demands', self.normalize_demands)
        self.normalizers = []

        # set random generator
        if random_state is None or isinstance(random_state, int):
            self.rnd = np.random.default_rng(random_state)
        else:
            self.rnd = random_state

        self._sample_nc, self._nc_params = False, None
        if not isinstance(n_components, int):
            n_components = parse_from_cfg(n_components)
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

        ### COORDS
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
        elif coords_sampling_dist == "explosion":
            self.s = self.rnd.exponential(scale=self.beta_exp)
        else:
            if self.coords_sampling_dist not in ["uniform", "uchoa", "explosion", "rotation"]:
                raise ValueError(f"unknown coords_sampling_dist: '{self.coords_sampling_dist}'")

        ### TWs
        # self.twindow_sampling_dist is not None and
        if self.twindow_sampling_dist == "solomon":
            # get cfg_params
            # SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
            # load estimated stats:
            LPATH = os.path.abspath(solomon_tw_cfg["stats_path"])
            with open(LPATH, 'rb') as f:
                dset_cfg_stats = pickle.load(f)
            cfg_stats = dset_cfg_stats[solomon_tw_cfg["group"]][
                solomon_tw_cfg["group"] + str(solomon_tw_cfg["type"])][f"tw_frac={solomon_tw_cfg['tw_frac']}"]

            # set key-value pairs from Solomon instance stats
            # as InstanceSampler instance attributes
            # print('cfg_stats[0]', cfg_stats[0])
            # print('cfg_stats[1]', cfg_stats[1])
            for k, v in cfg_stats[0].items():
                setattr(self, k, v)

            # TW start sampler
            ### e.g.
            # 'tw_start': {'dist': 'KDE', 'params': <scipy.stats.kde.gaussian_kde object at 0x7ff5fe8c72e0>,
            # 'tw_start': {'dist': 'normal', 'params': (0.34984000000000004, 0.23766332152858588)}
            if self.tw_start['dist'] == "gamma":
                self.tw_start_sampler = stats.gamma(*self.tw_start['params'])
            elif self.tw_start['dist'] == "normal":
                self.tw_start_sampler = stats.norm(*self.tw_start['params'])
            elif self.tw_start['dist'] == "KDE":
                self.tw_start_sampler = self.tw_start['params']  # assigns fitted KDE model
            else:
                raise ValueError(f"unknown tw_start_sampler cfg: {self.tw_start}.")

            # TW len sampler
            if self.tw_len['dist'] == "const":
                self.tw_len_sampler = self.tw_len['params']  # this is the normalized len, there is also self.org_tw_len
            elif self.tw_len['dist'] == "gamma":
                self.tw_len_sampler = stats.gamma(*self.tw_len['params'])
            elif self.tw_len['dist'] == "normal":
                self.tw_len_sampler = stats.norm(*self.tw_len['params'])
            elif self.tw_len['dist'] == "KDE":
                self.tw_len_sampler = self.tw_len['params']  # assigns fitted KDE model
            else:
                raise ValueError(f"unknown tw_len_sampler cfg: {self.tw_len}.")

            # service time in Solomon data is constant for each instance, so mean == exact value
            self.service_time = self.norm_summary.loc['mean', 'service_time']

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

    def sample_tsp(self,
                   n: int,
                   resample_mixture_components: bool = True,
                   **kwargs):
        """
        Args:
            n: number of samples to draw (coordinates)
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance
        Returns:
            coords: (n, n_dims)
        """
        coords, c_types, _ = self.sample_coords(n=n, resample_mixture_components=resample_mixture_components, **kwargs)

        return coords

    def sample_cvrp(self,
                    sample_size: int,
                    graph_size: int,
                    k: int,
                    cap: Optional[float] = None,
                    max_cap_factor: Optional[float] = None,
                    n_depots: int = 1,
                    resample_mixture_components: bool = True,
                    feasibility_insurance: bool = None,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """
        Args:
            sample_size: number of cvrp instances (to be ignored here)
            graph_size: number of samples to draw (coordinates)
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            n_depots: how many depots
            normalize_demands: whether to normalize demands with capa
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance
            feasibility_insurance: overwriting feasibility check property in sample_weights

        Returns:
            coords: (n, n_dims)
            weights: (n, )
        """

        n = graph_size
        if max_cap_factor is None and self.weights_sampling_dist in ["gamma", "uniform"]:
            warn(f"No 'max_cap_factor' specified for ['gamma','uniform'] weight distributions."
                 f" Setting 'max_cap_factor' to default of 1.5")
            max_cap_factor = 1.5

        # print('kwargs in sample_cvrp', kwargs)
        coords, c_types, d_types = self.sample_coords(n=n + n_depots,
                                                      resample_mixture_components=resample_mixture_components,
                                                      **kwargs)
        c_probs = np.ones_like(coords)

        if self.coords_sampling_dist == "uchoa":
            c_type, d_type = c_types[0], d_types[0]
        else:
            c_type, d_type = c_types, d_types
        # print('feasibility_insurance in samplecvrp', feasibility_insurance)
        weights, original_capa, q_type = self.sample_weights(n=n + n_depots, k=k, cap=cap,
                                                             max_cap_factor=max_cap_factor,
                                                             coords=coords,
                                                             demand_type=self.demand_type,
                                                             normalize=self.normalize_demands,
                                                             feasibility_insurance=feasibility_insurance)

        weights = weights[..., np.newaxis]

        return coords, c_probs, weights, original_capa, q_type, c_type, d_type

    def sample_cvrptw(self,
                      sample_size: int,
                      graph_size: int,
                      k: int,
                      cap: Optional[float] = None,
                      max_cap_factor: Optional[float] = None,
                      n_depots: int = 1,
                      resample_mixture_components: bool = True,
                      service_window=1000,
                      service_duration=10,
                      time_factor=100.0,
                      tw_expansion=3.0,
                      solomon_tw_cfg: Dict = None,
                      early_tw_soft=False,
                      late_tw_soft=False,
                      early_tw_penalty=0.1,
                      late_tw_penalty=0.5,
                      org_service_horizon: int = 100,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Args:
            sample_size: number of cvrptw instances (to be ignored here)
            graph_size: number of samples to draw (graph_size)
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            n_depots: how many depots
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance
            # For generic CVRPTW instance:
                service_window (int): gives maximum of time units
                service_duration (int): duration of service
                time_factor (float): value to map from distances in [0, 1] to time units (transit times)
                tw_expansion (float): expansion factor of TW w.r.t. service duration
                early_tw_soft (bool): soft TW for early arrival (can arrive early - but incur penalty)
                late_tw_soft (bool): soft TW for late departure (can depart late - but incur penalty)
                early_tw_penalty (float): amount of penalty for early arrival
                late_tw_penalty (float): amount of penalty for late departure
            # For Solomon-type Instance:
                org_service_horizon (int): gives normalizing value for time windows btw. 0-1
                solomon_cfg (Dict): config for solomon data sampling (tw_frac, group, type)


        Returns:
            coords: (n, n_dims)
            weights: (n, )
            tw: (n, n_dims)
        """

        if max_cap_factor is None and self.weights_sampling_dist in ["gamma", "uniform"]:
            warn(f"No 'max_cap_factor' specified for ['gamma','uniform'] weight distributions."
                 f" Setting 'max_cap_factor' to default of 1.5")
            max_cap_factor = 1.5

        # coords = self.sample_coords(n=n + n_depots, resample_mixture_components=resample_mixture_components, **kwargs)
        # weights = self.sample_weights(n=n + n_depots, k=k, cap=cap, max_cap_factor=max_cap_factor)
        # time_w = self.sample_tw()
        i = 0
        tw = None
        time_to_depot = None
        service_time = None
        feasible = False
        print('cap', cap)
        while not feasible:
            if i > 100:
                raise RuntimeError(f"Encountered many infeasible instances during sampling. "
                                   f"Try to adapt sampling parameters.")
            try:
                coords, c_types, d_types = self.sample_coords(n=graph_size + n_depots,
                                                              resample_mixture_components=resample_mixture_components,
                                                              **kwargs)
                coords = np.expand_dims(coords, axis=0)

                weights, capacity_original, q_type = self.sample_weights(n=graph_size + n_depots, k=k, cap=cap,
                                                                         max_cap_factor=max_cap_factor, coords=coords,
                                                                         normalize=self.normalize_demands)
                weights = np.expand_dims(weights, axis=0)
                if self.twindow_sampling_dist != "solomon":
                    # default to generic tw sampling
                    # if not early_tw_soft and not late_tw_soft:
                    #     cost_function_params = ObjectiveDict(type="edge_cost_1d")
                    # else:
                    #     cost_function_params = ObjectiveDict(type="edge_cost_1d", params={})
                    # calculate edge weights as l2 distance * time_factor
                    edges = self._create_edges(coords * time_factor)
                    num_v = k if k is not None else STD_K[graph_size]
                    capa = cap if cap is not None else TW_CAPACITIES[graph_size]
                    vehicles = [
                        [0, 3 * num_v, capa]
                    ]  # (type, num vehicles, vehicle capacity)
                    tw_start, tw_end = self._sample_tw(size=1,
                                                       graph_size=graph_size,
                                                       edges=edges,
                                                       service_duration=service_duration,
                                                       service_window=service_window,
                                                       time_factor=time_factor,
                                                       tw_expansion=tw_expansion,
                                                       normalize_tw=self.normalize_tws,
                                                       n_depots=1)
                    service_durations = np.broadcast_to(np.array([0] + [service_duration] * graph_size),
                                                        (graph_size + 1))

                    # print('tw_start[:5]', tw_start[:5])
                    # print('tw_end[:5]', tw_end[:5])
                    # print('service_durations[:5]', service_durations[:5])
                    # print('tw_start.shape', tw_start.shape)
                    # print('tw_end.shape', tw_end.shape)
                    # print('service_durations.shape', service_durations.shape)
                    # node_features --> created in generator
                    # nodes = self._create_nodes(size, graph_size, features=[coords, demands, a, b, service_durations])

                    # early_tw_constraint = start_tw_constraint_soft if early_tw_soft else start_tw_constraint_hard
                    # late_tw_constraint = end_tw_constraint_soft if late_tw_soft else end_tw_constraint_hard

                    # NOTE: separate TW at depot is modeled with >1 different depot nodes
                    #  stochastic / time dependent travel times can be done with edge transform
                    # tw = np.concatenate((tw_start, tw_end), axis=1)
                    # tw = (np.concatenate((
                    #     np.array([[0, 1]]),  # add depot tw start 0 and end 1
                    #     np.concatenate((tw_start[:, None], tw_end[:, None]), axis=-1)
                    # ), axis=0))
                    tw = np.expand_dims(np.concatenate((tw_start, tw_end)).transpose(), axis=0)

                    if self.normalize_tws:
                        tw = tw / (service_window/10)
                        # if solomon data then service duration is constant --> just have an array of shape n
                        service_durations = 0.1  # np.zeros(graph_size, dtype=np.float32)

                    # print('tw[:5]', tw[:5])
                    # print('tw.shape', tw.shape)

                else:
                    coords, c_types, d_types = self.sample_coords(n=graph_size + n_depots,
                                                                  resample_mixture_components=resample_mixture_components,
                                                                  **kwargs)
                    # Euclidean distance
                    dist_to_depot = dimacs_challenge_dist_fn_np(coords[1:], coords[
                        0])  # self.get_dist_to_depot(coords[1:], coords[0])
                    # print('dist_to_depot', dist_to_depot)
                    # distance.euclidean(coords[1:], coords[0])
                    time_to_depot = dist_to_depot / org_service_horizon
                    # print('time_to_depot', time_to_depot)
                    # cfg_stats
                    tw_start, tw_mask, num_tws = self._sample_tw_start(graph_size + n_depots, time_to_depot,
                                                                       org_service_horizon)
                    tw = self._sample_tw_end(
                        size=graph_size + n_depots,
                        tw_start=tw_start,
                        time_to_depot=time_to_depot,
                        tw_mask=tw_mask,
                        num_tws=num_tws,
                    )
            except AssertionError as ae:
                logger.debug(f"error while sampling. retrying... \n {ae}")
                i += 1
                continue
            feasible = True

        # print(time_to_depot)
        # print(tw[:, 1])
        # if np.any((tw[1:, 1] + time_to_depot + self.service_time) > 1.0):
        #     print("stop")
        assert not np.any((tw[1:, 1] + time_to_depot + service_time) > 1.0)

        return coords, weights, tw, service_durations, service_window

    def sample_coords(self,
                      n: int,
                      num_samples: Optional[int] = None,
                      resample_mixture_components: bool = True,
                      depot_type: Optional[str] = None,
                      customer_type: Optional[str] = None,
                      ensure_uniqueness: Optional[bool] = True,
                      **kwargs) -> Tuple[np.ndarray, List, List]:
        """
        Args:
            n: number of samples to draw (graph-size)
            num_samples: number of instance samples to draw
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance
            depot_type: Depot Position (uchoa only): C = central (500, 500), E = eccentric (0, 0), R = random
            customer_type: Customer Position (uchoa only): C = Clustered, RC = Random-Clustered (half half), R = Random
        Returns:
            coords: (n, n_dims)
        """
        if self.coords_sampling_dist == "uniform":
            coords = self._sample_unf_coords(n, **kwargs)
            c_types, d_types = ["uniform"] * 2
        elif self.coords_sampling_dist == "uchoa":
            # d_types --> {'C': 0, 'E': 1, 'R': 2}
            # c_types --> {'R': 0, 'C': 1, 'RC': 2}
            # print('customer_type', customer_type)
            customer_type = customer_type if customer_type is not None else self.customer_type
            depot_type = depot_type if depot_type is not None else self.depot_type
            coords, c_types, d_types, grid_s = self.sample_coords_uchoa(n - 1, num_samples=num_samples,
                                                                        depot_type=depot_type,
                                                                        customer_type=customer_type,
                                                                        ensure_uniqueness=ensure_uniqueness)
            coords = coords.squeeze() if num_samples is None else coords
            coords = coords / grid_s
        elif self.coords_sampling_dist == "explosion":
            # Following Bossek et al. (2019)
            # coords generated by mutating uniform distributed nodes by simulating a random explosion
            coords_unif = self._sample_unf_coords(n, **kwargs)  # sample uniformly distrib. customers
            coords = self._mutate_explosion(coords_unif, **kwargs)
            c_types, d_types = ["explosion"] * 2
            # pass
        elif self.coords_sampling_dist == "rotation":
            # Following Bossek et al. (2019)
            # coords generated by mutating uniform distributed nodes by simulating a random explosion
            coords_unif = self._sample_unf_coords(n, **kwargs)  # sample uniformly distrib. customers
            # print('kwargs', kwargs)
            coords = self._mutate_rotation(coords_unif, **kwargs)
            c_types, d_types = ["rotation"] * 2
            pass
        else:
            if self._sample_nc:
                self.nc = self.sample_rnd_int(*self._nc_params)
                self.resample_gm()
            elif resample_mixture_components:
                self.resample_gm()

            if self.coords_sampling_dist == "gm_unif_mixed":
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
                c_types = ["gm_unif"]
            else:
                # Sampling only gm coordinates
                n_per_c = math.ceil(n / self.nc)
                coords = self._sample_gm_coords(n_per_c, n, **kwargs)
                c_types = ["gm"]
            # depot stays uniform!
            coords[0] = self._sample_unf_coords(1, **kwargs)
            d_types = ["uniform"]
        return coords.astype(np.float32), c_types, d_types

    def sample_weights(self,
                       n: int,
                       k: int,
                       cap: Union[float, int],
                       max_cap_factor: Optional[float] = None,
                       coords: Optional[np.ndarray] = None,
                       demand_type: Optional[int] = None,
                       normalize: Optional[bool] = None,
                       feasibility_insurance: Optional[bool] = None
                       ) -> Tuple[ndarray, Union[float, int, ndarray], Union[str, int]]:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            coords: prev. sampled coordinates (needed for Uchoa type demands)
            demand_type: for uchoa data, which type of demand [0 (unitary) - 6 (many small, few large)]
            normalize: whether to normalize demands by capacity
            feasibility_insurance: overwrite self.try_ensure_feasibility (needed for large instance generation)
        Returns:
            weights: (n, )
        """
        n_wo_depot = n - 1
        try_ensure_feasibility_ = feasibility_insurance if feasibility_insurance is not None \
            else self.try_ensure_feasibility
        # print('try_ensure_feasibility_', try_ensure_feasibility_)
        # sample a weight for each point
        # print('self.weights_sampling_dist', self.weights_sampling_dist)
        if self.weights_sampling_dist in ["random_int", "random_k_variant"]:
            assert cap is not None, \
                f"weight sampling dist 'random_int' requires <cap> to be specified"

            if self.weights_sampling_dist == "random_int":
                # standard integer sampling adapted from Nazari et al. and Kool et al.
                # print('self.rnd', self.rnd)
                # random_
                weights = self.rnd.integers(1, 10, size=(n_wo_depot,))
                # print('weight raw', weights)
                normalizer = cap  # + 1
                # print('normalizer', normalizer)
                if try_ensure_feasibility_ and (np.sum(weights) > k and np.sum(weights) > k * cap):
                    need_k = int(np.ceil(weights.sum() / cap))
                    print('need_k', need_k)
                    print('(need_k / k)', (need_k / k))
                    normalizer *= ((need_k / k) + 0.1)
                # print('normalizer after try_ensure', normalizer)
                # print('normalize', normalize)
                if not normalize:
                    normalizer = 1

                type_w = "random_int"
            else:
                # weights = self.rnd.random_integers(1, (cap - 1) // 2, size=(n_wo_depot,))
                weights = self.rnd.integers(1, (cap - 1) // 2, size=(n_wo_depot,))
                # normalize weights by total max capacity of vehicles
                _div = max(2, self.sample_rnd_int(k // 4, k))
                if max_cap_factor is not None:
                    normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / _div
                else:
                    normalizer = np.ceil((weights.sum(axis=-1)) * 1.08) / _div
                print('normalizer2', normalizer)
                type_w = "random_k_variant"
        elif self.weights_sampling_dist in ["uniform", "gamma"]:
            # print('self.weights_sampling_dist', self.weights_sampling_dist)
            assert max_cap_factor is not None, \
                f"weight sampling dists 'uniform' and 'gamma' require <max_cap_factor> to be specified"
            if self.weights_sampling_dist == "uniform":
                weights = self._sample_uniform(n_wo_depot, *self.weights_sampling_params)
                type_w = "uniform"
            elif self.weights_sampling_dist == "gamma":
                weights = self._sample_gamma(n_wo_depot, *self.weights_sampling_params)
                type_w = "gamma"
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
            # print('normalizer3', normalizer)
        elif self.weights_sampling_dist == "uchoa":
            # sample uchoa type weights
            # print('uchoa coords in sampler', coords[:2])
            # print('demand_type', demand_type)
            demand_type = demand_type if demand_type is not None else self.demand_type
            weights_sc, capacity_orig, type_w = self.sample_weights_uchoa(coordinates=coords * GRID_SIZE,
                                                                          demand_type=demand_type,
                                                                          n=n_wo_depot)
            # print('weights_sc[:5]', weights_sc[:5])
            weights = np.squeeze(weights_sc)
            # print('weights.shape', weights.shape)
            # print('weights[:5]', weights[:5])
            cap = capacity_orig[0]
            type_w = type_w[0]
            # print('cap', cap)
            # print('type_w', type_w)
            normalizer = cap
            # print('normalizer3', normalizer)
        else:
            raise ValueError(f"unknown weight sampling distribution: {self.weights_sampling_dist}")

        # print('normalize', normalize)
        # print('normalizer', normalizer)
        if normalize:
            # print('normalizer4', normalizer)
            weights = weights / normalizer
            # print(np.clip(weights, None, 1.000000))
            # print('weights.any() > 1.0000000', weights.any() > 1.0000000)
            if try_ensure_feasibility_:
                # print(f"Make sure customer demands are not larger than vehicle capacity! Clipping demands to 1.0.")
                weights = np.clip(weights, None, 1.000000000)

        # print(f"np.sum(weights): {np.sum(weights)}")
        # print('k', k)
        # print('np.sum(weights) > k', np.sum(weights) > k)
        # only bigger than k here because demands are normalized and cap then set to 1.0
        if np.sum(weights) > k and np.sum(weights) > k * cap:
            if self.verbose:
                warn(f"generated instance is infeasible just by demands (sum(demands)={np.sum(weights)}) vs. "
                     f"total available vehicle capacity (k*cap={k * cap}) of specified number of vehicles.")
            if try_ensure_feasibility_:
                logger.info(f"generated instance is infeasible just by demands (sum(demands)={np.sum(weights)}) vs. "
                            f"total available vehicle capacity (k*cap={k * cap}) of specified number of vehicles.")
                raise RuntimeError

        weights = np.concatenate((np.array([0]), weights), axis=-1)  # add 0 weight for depot
        if normalize:
            weights = weights.astype(np.float32)
        elif self.weights_sampling_dist == "random_int" and not normalize:
            weights = weights.astype(np.int32)
        else:
            pass
        # print('cap', cap)

        return weights, cap, type_w

    # def sample_tw(self,
    #               n: int,
    #               resample_mixture_components: bool = True,
    #               **kwargs) -> np.ndarray:

    def sample_rnd_int(self, lower: int, upper: int) -> int:
        """Sample a single random integer between lower (inc) and upper (excl)."""
        # return self.rnd.random_integers(lower, upper, 1)[0]
        return self.rnd.integers(lower, upper, 1)[0]

    # from DPDP (Kool et al. 2020)
    def sample_coords_uchoa(self,
                            n: int,
                            num_samples: int = None,
                            depot_type: [str] = None,
                            customer_type: [str] = None,
                            int_locs: bool = True,
                            min_seeds: int = 3,
                            max_seeds: int = 8,
                            ensure_uniqueness: bool = True) -> Tuple[np.ndarray, List, List, int]:

        """
        Args:
            n: number of samples to draw
            num_samples: number of instances to sample --> batch size
            depot_type: which type of depot centrality (central, eccentric, random)
            customer_type: node distribution
            int_locs: whether coordindates should be integers
            min_seeds: min nr. of seeds to be sampled
            max_seeds: max nr. of seeds to be sampled
            ensure_uniqueness: unique coordinates (default True)

        Returns:
            coords: (n, n_dims)
        """

        if num_samples is None:
            num_samples = 1

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
        # depot_types = (np.random.rand(num_samples) * 3).astype(int)
        depot_types = (self.rnd.random(num_samples) * 3).astype(int)
        # (torch.rand(batch_size, device=device) * 3).int()
        if depot_type is not None:  # else mix
            # Central, Eccentric, Random
            codes = {'C': 0, 'E': 1, 'R': 2}
            depot_types[:] = codes[depot_type.upper()]

        # depot_locations = np.random.rand(num_samples, 2) * GRID_SIZE
        depot_locations = self.rnd.random((num_samples, 2)) * GRID_SIZE
        depot_locations[depot_types == 0] = GRID_SIZE / 2
        depot_locations[depot_types == 1] = 0

        # Customer position
        # 0 = random, 1 = clustered, 2 = random clustered 50/50
        # We always do this so we always pull the same number of random numbers
        # customer_types = (np.random.rand(num_samples) * 3).astype(int)
        # use random state for sampler:
        customer_types = (self.rnd.random(num_samples) * 3).astype(int)

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
        # print('customer_types', customer_types)
        # Sample number of seeds uniform (inclusive)
        # num_seeds = (np.random.rand(num_samples) * ((max_seeds - min_seeds) + 1)).astype(int) + min_seeds
        # use random state for sampler:
        num_seeds = (self.rnd.random(num_samples) * ((max_seeds - min_seeds) + 1)).astype(int) + min_seeds

        # We sample random and clustered coordinates for all instances, this way, the instances in the 'mix' case
        # Will be exactly the same as the instances in one of the tree 'not mixed' cases and we can reuse evaluations
        # rand_coords = np.random.rand(num_samples, n, 2) * GRID_SIZE
        # use random state for sampler:
        rand_coords = self.rnd.random((num_samples, n, 2)) * GRID_SIZE
        clustered_coords = self.generate_clustered_uchoa(num_seeds, n, max_seeds=max_seeds)

        # Clustered
        rand_coords[customer_types == 1] = clustered_coords[customer_types == 1]
        # Half clustered
        rand_coords[customer_types == 2, :(n // 2)] = clustered_coords[customer_types == 2, :(n // 2)]

        # stack depot coord and customer coords
        coords = np.stack([np.vstack((depot_locations[i].reshape(1, 2), rand_coords[i])) for i in range(num_samples)])
        coords = coords.astype(int) if int_locs else coords

        # ensure uniqueness
        if ensure_uniqueness:
            duplicates = True
            while duplicates:
                coords_torch = torch.from_numpy(coords[0])
                prev_nodes = [coords_torch[0]]
                prev_node_ids = [0]
                # print('CHECKING UNIQUENESS')
                # print('len(coords[0])', len(coords[0]))
                double_idxs = []
                for i in range(1, len(coords[0])):
                    if any([(coords_torch[i] == c_).all() for c_ in prev_nodes]):  # coor_[i] in prev_nodes:
                        idx_of_same = torch.where((torch.stack(prev_nodes) == coords_torch[i]).all(dim=1))[0]
                        # print(f'node coord {coords_torch[i]} with node ID {i} exists already for node {idx_of_same.item()}')
                        # print(f'currently  added node coords (ID {i}):', coords_torch[i])
                        # print(f'previously added node coords (ID {idx_of_same.item()}):', prev_nodes[idx_of_same])
                        double_idxs.append(i)
                    else:
                        prev_nodes.append(coords_torch[i])
                        prev_node_ids.append(i)
                if double_idxs:
                    if self.verbose:
                        print(f'there are duplicates for {double_idxs}')
                    for idx in double_idxs:
                        # resample for duplicates
                        # if customer_types[idx] != 1:
                        #     coords[idx] = self.rnd.random((1, n, 2)) * GRID_SIZE
                        # add small integer number to duplicate node coordinates
                        assert coords[0][idx][0].dtype == np.dtype('int64')
                        coords[0][idx][0] = coords[0][idx][self.rnd.integers(0, 2)] + self.rnd.integers(-10, 20) # 1
                        # print('coords[0][idx] now:', coords[0][idx])
                else:
                    # print('no duplicates found')
                    duplicates = False

        return coords, customer_types.tolist(), depot_types.tolist(), GRID_SIZE

    # from DPDP (Kool et al. 2020)
    # @staticmethod
    def sample_weights_uchoa(self,
                             coordinates: np.ndarray,
                             demand_type: int = None,
                             n: int = None) -> Tuple[np.ndarray, np.ndarray, List]:

        demand_type = int(demand_type) if demand_type is not None else demand_type
        try:
            batch_size, graph_size, _ = coordinates.shape
        except ValueError:
            graph_size, _ = coordinates.shape
            batch_size = 1
        if n is not None:
            graph_size = n
        # Demand distribution
        # 0 = unitary (1)
        # 1 = small values, large variance (1-10)
        # 2 = small values, small variance (5-10)
        # 3 = large values, large variance (1-100)
        # 4 = large values, large variance (50-100)
        # 5 = depending on quadrant top left and bottom right (even quadrants) (1-50), others (51-100) so add 50
        # 6 = many small, few large most (70 to 95 %, unclear so take uniform) from (1-10), rest from (50-100)
        lb = torch.tensor([1, 1, 5, 1, 50, 1, 1], dtype=torch.long, device="cpu")
        ub = torch.tensor([1, 10, 10, 100, 100, 50, 10], dtype=torch.long, device="cpu")
        if demand_type is not None:
            customer_positions = (torch.ones(batch_size, device="cpu") * demand_type).long()
        else:
            # customer_positions = (torch.rand(batch_size, device="cpu") * 7).long()
            customer_positions = (torch.from_numpy(self.rnd.random(batch_size)) * 7).long()
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
        rand_1 = torch.from_numpy(self.rnd.random((batch_size, graph_size)))
        rand_2 = torch.from_numpy(self.rnd.random((batch_size, graph_size)))
        rand_3 = torch.from_numpy(self.rnd.random(batch_size))
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
        # print("batchsize", batch_size)
        # print('self.rnd', self.rnd)
        r = sample_triangular(batch_size, 3, 6, 25, rnd_state=self.rnd, device="cpu")
        capacity = torch.ceil(r * demands.float().mean(-1)).long()
        # It can happen that demand is larger than capacity, so cap demand
        demand = torch.min(demands, capacity[:, None])

        # print('customer_positions.cpu().tolist()', customer_positions.cpu().tolist())
        # print('demand.cpu().numpy()', demand.cpu().numpy())

        return demand.cpu().numpy(), capacity.cpu().numpy(), customer_positions.cpu().tolist()

    # from DPDP (Kol et al. 2020)
    # @staticmethod
    def generate_clustered_uchoa(self, num_seeds, graph_size, max_seeds=None):
        if max_seeds is None:
            max_seeds = num_seeds.max()
            # .item()
        num_samples = num_seeds.shape[0]
        batch_rng = torch.arange(num_samples, dtype=torch.long, device="cpu")
        # batch_rng = np.arange(num_samples, dtype=int)
        seed_coords = (torch.from_numpy(self.rnd.random((num_samples, max_seeds, 2)) * GRID_SIZE))
        # print('seed_coords', seed_coords)
        # We make a little extra since some may fall off the grid
        n_try = graph_size * 2
        while True:
            # (torch.from_numpy(rnd.random((num_samples, n_try
            loc_seed_ind = (torch.from_numpy(self.rnd.random((num_samples, n_try)))
                            * num_seeds[:, None].astype(float)).long()
            # loc_seed_ind = (np.random.rand(num_samples, n_try) * num_seeds[:, None].astype(float)).astype(int)
            # print('batch_rng', batch_rng)
            # print('loc_seed_ind', loc_seed_ind)
            loc_seeds = seed_coords[batch_rng[:, None], loc_seed_ind]
            # alpha = torch.rand(num_samples, n_try) * 2 * math.pi
            alpha = torch.from_numpy(self.rnd.random((num_samples, n_try))) * 2 * math.pi
            # d = -40 * torch.rand(num_samples, n_try).log()
            d = -40 * torch.from_numpy(self.rnd.random((num_samples, n_try))).log()
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
        print('low,', low)
        print('high,', high)
        return self.rnd.uniform(size=size, low=low, high=high)

    def _mutate_explosion(self,
                          original_coords: np.ndarray,
                          # radius: float = 0.3,
                          low: Union[int, np.ndarray] = 0.0,
                          high: Union[int, np.ndarray] = 1.0,
                          **kwargs):
        # sample center of explosion
        v_c = np.array([0.5, 0.5])  # self.rnd.uniform(size=2, low=low, high=high) # hard coded to (0.5, 0.5) for SLI
        # get coords in radius of v_c
        coors_in_r2 = [coord for coord in original_coords if
                       np.linalg.norm(v_c - coord) <= self.radius]
        # nodes within explosion radius moved away from v_c following func from Bossek et al. (2019)/Zhou et al. (2023)
        new_coords = []
        for coord in original_coords:

            if (coord == list(coors_in_r2)).any():
                # direction of shift
                direction = (v_c - coord) / np.linalg.norm(v_c - coord)
                v_i = v_c + ((self.radius + self.s) * direction)
            else:
                v_i = coord
            new_coords.append(v_i)

        # Normalised [0,1]
        coords_exp = (np.array(new_coords) - np.min(np.array(new_coords))) / np.ptp(np.array(new_coords))
        # np.clip(np.array(new_coords), 0.0, 1.0)
        return coords_exp

    def _mutate_rotation(self,
                         original_coords: np.ndarray,
                         pm: float = 0.4,
                         angle_range: tuple = (0, 2),
                         **kwargs):
        # A subset \eqn{Q \subseteq P} of the points is selected and rotated
        # by a randomly sampled angle around its center.
        pm = pm if self.pm is None else self.pm
        angle_range = angle_range if self.angle_range is None else self.angle_range
        # print('type(angle_range)', type(angle_range))
        original_coords = self.rnd.uniform(size=(len(original_coords), 2))
        coords = original_coords.copy()
        idx_to_rotate = np.where(self.rnd.uniform(0, 1, len(original_coords)) < pm)[0]  # randomly sample rows
        # get rotation angle
        # angle = self.rnd.uniform(0, 360)
        # angle = self.rnd.uniform(1.2, 1.3 * np.pi)
        angle = self.rnd.uniform(angle_range[0], angle_range[1] * np.pi)
        # print('angle', angle)
        # rotation matrix to multiply selected coord rows
        rotation_mat = self._get_rotation_matrix(angle)
        # Rotation centered around the origin --> shift the rotated point cloud by a random number
        # random_shift = self.rnd.uniform(size=2)  # Generate 2 random numbers
        mutants = (rotation_mat @ original_coords[idx_to_rotate, :].T).T # + random_shift  # MatMul and shifting

        # Update the coordinates for the selected indices
        coords[idx_to_rotate, :] = mutants
        # return np.clip(np.array(coords), 0.0, 1.0)
        return np.clip((coords + 1) / 2, 0.0, 1.0)   # scale values from -1 - 1 to 0 and 1

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
        # print('coords.shape in create_edges', coords.shape)
        # print('coords in create_edges', coords)
        return np.linalg.norm(coords[:, :, None] - coords[:, None, :], ord=l_norm, axis=-1)[:, :, :, None]
    @staticmethod
    def _get_rotation_matrix(degree):
        alpha = degree # np.radians(degree)
        return np.array([[np.cos(alpha), -np.sin(alpha)],
                         [np.sin(alpha), np.cos(alpha)]])

    # from JAMPR v2.0 repo - generic time window sampler
    def _sample_tw(self,
                   size: int,
                   graph_size: int,
                   edges: np.ndarray,
                   service_duration: Union[int, float, np.ndarray],
                   service_window,
                   time_factor,
                   tw_expansion,
                   normalize_tw,
                   n_depots: int = 1):
        """Sample feasible time windows."""
        # print('service_duration', service_duration)
        # print('service_window', service_window)
        # print('time_factor', time_factor)
        # print('tw_expansion', tw_expansion)
        # TW start needs to be feasibly reachable directly from depot
        min_t = np.ceil(edges[:, 0,
                        1:] - service_duration + 1)  # TODO: adapt to multiple vehcs with different start depots --> n_start_depots & n_end_depots?
        # TW end needs to be early enough to perform service and return to depot until end of service window
        max_t = service_window - np.ceil(edges[:, 0, 1:] + 1)
        # print('max_t', max_t)
        # horizon allows for the feasibility of reaching nodes /
        # returning from nodes within the global tw (service window)
        horizon = np.concatenate((min_t, max_t), axis=-1)
        # print('horizon[:5]', horizon[:5])
        epsilon = np.maximum(np.abs(self.rnd.standard_normal([size, graph_size])), 1 / time_factor)

        # sample earliest start times a
        # a = self.rnd.randint(horizon[:, :, 0], horizon[:, :, 1])
        a = self.rnd.integers(horizon[:, :, 0], horizon[:, :, 1])
        # print('earliest start times a', a)
        # calculate latest start times b, which is
        # = a + service_time_expansion x normal random noise, all limited by the horizon
        b = np.minimum(a + tw_expansion * time_factor * epsilon, horizon[:, :, -1]).astype(int)
        # print('latest start times b', b)

        # add depot TWs and return
        return (
            # np.concatenate((np.array([0]*size)[:, None], a), axis=-1),
            np.concatenate((np.zeros((size, n_depots)), a), axis=-1),
            np.concatenate((
                np.broadcast_to(np.array([service_window])[:, None], (size, n_depots)),
                b
            ), axis=-1),
        )

    # for solomon type CVRPTW instances
    # from DIMACS_JAMPR repo
    def _sample_tw_start(self, size: int, time_to_depot: float, org_service_horizon: int,
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
        """sample start time of TW according to Solomon cfg specifications."""

        # get fraction of TW
        if self.tw_frac < 1.0:
            num_tws = int(np.ceil((size - 1) * self.tw_frac))
            tw_mask = np.zeros((size - 1), dtype=np.bool)
            tw_mask[self.rnd.choice(np.arange((size - 1)), size=num_tws, replace=False)] = 1
        else:
            num_tws = size - 1
            tw_mask = np.ones(size - 1, dtype=np.bool)

        # rejection sampling
        mean_tw_len = self.norm_summary.loc['mean', 'tw_len']
        # mean_tw_len = solomon_stats[0]["norm_summary"].loc['mean', 'tw_len']
        eps = 1. / org_service_horizon
        m = 10
        infeasible = True
        n = num_tws
        out = np.empty_like(time_to_depot)
        smp_idx = tw_mask.nonzero()[0]
        print('tw_mask', tw_mask)
        print('tw_mask.shape', tw_mask.shape)
        print('smp_idx', smp_idx)
        print('time_to_depot.shape', time_to_depot.shape)

        while infeasible:
            print('m', m)
            max_tw_start = 1. - np.repeat(time_to_depot[smp_idx] + self.service_time, m, axis=-1) - mean_tw_len / 2
            assert np.all(max_tw_start > 0)

            # if self.tw_start['dist'] == "gamma":
            if self.tw_start['dist'] == "gamma":
                smp = self.tw_start_sampler.rvs(size=m * n, random_state=self.rnd)
            elif self.tw_start['dist'] == "normal":
                smp = self.tw_start_sampler.rvs(size=m * n, random_state=self.rnd)
            elif self.tw_start['dist'] == "KDE":
                smp = self.tw_start_sampler.resample(size=m * n, seed=self.rnd)
            else:
                raise RuntimeError

            smp = smp.reshape(-1, m) + eps
            feasible = (smp > 0.0) & (smp <= max_tw_start.reshape(-1, m))
            has_feasible_val = np.any(feasible, axis=-1)
            # argmax returns idx of first True value if there is any, otherwise 0.
            first_feasible_idx = feasible[has_feasible_val].argmax(axis=-1)
            out[smp_idx[has_feasible_val]] = smp[has_feasible_val, first_feasible_idx]

            if np.all(has_feasible_val):
                infeasible = False
            else:
                no_feasible_val = ~has_feasible_val
                smp_idx = smp_idx[no_feasible_val]
                n = no_feasible_val.sum()
                m *= 2
            if m >= 320:  # 5
                # fall back to uniform sampling from valid interval
                s = eps
                print('s', s)
                e = max_tw_start
                print('e', e)
                print('n', n)
                # [:, None]
                out[smp_idx] = self.rnd.uniform(s, e, size=n)
                infeasible = False

        # set tw_start to 0 for nodes without TW
        out[~tw_mask] = 0

        return out, tw_mask, num_tws

    @staticmethod
    def get_dist_to_depot(i: Union[np.ndarray, float],
                          j: Union[np.ndarray, float],
                          scale: int = 100
                          ) -> np.ndarray:
        print('i', i)
        return np.floor(10 * np.sqrt(((scale * (i - j)) ** 2).sum(axis=-1))) / 10

    # from DIMACS_JAMPR repo
    def _sample_tw_end(self,
                       size: int,
                       tw_start: np.ndarray,
                       time_to_depot: float,
                       tw_mask: np.ndarray,
                       num_tws: int,
                       ) -> np.ndarray:
        """sample end time of TW according to cfg specifications."""
        # make sure sampled end is feasible by checking if
        # service time + time to return to depot is smaller than total service horizon
        eps = 1. / self.org_service_horizon
        t_delta = time_to_depot[tw_mask]
        inc_time = t_delta + self.service_time + eps
        smp_idx = tw_mask.nonzero()[0]
        out = np.empty_like(time_to_depot)

        if self.tw_len['dist'] == "const":
            assert np.all(inc_time + t_delta + self.tw_len_sampler < 1.0), \
                f"infeasible coordinates encountered"
            smp = self.tw_len_sampler  # all same constant value
            return_time = tw_start[tw_mask] + smp + inc_time
            infeasible = return_time >= 1.0
            if np.any(infeasible):
                inf_idx = smp_idx[infeasible]
                tw_start[inf_idx] = tw_start[inf_idx] - (return_time[infeasible] - 1 + eps)
                assert np.all(tw_start >= 0)

            out[tw_mask] = np.maximum(tw_start[tw_mask] + smp, t_delta + eps)

        else:
            # rejection sampling
            assert np.all(inc_time + t_delta < 1.0)
            m = 10
            infeasible = True
            n = num_tws

            while infeasible:
                if self.tw_len['dist'] == "gamma":
                    smp = self.tw_len_sampler.rvs(size=m * n, random_state=self.rnd)
                elif self.tw_len['dist'] == "normal":
                    smp = self.tw_len_sampler.rvs(size=m * n, random_state=self.rnd)
                elif self.tw_len['dist'] == "KDE":
                    smp = self.tw_len_sampler.resample(size=m * n, seed=self.rnd)
                else:
                    raise RuntimeError

                smp = smp.reshape(-1, m)
                # check feasibility
                # tw should be between tw_start + earliest possible arrival time from depot and
                # end of service horizon - time required to return to depot
                _tws = np.repeat(tw_start[smp_idx], m, axis=-1).reshape(-1, m)
                feasible = (
                        (_tws + np.repeat(t_delta, m, axis=-1).reshape(-1, m) < smp)
                        &
                        (_tws + np.repeat(inc_time, m, axis=-1).reshape(-1, m) + smp < 1.0)
                )
                has_feasible_val = np.any(feasible, axis=-1)
                # argmax returns idx of first True value if there is any, otherwise 0.
                first_feasible_idx = feasible[has_feasible_val].argmax(axis=-1)
                out[smp_idx[has_feasible_val]] = smp[has_feasible_val, first_feasible_idx]

                if np.all(has_feasible_val):
                    infeasible = False
                else:
                    no_feasible_val = ~has_feasible_val
                    smp_idx = smp_idx[no_feasible_val]
                    n = no_feasible_val.sum()
                    t_delta = t_delta[no_feasible_val]
                    inc_time = inc_time[no_feasible_val]
                    m *= 2
                if m >= 320:  # 5
                    # fall back to uniform sampling from valid interval
                    _tws = tw_start[smp_idx]
                    s = np.maximum(_tws, t_delta) + eps
                    e = 1. - inc_time

                    out[smp_idx] = self.rnd.uniform(s, e)
                    infeasible = False

        # add TW end as latest possible arrival time for all nodes without TW constraint
        out[~tw_mask] = 1.0 - time_to_depot[~tw_mask] - self.service_time - eps

        # assert np.all(out + time_to_depot + self.service_time < 1.0)
        return np.concatenate((
            np.array([[0, 1]]),  # add depot tw start 0 and end 1
            np.concatenate((tw_start[:, None], out[:, None]), axis=-1)
        ), axis=0)

    def get_normalizers(self) -> List:
        return self.normalizers


def dimacs_challenge_dist_fn_np(i: Union[np.ndarray, float],
                                j: Union[np.ndarray, float],
                                scale: int = 100,
                                ) -> np.ndarray:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j

    coords*100 since they were normalized to [0, 1]
    """
    return np.floor(10 * np.sqrt(((scale * (i - j)) ** 2).sum(axis=-1))) / 10
