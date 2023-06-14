# code mainly taken from l2o-meta (https://software.ismll.uni-hildesheim.de/jonas/l2o-meta/) + dimacs_jampr
from typing import NamedTuple, Union, Any, List, Optional
from torch import Tensor, LongTensor, BoolTensor
import numpy as np
import torch

__all__ = ["RPInstance", "TSPInstance", "CVRPInstance", "VRPTWInstance", "RPSolution"]


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"


class RPInstance(NamedTuple):
    """Typed Format Routing Problem Instance."""
    coords: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    graph_size: int
    depot_idx: List = [0]
    time_limit: float = None  # overall time limit for solving this instance (needed to calculate PI)

    # time_windows: Union[np.ndarray, torch.Tensor] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val


# inheritance in NamedTuple not possible
class TSPInstance(NamedTuple):
    """Typed Format Routing Problem Instance."""
    coords: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    graph_size: int
    depot_idx: List = [0]
    time_limit: float = None  # overall time limit for solving this instance (needed to calculate PI)
    BKS: float = None  # Best Known Solution (for this particular instance so far)
    instance_id: Optional[int] = None  # Test instances of a particular dataset have an ID - due to BKS registry

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val


class CVRPInstance(NamedTuple):
    """Typed Format Routing Problem Instance."""
    coords: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    graph_size: int
    vehicle_capacity: Union[Union[np.ndarray, torch.Tensor], float] = -1
    original_capacity: Union[int, np.ndarray, torch.Tensor] = None  # original_cap for NLNS or generally for Uchoa (
    # as they are non-uniform - array)
    max_num_vehicles: int = None  # soft upper limit
    depot_idx: List = [0]
    constraint_idx: List = [-1]
    time_limit: float = None  # overall time limit for solving this instance (needed to calculate PI)
    BKS: float = None  # Best Known Solution (for this particular instance so far)
    instance_id: Optional[int] = None  # Test instances of a particular dataset have an ID - due to BKS registry
    coords_dist: Union[int, str] = None  # For uchoa data it is an integer value, else 'unif' or 'gauss', 'mixed'
    depot_type: Union[int, str] = None  # For uchoa data it is an integer value, else 'unif' or 'gauss', 'mixed'
    demands_dist: Union[int, str] = None  # For uchoa -> integer value, else 'unif' or "random_int", "random_k_variant"
    original_locations: Union[np.ndarray, torch.Tensor] = None  # for NLNS
    type: str = None  # general "distribution" or type of data instance ('uniform', 'uchoa', 'dimacs', ...)
#     demand: Union[np.ndarray, torch.Tensor] = None # for NLNS


    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val

    def update(self, **kwargs):
        return self._replace(**kwargs)


class VRPTWInstance(NamedTuple):
    """Typed Format Routing Problem Instance."""
    coords: Union[np.ndarray, torch.Tensor]
    demands: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    tw: Union[np.ndarray, torch.Tensor]
    service_time: Union[np.ndarray, torch.Tensor, float]
    graph_size: int
    org_service_horizon: Union[float, int]
    max_vehicle_number: int
    vehicle_capacity: float = 1.0
    max_num_vehicles: int = 16
    service_horizon: float = 1.0
    depot_idx: List = [0]
    type: Union[int, str] = ""
    tw_frac: Union[float, str] = ""
    instance_id: Optional[int] = None
    time_limit: float = None  # overall time limit for solving this instance (needed to calculate PI)
    BKS: float = None  # Best Known Solution (for this particular instance so far) #TODO get from global registry?

    # constraint_idx: List = [-1]
    # vehicle_capacity: float = -1

    # @property
    # def node_features(self):
    #    return np.concatenate((self.coords, self.demands, self.tw), axis=-1)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val


class RPSolution(NamedTuple):
    """Typed wrapper for routing problem solutions."""
    solution: List[List]
    cost: float = None
    pi_score: float = None
    wrap_score: float = None
    num_vehicles: int = None
    run_time: float = None
    problem: str = None
    instance: Union[TSPInstance, CVRPInstance, VRPTWInstance] = None
    last_cost: Optional[float] = None  # previous instance cost that was PI-evaluated (for iterative PI computatn)
    last_runtime: Optional[float] = None  # previous instance runtime (for iterative PI computatn)
    running_costs: Optional[List] = None  # for PI and WRAP eval
    running_times: Optional[List] = None  # for PI and WRAP eval
    running_sols: Optional[List[List[List]]] = None  # for PI and WRAP eval
    instance_id: Optional[int] = None  # only needed for some models where solution lists are not sorted
    method_internal_cost: Optional[float] = None

    def update(self, **kwargs):
        return self._replace(**kwargs)
