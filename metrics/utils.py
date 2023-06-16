import warnings
from typing import Union, NamedTuple, Optional, Tuple, List, Dict
import numpy as np


def allign_times_costs(sorted_times, r_times, base_r_times, c_t, c_t_base):
    last_c, last_c_base, curr_c, curr_c_base = None, None, None, None
    c_adj_new, c_base_adj_new = None, None
    idx_base, idx = 0, 0
    c_adj, c_base_adj = [], []
    for i, t in enumerate(sorted_times):
        if t == base_r_times[idx_base]:
            # c_t_base[idx_base] == last_c_base only if
            curr_c_base = c_t_base[idx_base]
            if curr_c_base is not None and curr_c_base != last_c_base:
                idx_base = idx_base + 1
                # check if idx then surpassed length of found solutions
                if idx_base == len(c_t_base):
                    idx_base = len(c_t_base) - 1
        else:
            curr_c_base = last_c_base
        if t == r_times[idx]:
            curr_c = c_t[idx]
            if curr_c is not None and curr_c != last_c:
                idx = idx + 1
                # check if idx then surpassed length of found solutions
                if idx == len(c_t):
                    idx = len(c_t) - 1
        else:
            curr_c = last_c
        c_base_adj.append(curr_c_base)
        c_adj.append(curr_c)
        last_c, last_c_base = curr_c, curr_c_base
    # check that c_base_adj doesn't start with None:
    if None in c_base_adj:
        max_cost_base = max(filter(lambda x: x is not None, c_base_adj))
        c_base_adj_new = [max_cost_base if v is None else v for v in c_base_adj]
    else:
        c_base_adj_new = c_base_adj
    c_adj_new, sorted_times_new = c_adj, sorted_times
    return c_base_adj_new, c_adj_new, sorted_times_new
