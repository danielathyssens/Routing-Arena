import warnings
from typing import Union, NamedTuple, Optional, Tuple, List, Dict
import numpy as np


def allign_times_costs(sorted_times, r_times, base_r_times, c_t, c_t_base):
    last_c, last_c_base, curr_c, curr_c_base = None, None, None, None
    # last_c, last_c_base, curr_c, curr_c_base = None, c_t_base[0] * 1.2, None, None
    c_adj_new, c_base_adj_new = None, None
    idx_base, idx = 0, 0
    c_adj, c_base_adj = [], []
    # print('c_t_base', c_t_base)
    # print('base_r_times', base_r_times)
    # print('c_t', c_t)
    # print('r_times', r_times)
    # print('LEN SORTED ITEMS:', len('sorted_times'))
    for i, t in enumerate(sorted_times):
        # print('i', i)
        # print('idx_base', idx_base)
        # print('last_c_base', last_c_base)
        # print('idx', idx)
        # print('last_c', last_c)
        if t == base_r_times[idx_base]:
            #print(f"t [{t}] == base_r_times[idx_base] [{base_r_times[idx_base]}]")
            # c_t_base[idx_base] == last_c_base only if
            curr_c_base = c_t_base[idx_base]
            if curr_c_base is not None and curr_c_base != last_c_base:
                #print('updating base IDX')
                idx_base = idx_base + 1
                # check if idx then surpassed length of found solutions
                if idx_base == len(c_t_base):
                    #print('hit end of found solutions list')
                    idx_base = len(c_t_base) - 1
                #print('base IDX NOW: ', idx_base)
        else:
            #print(f'Keep OLD idx_base={idx_base} and reset curr_c_base to last_c_base={last_c_base}')
            curr_c_base = last_c_base
        if t == r_times[idx]:
            #print(f"t [{t}] == r_times[idx] [{r_times[idx]}]")
            curr_c = c_t[idx]
            if curr_c is not None and curr_c != last_c:
                #print('updating IDX')
                idx = idx + 1
                # check if idx then surpassed length of found solutions
                if idx == len(c_t):
                    #print('hit end of found solutions list')
                    idx = len(c_t) - 1
                #print('IDX NOW: ', idx)
        else:
            #print(f'Keep OLD idx={idx} and reset curr_c to last_c={last_c}')
            curr_c = last_c
        c_base_adj.append(curr_c_base)
        c_adj.append(curr_c)
        last_c, last_c_base = curr_c, curr_c_base
        #print('r_all_unique', sorted_times)
        #print('c_base_adj', c_base_adj)
        #print('c_adj', c_adj)
    # TODO: what if c_adj or c_base_adj starts with None values? -> for now cut entries with None in both lists
    # print('c_adj', c_adj)
    # print('len(c_adj)', len(c_adj))
    # print('c_base_adj', c_base_adj)
    # print('len(c_base_adj)', len(c_base_adj))
    # print('len(sorted_times)', len(sorted_times))
    # print('c_adj', c_adj)
    # print('c_base_adj', c_base_adj)
    # print('sorted_times', sorted_times)
    # c_adj_new, c_base_adj_new, sorted_times_new = del_None_entries(c_adj, c_base_adj, sorted_times)
    # check that c_base_adj doesn't start with None:
    if None in c_base_adj:
        max_cost_base = max(filter(lambda x: x is not None, c_base_adj))
        # print('max_cost_base', max_cost_base)
        c_base_adj_new = [max_cost_base if v is None else v for v in c_base_adj]
        # print('c_base_adj_new', c_base_adj_new)
            # list(map(lambda x: x.replace(None, max_cost_base), c_base_adj))
    else:
        c_base_adj_new = c_base_adj
    c_adj_new, sorted_times_new = c_adj, sorted_times
    # print('len(c_adj_new)', len(c_adj_new))
    # print('len(c_base_adj_new)', len(c_base_adj_new))
    # print('len(sorted_times_new)', len(sorted_times_new))
    return c_base_adj_new, c_adj_new, sorted_times_new


def del_None_entries(c, c_b, sort_times):
    has_None_base = True if None in c_b else False
    # print('has_None_base', has_None_base)
    has_None = True if None in c else False
    # print('has_None', has_None)
    if has_None_base and not has_None:
        del_cb = frozenset(set([i for i, e in enumerate(c_b) if e is None]))
        # print("cutting off b/c of c_b")
        c_b_new = [v for i, v in enumerate(c_b) if i not in del_cb]
        c_new = [v for i, v in enumerate(c) if i not in del_cb]
        sort_times_new = [v for i, v in enumerate(sort_times) if i not in del_cb]
    elif has_None and not has_None_base:
        del_c_ = frozenset(set([i for i, e in enumerate(c) if e is None]))
        # print("cutting off b/c of c")
        c_b_new = [v for i, v in enumerate(c_b) if i not in del_c_]
        c_new = [v for i, v in enumerate(c) if i not in del_c_]
        sort_times_new = [v for i, v in enumerate(sort_times) if i not in del_c_]
    elif has_None and has_None_base:
        del_c = frozenset(set([i for i, e in enumerate(c) if e is None]))
        # print('del_c', del_c)
        del_c_b = frozenset(set([i for i, e in enumerate(c_b) if e is None]))
        # print('del_c_b', del_c_b)
        c_b_new = [v for i, v in enumerate(c_b) if i not in del_c and i not in del_c_b]
        c_new = [v for i, v in enumerate(c) if i not in del_c and i not in del_c_b]
        sort_times_new = [v for i, v in enumerate(sort_times) if i not in del_c and i not in del_c_b]
    else:
        c_b_new = c_b
        c_new = c
        sort_times_new = sort_times
    return c_new, c_b_new, sort_times_new

# t_merge = self.base_runtimes.copy()
# for i in range(len(runtimes) - 1):
#    for j in range(len(self.base_runtimes) - 1):
#        if runtimes[i] > self.base_runtimes[j] and runtimes[i] < self.base_runtimes[j + 1]:
#            t_merge.insert(j + 1, runtimes[i])
