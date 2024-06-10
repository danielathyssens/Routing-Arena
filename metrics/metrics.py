import warnings
from typing import Union, NamedTuple, Optional, Tuple, List, Dict
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt

from .utils import allign_times_costs

logger = logging.getLogger(__name__)
### CPU
# Quote: Currently, the top single thread CPU mark is 4,202, while mid-range desktop processors have marks around 2,000.
# https://www.cpubenchmark.net/singleThread.html#laptop-thread
CPU_BASE_REF_SINGLE = 2000  # equivalent to AMD Ryzen 7 PRO 3700U or Intel Xeon E3-1505M v5 @ 2.80GHz (single thread)
CPU_BASE_REF_MULTI = 8000  # roughly equivalent to AMD Ryzen 7 PRO 3700U or Intel Xeon E3-1505M v5 @ 2.80GHz 4C 8T
CPU_PASSMARK_BASE_SOL = 1612  # 1174 (uniform), 1612 (XML), 1612 (XE_1)
### GPU
# high-midrange passmarks of 750-670 (end of high-range GPU passmarks is around 2000)
# top high range GPU passmarks: 29000
# mean passmark of highest and low midrange beginning passmark is 15335
# https://www.videocardbenchmark.net/high_end_gpus.html
# GPU_BASE_REF = 5000
# GPU_BASE_REF = 18000  # reasoning: no one is using less than GeForce GTX 1080 Ti (PassMark: 18399)
# GPU_BASE_REF = 15000  # reasoning: no one is using less than GeForce GTX 1080 (PassMark: 15416)

######### NEW PASSMARK SETTING ############
GPU_3D_BASE_REF = 15000  # reasoning: no one is using less than GeForce GTX 1080 (PassMark: 15416)
GPU_2D_BASE_REF = 896  # reasoning: no one is using less than GeForce GTX 1080 (PassMark: 15416)
GPU_BASE_REF = round((CPU_BASE_REF_SINGLE + ((0.5 * GPU_3D_BASE_REF) + (0.5*GPU_2D_BASE_REF)))/2)
# GPU_BASE_REF = round(1 / (((1 / (CPU_BASE_REF_SINGLE * 0.396566187))
#                            + (1 / (GPU_2D_BASE_REF * 3.178718116)) + (1 / (GPU_3D_BASE_REF * 2.525195879))) / 3))

######## MACHINE PASSMARK SETTING #########
std_nr_threads = 1
alpha = 0.5
std_nr_gpus = 1
#MACHINE_BASE_REF_v1 = round((std_nr_threads * CPU_BASE_REF_SINGLE) + alpha * (std_nr_gpus * GPU_3D_BASE_REF))
MACHINE_BASE_REF_v1 = round((CPU_BASE_REF_SINGLE + ((0.5 * GPU_3D_BASE_REF) + (0.5 * GPU_2D_BASE_REF)))/2)

PASSMARK_VERSION = "sep"  # "v1"

# GPU_BASE_REF = 10823  # reasoning: no one is using less than Tesla T4 (PassMark: 10823)
eps = 0.05  # epsilon for runtime within time limit so if time limit 10, will accept 10.025


class Metrics:
    def __init__(self,
                 BKS: dict,
                 passMark: int,
                 TimeLimit_: Union[int, float],
                 passMark_cpu: int = None,
                 base_sol_results: dict = None,
                 scale_costs: int = None,
                 cpu: bool = True,
                 single_thread: bool = True,
                 is_cpu_search: bool = False,
                 verbose: bool = False,
                 passmark_v: str = PASSMARK_VERSION,
                 ):
        """
        Args:
            BKS: stores the best known solutions for a particular test dataset
            passMark: passMark for the performance of the CPU or GPU machine tested on
            passMark_cpu: additional CPU PassMark for GPU construction methods, where search performed on CPU only
            TimeLimit_: un-adjusted time limit for the solver to solve the instance
            base_runtimes: list of lists of the timepoints where base solver found solutions for instances
            base_costs: list of lists of tuples (cost, time) for found solution of the base solver across instances
            verbose: verbosity flag to print additional info and warnings
        """
        self.bks = BKS
        self.pass_mark = passMark
        self.pass_mark_cpu = passMark_cpu
        self.time_limit_ = TimeLimit_
        self.base_sol_results = base_sol_results
        # self.base_runtimes = [self.base_sol_result[str(inst)]]
        # self.base_costs = base_costs
        self.verbose = verbose
        self.scale_costs = scale_costs
        self.base_ref_construct = None
        if self.verbose:
            logger.info(f'self.pass_mark in Metrics: {self.pass_mark}')
            logger.info(f'self.pass_mark_cpu in Metrics: {self.pass_mark_cpu}')
        if passmark_v == "v1":
            print(f"MACHINE_BASE_REF_v1: {MACHINE_BASE_REF_v1}")
            self.base_ref = MACHINE_BASE_REF_v1
            if not cpu and is_cpu_search:
                self.base_ref_construct = MACHINE_BASE_REF_v1
        else:
            if cpu and single_thread:
                self.base_ref = CPU_BASE_REF_SINGLE
            elif cpu and not single_thread:
                self.base_ref = CPU_BASE_REF_MULTI
            elif not cpu and is_cpu_search:
                # means method constructs with GPU but search always done with CPU
                self.base_ref = CPU_BASE_REF_MULTI if not single_thread else CPU_BASE_REF_SINGLE
                self.base_ref_construct = GPU_BASE_REF
            else:
                assert cpu is False and is_cpu_search is False
                self.base_ref = GPU_BASE_REF
        # self.base_ref_base = CPU_BASE_REF_BASE_SOL
        self.base_pass_mark = CPU_PASSMARK_BASE_SOL
        self.base_ref_base = CPU_BASE_REF_SINGLE
        logger.info(f'Base Reference for this machine in metrics initialisation set to {self.base_ref}')
        if self.base_ref_construct is not None:
            logger.info(f"Base Reference for first GPU constructed method set to {self.base_ref_construct}")

    @staticmethod
    def RPI(c_t: float, c_t_base: float, c_opt: float):
        # print('c_t', c_t)
        # print('c_t_base', c_t_base)
        # print('c_opt', c_opt)
        if c_t is not None:  # only return when first c_t is obtained
            # print('min(c_t, c_t_base) - c_opt / c_t_base - c_opt ', (min(c_t, c_t_base) - c_opt) / (c_t_base - c_opt))
            # print('c_t - c_opt / c_t_base - c_opt', (c_t - c_opt) / (c_t_base - c_opt))
            try:
                return (min(c_t, c_t_base) - c_opt) / (c_t_base - c_opt)
            except ZeroDivisionError:
                warnings.warn(f"Zero Division in RPI Calculation, Base Solution = BKS for this Instance.")
                return 1.0
            # return (c_t - c_opt) / (c_t_base - c_opt)
        else:  # else if c_t is not yet obtained, while c_t_base exists for t return worst RPI score 1
            return 1.0

    def compute_wrap(self,
                     instance_id: str,
                     costs_: list,
                     runtimes_: list,
                     normed_inst_timelimit: Union[int, float]):
        """Computes the Weighted Relative Average Performance (WRAP).
           Can only be computed if BKS is defined and not None."""

        # check if no global time_limit given - else per instance time limit
        if self.time_limit_ is None:
            self.time_limit_ = normed_inst_timelimit * (self.pass_mark / self.base_ref)

        # get base sol runtimes and costs
        base_costs_, base_runtimes_ = self.base_sol_results[instance_id][0], self.base_sol_results[instance_id][1]
        c_opt = self.bks[instance_id][0]
        if self.verbose:
            logger.info(f"First 5 Base-runtimes {base_runtimes_[:5]}")
            logger.info(f"First 5 raw Model-runtimes {runtimes_[:5]}")
            logger.info(f"First 5 Base-costs {base_costs_[:5]}")
            logger.info(f"First 5 raw Model-costs {costs_[:5]}")
        if self.scale_costs is not None:
            if self.verbose:
                logger.info(f'SCALING COSTS with Metrics.scale_costs = {self.scale_costs}...')
                logger.info(f'c_opt (unscaled): {c_opt}')
            # SCALE ALL COST ENTITIES
            base_costs_ = [int(base_cost * 10000) for base_cost in base_costs_]
            costs_ = [int(cost * 10000) for cost in costs_]
            c_opt = int(c_opt * 10000)
            if self.verbose:
                logger.info(f'Scaled Base-costs now {base_costs_[:5]}')
                logger.info(f'Scaled Model-costs now {costs_[:5]}')
                logger.info(f'c_opt (scaled)', c_opt)

        # Normalize and round runtime according to PassMark
        # TimeLimit_normed = np.round(self.time_limit_ / (self.pass_mark / CPU_BASE_REF)) should be given to controller!
        # in order to match t_i (last) <= actual time limit
        if self.verbose:
            logger.info(f'Normalize and round runtime according to PassMark for WRAP Evaluation...')
        base_runtimes_normed = [base_runtime_ * (self.base_pass_mark / self.base_ref_base) for base_runtime_ in
                                base_runtimes_]
        base_runtimes_round = [int((t_i * 1000) + .5) / 1000.0 for t_i in base_runtimes_normed]
        if self.base_ref_construct is not None:
            # first GPU constructed value normalized separately
            runtimes_normed_first = [runtimes_[0] * (self.pass_mark / self.base_ref_construct)]
            # rest normalized with CPU pass_mark as GORT default search performed on cpu
            runtimes_normed_other = [runtime * (self.pass_mark_cpu / self.base_ref) for runtime in runtimes_[1:]]
            runtimes_normed = runtimes_normed_first + runtimes_normed_other
        else:
            runtimes_normed = [runtime * (self.pass_mark / self.base_ref) for runtime in runtimes_]
        runtimes_round = [int((t_i * 1000) + .5) / 1000.0 for t_i in runtimes_normed]
        # t_i = runtime * (self.pass_mark / self.base_ref)
        # t_i = int((t_i * 1000) + .5) / 1000.0
        if self.verbose:
            logger.info(f'Normalized-rounded Base-runtimes {base_runtimes_round[:5]}')
            logger.info(f'Normalized-rounded Model-runtimes {runtimes_round[:5]}')
        c_b_last = float('inf')
        base_runtimes, base_costs = [], []
        # get only the improvement list of sols from base solver (not the constant best sol list for all timepoints
        # where (potentially worse) sol is found in base solver
        for t_b, c_b in zip(base_runtimes_round, base_costs_):
            if t_b <= self.time_limit_ and c_b < c_b_last:
                base_runtimes.append(t_b)
                base_costs.append(c_b)
                c_b_last = c_b
        runtimes, costs = [], []
        # To-Assess Solver already has only improving sols list - so only cut at time limit for normalized run times
        last_t = 0.0
        for t, c in zip(runtimes_round, costs_):
            if t <= self.time_limit_:
                if t == last_t:
                    t += 0.0001
                runtimes.append(t)
                costs.append(c)
            last_t = t
        if self.verbose:
            logger.info(f"Final 'to-assess' Model-runtimes (first 5) {runtimes[:5]}")
            logger.info(f"Final Base-runtimes (first 5) {base_runtimes[:5]}")
            logger.info(f"Final 'to-assess' Model-costs (first 5) {costs[:5]}")
            logger.info(f"Final Base-costs (first 5) {base_costs[:5]}")
        # check if there's sol found in Time Limit
        if not costs:
            warnings.warn(f"No Solution for instance {instance_id} found in time limit. "
                          f"Original runtime is {runtimes_[0]}. "
                          f"Global Time Limit is {self.time_limit_}"
                          f"Setting WRAP to 1.")
            final_wrap = 1

        else:
            # adjust time scale
            t_merge = runtimes.copy()
            t_merge.extend(base_runtimes)
            t_merge_unique = list(set(t_merge))
            t_merge_unique.sort()  # sorted list of all timepoints in self.base_runtimes and runtimes
            if self.verbose:
                logger.info(f'Merged Time list for WRAP evaluation (first 5): {t_merge_unique[:5]}')
            # adjust costs according to merged time scale
            # - if improvement in one cost but not in the other cost for not improving method stays at c_t-1
            c_base_adj, c_adj, t_merge_adj = allign_times_costs(t_merge_unique, runtimes, base_runtimes, costs,
                                                                base_costs)

            if self.verbose:
                logger.info(f'Adjusted Model-Costs list acc. to merged times (first 5): {c_adj[:5]}')
            wrap = 0
            t_n_1 = 0
            RPI_t_0 = 1.0
            T_N = self.time_limit_
            solver_c_n_1, base_c_n_1 = None, None
            for n, t_n in enumerate(runtimes):
                t_n_idx = t_merge_adj.index(t_n)
                solver_c = c_adj[t_n_idx]
                base_c = c_base_adj[t_n_idx]
                RPI_t_n_1 = self.RPI(solver_c_n_1, base_c_n_1, c_opt)
                wrap_n = RPI_t_n_1 * (t_n - t_n_1)
                wrap += wrap_n
                t_n_1 = t_n
                solver_c_n_1 = solver_c
                base_c_n_1 = base_c
            # calc final RPI(t_n+1) for t_N+1=T:
            RPI_T = self.RPI(solver_c_n_1, base_c_n_1, c_opt)
            wrap += RPI_T * (T_N - t_n_1)
            final_wrap = (1 / self.time_limit_) * wrap
        return final_wrap

    def compute_pi(self,
                   instance_id: str,
                   costs: list,
                   runtimes: list,
                   normed_inst_timelimit):
        """Computes the Primal Integral (PI) score based on the formulation used for the DIMACS Challenge 2021.
           Can only be computed if BKS is defined and not None."""

        # check if no global time_limit given - else per instance time limit
        if self.time_limit_ is None:
            self.time_limit_ = normed_inst_timelimit * (self.pass_mark / self.base_ref)

        # get best known solution for this instance
        bks = self.bks[instance_id][0]
        assert bks is not None, f" PI and WRAP Evaluation cannot be performed without a Best Known Solution"
        # get number of solutions found
        assert len(costs) == len(runtimes), f"Found more solutions than runtimes"

        if self.verbose:
            print(f"Normalised Time Limit for Solver based on Passmark of used Machine "
                  f"is {np.round(self.time_limit_ / (self.pass_mark / self.base_ref))}")

        # Normalize Time Limit according to PassMark
        # TimeLimit_normed = self.time_limit_
        # TimeLimit_normed = np.round(self.time_limit_ / (self.pass_mark / CPU_BASE_REF)) should be given to controller!
        # in order to match t_i (last) <= actual time limit !!!!!!!!!!!!!

        # set min. threshold for cost to v(0) = BKS * 1.1
        base_cost = bks * 1.1

        if self.scale_costs is not None:
            # SCALE ALL COST ENTITIES
            if self.verbose:
                logger.info(f'SCALING COSTS with Metrics.scale_costs = {self.scale_costs}...')
                logger.info(f'BKS (unscaled): {bks}')
            base_cost = int(base_cost * 10000)
            costs = [int(cost * 10000) for cost in costs]
            bks = int(bks * 10000)
            if self.verbose:
                logger.info(f'Scaled Base-costs now {base_cost}')
                logger.info(f'Scaled Model-costs now {costs[:5]}')
                logger.info(f'BKS (scaled)', bks)


        # at initial time step t_0 = 0, v_0 = BKS * 1.1, PrimalIntegral = 0 and final flag is False
        cost_last, time_last, primalIntegral, final_primalIntegral = base_cost, 0, 0, 0
        # flag for GPU constructed initial solution
        is_init_gpu = True if self.base_ref_construct is not None else False
        passmark_init = self.pass_mark if self.base_ref_construct is not None else None
        pass_mark = self.pass_mark_cpu if self.base_ref_construct else self.pass_mark
        if self.verbose:
            logger.info(f'Calculating PI for time budget {self.time_limit_}...')
        for i, (cost, runtime) in enumerate(zip(costs, runtimes)):
            if self.verbose:
                logger.info(f'Current Time: {runtime}, Base-Cost: {base_cost}, Cost: {cost} ')
            # Normalize and round runtime according to PassMark
            # t_i = runtime * (self.pass_mark / CPU_BASE_REF)
            t_i = runtime * (pass_mark / self.base_ref) if not is_init_gpu \
                else runtime * (passmark_init / self.base_ref_construct)
            t_i = int((t_i * 1000) + .5) / 1000.0
            # only evaluate PI if found cost is better than base cost in time limit
            if not ((cost_last - cost) < 0.1) and t_i <= self.time_limit_ + eps:
                # Normalize and round runtime according to PassMark
                # t_i = runtime * (self.pass_mark / CPU_BASE_REF)
                # t_i = int((t_i * 1000) + .5) / 1000.0
                # cost(i - 1) * (time(i) - time(i - 1)) / BKS * T
                primalIntegral += (cost_last * (t_i - time_last) / (self.time_limit_ * bks))
                # only reset time_last = t_i if evaluated already!!! (else never take into account t_0-t_i time)
                time_last = t_i
                # move index
                cost_last = cost

            elif (base_cost - cost) < 0.1:
                # move to next cost (without moving time to take into account t_i-0 of first valid found sol (i))
                # cost_last = cost
                pass

            elif not t_i <= self.time_limit_ + eps:  # if time limit surpassed - take last valid found cost and break
                logger.info(f"Cost {cost} for instance {instance_id} not found in time limit. "
                            f"Original runtime is {runtime}. "
                            f"Normalized runtime is {t_i}. "
                            f"Global Time Limit is {self.time_limit_}")
                logger.info(f"Working with last cost found in time limit: "
                            f"Cost_last {cost_last} found in {time_last} and break.")
                break

            if self.verbose:
                logger.info(f"**** PI Calculation INFO ****")
                logger.info(f"running PrimalIntegral: {primalIntegral}")

            is_init_gpu = False  # re-set flag for init. solution GPU time normalization

        # calculate final Primal Integral if solution better than base cost found in time limit - else set to 10
        if primalIntegral != 0:
            #     // FINAL Primal Integral is calculated based on DIMACS Challenge Rules found in
            #        # http://dimacs.rutgers.edu/programs/challenge/vrp/cvrp/
            #     // add cost(n)*(T - t(n))/BKS*T
            #     // and normalize: 100 * (PI - 1)
            primalIntegral += (cost_last * (self.time_limit_ - time_last) / (self.time_limit_ * bks))
            if self.verbose:
                logger.info(f"PrimalIntegral before norm: {primalIntegral}")
            primalIntegral -= 1
            final_primalIntegral = primalIntegral * 100
            if final_primalIntegral < 0:
                warnings.warn(f'Negative PI found for this instance - BUG! or New Best Known Solution Found')
        else:
            final_primalIntegral = 10
            if self.verbose:
                logger.info(f"No solution better than base cost found in Time limit {self.time_limit_} for "
                            f"instance {instance_id}. "
                            f"Normalised Time Limit for Solver based on Passmark of current Machine "
                            f"is {np.round(self.time_limit_ / (self.pass_mark / self.base_ref))}"
                            f"Primal Integral will be set to {final_primalIntegral}.")

        if self.verbose:
            # logger.info(f"(final_cost, final_time): ({cost},{runtime})")
            logger.info(f"BKS: {bks}, base cost to beat: {base_cost}, original Time Limit: {self.time_limit_}")
            logger.info(f"First found cost: {costs[0]}, Best found cost: {min(costs)}")
            logger.info(f"First found time: {runtimes[0]} Last found time: {runtimes[-1]}")
            logger.info(f"Final PrimalIntegral: {final_primalIntegral}")

        return final_primalIntegral
