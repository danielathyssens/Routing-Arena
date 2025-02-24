import warnings

import numpy as np
import torch
import logging
import os
import psutil
import cpuinfo
from warnings import warn
import math
import itertools
import matplotlib.pyplot as plt
from formats import RPSolution, CVRPInstance
from typing import Optional, Dict, Union, List
import omegaconf
import shutil

from .general_utils import get_machine_info

logger = logging.getLogger(__name__)

CPU_BASE_REF_SINGLE = 2000  # equivalent to AMD Ryzen 7 PRO 3700U or Intel Xeon E3-1505M v5 @ 2.80GHz (single thread)
CPU_BASE_REF_MULTI = 8000  # roughly equivalent to AMD Ryzen 7 PRO 3700U or Intel Xeon E3-1505M v5 @ 2.80GHz 4C 8T
# GPU_BASE_REF = 18000  # reasoning: no one is using less than GeForce GTX 1080 Ti (PassMark: 18399)
GPU_3D_BASE_REF = 15473  # reasoning: no one is using less than GeForce GTX 1080 (PassMark: 15416)
GPU_2D_BASE_REF = 896  # reasoning: no one is using less than GeForce GTX 1080 (PassMark: 15416)
# GPU_BASE_REF = round(1 / (((1 / (CPU_BASE_REF_SINGLE * 0.396566187))
#                            + (1 / (GPU_2D_BASE_REF * 3.178718116)) + (1 / (GPU_3D_BASE_REF * 2.525195879))) / 3))
GPU_BASE_REF = round((CPU_BASE_REF_SINGLE + ((0.5 * GPU_3D_BASE_REF) + (0.5 * GPU_2D_BASE_REF))) / 2)

# round((cpu_performance + ((0.5 * threeD_Mark) + (0.5 * twoD_Mark))) / 2)


######## MACHINE PASSMARK SETTING #########
std_nr_threads = 1
alpha = 0.5
std_nr_gpus = 1
# MACHINE_BASE_REF_v1 = round((std_nr_threads * CPU_BASE_REF_SINGLE) + alpha * (std_nr_gpus * GPU_3D_BASE_REF))
MACHINE_BASE_REF_v1 = round((CPU_BASE_REF_SINGLE + ((0.5 * GPU_3D_BASE_REF) + (0.5 * GPU_2D_BASE_REF))) / 2)

PASSMARK_VERSION = "sep"

SET_TYPES = ["X", "XE", "XML"]

# DEVICE, NUMBER_THREADS_USED = None, None

GPU_MACHINES = {
    # "GPU_NAME" = [GPU-Average G3D Mark, GPU-Average G2D Mark]
    'NVIDIA GeForce GTX 1080': [15473, 896],
    'NVIDIA GeForce RTX 4090': [39487, 1325],
    'NVIDIA GeForce RTX 3060': [17177, 979],
    'NVIDIA GeForce RTX 3060 Ti': [20608, 1002],
    'NVIDIA A40': [9170, 502],  # 2x on node gpu-110
    'NVIDIA GeForce RTX 3090': [26962, 1047],  # 2x on node gpu-110 and 2x on gpu-100 and 3x on gpu-101
    'NVIDIA RTX A4000': [19097, 964],  # 8x on node gpu-120
    'NVIDIA GeForce GTX 950M': [2600, 217],  # Home old
    'Apple M2 10-Core GPU': [24880, 1000],  # Home
    'Apple M1 10-Core GPU': [24880, 1000],  # Wilson
    'NVIDIA GeForce RTX 2080 Ti': [21893, 939],  # 4x each on node gpu-020 - gpu-023, 3x on node gpu-022 (Prt. GPU)
    'NVIDIA GeForce GTX 1080 Ti': [18508, 943],  # 1x on gpu-000 - gpu-003 and 2x each on gpu-010 - gpu-019  (Prt. GPU)

}

CPU_MACHINES = {
    # # "CPU_NAME" = [cpuMark, cpuMark_single_thread, total_cores (as should be), total_threads (as should be)]
    'Intel(R) Core(TM) i5-6300HQ CPU @ 2.30GHz': [4692, 1790, 4, 4],  # Home
    'Apple M2': [15383, 3932, 8, 8],  # Home
    'Apple M1': [15383, 3932, 8, 8],  # Wilson
    'Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz': [11980, 2714, 6, 12],  # Work-OLD
    '13th Gen Intel(R) Core(TM) i7-1355U': [15361, 3603, 10, 12],  # Work
    'AMD EPYC 7402 24-Core Processor': [42245, 1947, 24, 48],  # on node gpu-100, gpu-101
    'AMD EPYC 7543 32-Core Processor': [56562, 2563, 32, 64],  # on node gpu-110
    'AMD EPYC 7713P 64-Core Processor': [83439, 2742, 64, 128],  # on node gpu-120
    'Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz': [9251, 1635, 16, 32],  # on node gpu-020 - gpu-023 (MARKS for 8C,16Th)
    'Intel(R) Xeon(R) CPU E5-1660 v4 @ 3.20GHz': [13500, 2187, 8, 16],  # on node gpu-010 - gpu-019
    'Intel(R) Xeon(R) CPU E5-1620 v4 @ 3.50GHz': [7416, 2236, 4, 8],  # on node gpu-000 - gpu-003
    'Intel(R) Xeon(R) CPU E5620 @ 2.40GHz': [6442, 1077, 8, 16],  # on node node-010, node-011 (Prt. CPU0)
    'Intel(R) Xeon(R) CPU           E5620  @ 2.40GHz': [6442, 1077, 8, 16],  # on node node-010, node-011 (Prt. CPU0)
    'Intel(R) Xeon(R) CPU E5645 @ 2.40GHz': [9148, 1174, 12, 24],  # on node node-100 - node-103 (Prt. CPU1)
    'Intel(R) Xeon(R) CPU           E5645  @ 2.40GHz': [9148, 1174, 12, 24],  # on node node-100 - node-103 (Prt. CPU1)
    'Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz': [20036, 1612, 20, 40],  # on node node-200 - node-219 (Prt. CPU2)
    '11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz': [20610, 3131, 8, 16],  # tim laptop
    'AMD EPYC 9334 32-Core Processor': [64841, 2367, 32, 64]   # node gpu-200
}

NORMED_BENCHMARKS = ['cvrp20_test_seed1234.pkl',
                     'cvrp50_test_seed1234.pkl',
                     'cvrp100_test_seed1234.pkl',
                     'val_seed4321_size512.pkl',
                     'val_seed123_size512.pt',
                     'val_seed123_size512.pkl',
                     'E_R_6_seed123_size512.pt',
                     'val_seed123_size4321.pkl',
                     'val_seed123_size4321.pt',
                     'val_E_size2000.pkl',
                     'val_R_size2000.pkl']

XE_DIMS = {'XE_1': 100, 'XE_2': 124, 'XE_3': 128, 'XE_4': 161, 'XE_5': 180, 'XE_6': 185, 'XE_7': 199,
           'XE_8': 203, 'XE_9': 213, 'XE_10': 218, 'XE_11': 236, 'XE_12': 241, 'XE_13': 269, 'XE_14': 274,
           'XE_15': 279, 'XE_16': 293, 'XE_17': 297}


# X_DIMS =

def get_time_limit(cfg):
    print('cfg.test_cfg.time_limit', cfg.test_cfg.time_limit)
    if cfg.test_cfg.time_limit is not None and isinstance(cfg.test_cfg.time_limit, int):
        return cfg.test_cfg.time_limit
    else:
        if cfg.test_cfg.time_limit is None:
            logger.info(f"No explicit time_limit set for evaluation. Defaulting to implicit Time Budget.")
        if cfg.graph_size is not None:
            return get_budget_per_size(cfg.graph_size)
        else:
            if 'XE_type' in list(cfg.keys()):
                assert cfg.XE_type in XE_DIMS.keys()
                problem_size = XE_DIMS[cfg.XE_type]
                return get_budget_per_size(problem_size)
            else:
                logger.info(f"Different per-instance Time Budgets. Size-dependent Time Limits are loaded "
                            f"with instances in Dataset...")


def get_budget_per_size(problem_size: int, round_up: bool = True):
    # TL = (240/100)*n (as in HGS-CVRP paper)
    if round_up:
        return int(np.ceil((240 / 100) * problem_size))
    else:
        return (240 / 100) * problem_size


def eval_inference(run, nr_runs, sols_, ds_class, log_path, acronym, test_cfg, debug):
    results, per_instance_summaries, new_BKS_for, stats = [], [], [], None
    for solution in sols_:
        # print('solution', solution)
        updated_sol, summary_, new_bks_ = ds_class.eval_solution(solution=solution,
                                                                 model_name=acronym,
                                                                 eval_mode=test_cfg.eval_type,
                                                                 save_trajectory=test_cfg.save_trajectory,
                                                                 save_trajectory_for=test_cfg.save_traj_for)
        results.append(updated_sol)
        per_instance_summaries.append(summary_)
        if new_bks_ is not None:
            new_BKS_for.append(new_bks_)

    try:
        test_data_path = test_cfg.data_file_path
    except omegaconf.errors.ConfigAttributeError:
        test_data_path = ds_class.store_path
    if new_BKS_for:
        update_bks(results, new_BKS_for, test_data_path, ds_class, acronym)   # else None
    updates = None  # update_ranking()

    stats = get_stats(sols=results, logger_=logger, model_name=acronym,
                      debug_flag=debug, ranking_updates=updates, N_runs=nr_runs, run_i=run)

    # try:
    cpu_name = get_cpu_specs()[0]
    cpu_count = NUMBER_THREADS_USED
    gpu_name = torch.cuda.get_device_name() if (torch.cuda.is_available() and DEVICE == 'cuda') else str(None)
    gpu_count = torch.cuda.device_count() if (torch.cuda.is_available() and DEVICE == 'cuda') else None

    if test_cfg.save_for_analysis:
        acronym_analysis = acronym.replace("_", "-")
        file_name = "run_" + str(run) + "_results_" + acronym_analysis + ".pkl"
        if test_cfg.out_name.split("_")[1] not in SET_TYPES:
            set_name = test_cfg.out_name
            saved_res_path = os.path.join(test_cfg.saved_res_dir, set_name, "TL_" + str(test_cfg.time_limit))
        else:
            try:
                set_type = test_cfg.data_file_path.split("/")[-2]
                set_name = test_cfg.data_file_path.split("/")[-1].split(".")[0]
            except omegaconf.errors.ConfigAttributeError:
                set_type = ds_class.store_path.split("/")[-2]
                set_name = ds_class.store_path.split("/")[-1].split(".")[0]
            saved_res_path = os.path.join(test_cfg.saved_res_dir, set_name, "TL_" + str(test_cfg.time_limit)) \
                if set_type != "XE" else \
                os.path.join(test_cfg.saved_res_dir, set_type, set_name, "TL_" + str(test_cfg.time_limit))
        # make analysis dir for this set if it does not yet exist
        os.makedirs(saved_res_path, exist_ok=True)
        logger.info(f"Storing Results of run {run} for analysis in {saved_res_path}")
        save_results(
            result={
                "solutions": results,
                "summary": stats,
                "machine": {'CPU': cpu_name + ":" + str(cpu_count), 'GPU': gpu_name + ":" + str(gpu_count)}
            },
            log_pth=saved_res_path,
            file_name=file_name,
            run_id=run)

    elif test_cfg.save_solutions:
        logger.info(f"Storing Results of run {run} in {log_path}")
        save_results(
            result={
                "solutions": results,
                "summary": stats,
                "machine": {'CPU': cpu_name + ":" + str(cpu_count), 'GPU': gpu_name + ":" + str(gpu_count)}
            },
            log_pth=log_path,
            run_id=run)

    return results, per_instance_summaries, stats


def save_results(result: Dict, log_pth, file_name: str = None, run_id: int = 0):
    if file_name is None:
        pth = os.path.join(log_pth, "run_" + str(run_id) + "_results.pkl")
    else:
        pth = os.path.join(log_pth, file_name)
    torch.save(result, pth)


def update_bks(sols, new_bks_list, ds_path, ds_class, acronym):
    """update the new Best Known Solution Dict in case new_bks_list is not empty."""

    # load registry for BKS and Top 8 Ranking of Test set
    if ds_path[-3:] not in [".pt", "pkl"]:
        try:
            BKS_path = os.path.join(ds_path, "BKS_" + ds_path.split("/")[-1] + ".pkl")
            # bks_registry = torch.load(BKS_path)
            print("BKS_path in 1st try", BKS_path)
        except FileNotFoundError:
            BKS_path = os.path.join(ds_path, "BKS_" + ds_path.split("/")[-2] + ".pkl")
            # bks_registry = torch.load(BKS_path)
            print("BKS_path in 1st except", BKS_path)
    else:
        print('path bks', os.path.join(os.path.dirname(ds_path), "BKS_"
                                       + os.path.basename(ds_path).split('.')[0] + ".pkl"))
        if "seed" in os.path.basename(ds_path):
            BKS_path = os.path.join(os.path.dirname(ds_path), "BKS_" +
                                    os.path.basename(ds_path).split('_seed')[0] + ".pkl")
            print("BKS_path in if", BKS_path)
        elif "size" in os.path.basename(ds_path):
            BKS_path = os.path.join(os.path.dirname(ds_path), "BKS_" +
                                    os.path.basename(ds_path)[:3] + ".pkl")
            print("BKS_path in elif", BKS_path)
        else:
            BKS_path = os.path.join(os.path.dirname(ds_path), "BKS_" +
                                    os.path.basename(ds_path).split('.')[0] + ".pkl")
            print("BKS_path in else (no 'seed' or 'size' in data_load_path)", BKS_path)
    logger.info(f"Storing new BKS in: {BKS_path}")
    bks_registry = torch.load(BKS_path)

    # update BKS for this test set if there are new best costs
    if new_bks_list:
        for id_ in new_bks_list:
            solution_tuple = [sol for sol in sols if sol.instance.instance_id == id_
                              or str(sol.instance.instance_id) == id_][0]  # for unsorted IDS
            # or sol.instance.instance_id == int(id_)
            if len(bks_registry[id_]) < 4:
                bks_registry[id_] = (solution_tuple.cost, solution_tuple.solution, acronym)
            else:
                if bks_registry[id_][3] == 'opt':
                    if math.ceil(solution_tuple.cost) < bks_registry[id_][0]:
                        logger.info(f"Incurred Cost that is lower than a proven optimal cost for instance {id_}! " 
                                    f"Not updating BKS for this Instance. " 
                                    f"Please double check calculation of cost.")
                    if math.ceil(solution_tuple.cost) == bks_registry[id_][0] or solution_tuple.cost == \
                            bks_registry[id_][0]:
                        logger.info(f"Incurred the same cost than a proven optimal cost for instance {id_}! " 
                                    f"Not updating BKS for this Instance. ")
                else:
                    bks_registry[id_] = (solution_tuple.cost, solution_tuple.solution, acronym, 'not_opt')
                    # logger.info(f"Storing new BKS for instances {new_bks_list} in {BKS_path}")
        logger.info(f"Storing new BKS for instances {new_bks_list} in {BKS_path}")

    # save back the update registry of BKS
    torch.save(bks_registry, BKS_path)


def update_bks_post_inference(sols: List[RPSolution], BKS_path: str, acronym:str, dataset_len=None, save_bks=False):
    new_bks_list = []
    dataset_len = len(sols) if dataset_len is None else dataset_len
    # load bks
    bks_registry = torch.load(BKS_path)
    for id_ in range(dataset_len):
        id_str = str(id_)
        solution_tuple = [sol for sol in sols if sol.instance.instance_id == id_
                          or str(sol.instance.instance_id) == id_][0]  # for unsorted IDS
        # # or sol.instance.instance_id == int(id_)
        print('id_str', id_str)
        print('solution_tuple.instance.instance_id', sols[id_].instance.instance_id)
        print('solution_tuple.cost', sols[id_].cost)
        print('bks_registry[id_str][0]', bks_registry[id_str][0])
        print('solution_tuple.cost < bks_registry[id_str][0]', solution_tuple.cost < bks_registry[id_str][0])
        print('solution_tuple.instance.instance_id == id_str', solution_tuple.instance.instance_id == id_str)
        if solution_tuple.cost < bks_registry[id_str][0] and solution_tuple.instance.instance_id == id_:
            new_bks_list.append(id_str)
            if len(bks_registry[id_str]) < 4:
                bks_registry[id_str] = (solution_tuple.cost, solution_tuple.solution, acronym)
            else:
                if bks_registry[id_str][3] == 'opt':
                    if math.ceil(solution_tuple.cost) < bks_registry[id_][0]:
                        logger.info(f"Incurred Cost that is lower than a proven optimal cost for instance {id_}! " 
                                    f"Not updating BKS for this Instance. " 
                                    f"Please double check calculation of cost.")
                    if math.ceil(solution_tuple.cost) == bks_registry[id_str][0] or solution_tuple.cost == \
                            bks_registry[id_][0]:
                        logger.info(f"Incurred the same cost than a proven optimal cost for instance {id_}! " 
                                    f"Not updating BKS for this Instance. ")
                else:
                    bks_registry[id_str] = (solution_tuple.cost, solution_tuple.solution, acronym, 'not_opt')
                    # logger.info(f"Storing new BKS for instances {new_bks_list} in {BKS_path}")
    print(f"Storing new BKS for instances {new_bks_list} in {BKS_path}")
    logger.info(f"Storing new BKS for instances {new_bks_list} in {BKS_path}")
    if save_bks:
        # save back the update registry of BKS
        torch.save(bks_registry, BKS_path)
    return bks_registry


# def get_combined_base_ref(cpu_base_single, cpu_base_multi, gpu_mark_2D, gpu_mark_3D):
#     cpu_power
#     return 1 / (((1 / (cpu_power * 0.396566187)) + (1 / (twoD_Mark * 3.178718116))
#           + (1 / (threeD_Mark * 2.525195879))) / 3)

def get_cpu_specs():
    cpu_info_dct = cpuinfo.get_cpu_info()
    # print('cpu_info_dct', cpu_info_dct)
    if bool(cpu_info_dct) and 'brand_raw' in cpu_info_dct.keys():
        cpu_name = cpu_info_dct['brand_raw']
    else:
        f = open('/proc/cpuinfo', 'r')
        cpu_info_lst = f.readlines()
        f.close()
        name_list = [string for string in cpu_info_lst if string[:10] == "model name"]
        cpu_name = list(set(name_list))[0].split(': ')[1].split('\n')[0]
        # print('cpu_name', cpu_name)
    total_cores = psutil.cpu_count(logical=False)  # psutil.cpu_count()
    threads_per_cpu = int(psutil.cpu_count() / psutil.cpu_count(logical=False))
    logger.info(f'CPU Specs: cpu_name: {cpu_name}, threads_per_cpu: {threads_per_cpu}, total_cores: {total_cores}')
    return cpu_name, threads_per_cpu, total_cores


def get_seperate_PassMarks(CPU_Mark: int, CPU_Mark_single: int, threeD_Mark: int, twoD_Mark: int, number_threads: int,
                           total_threads: int, total_cpus: int = 1, total_gpus: int = 1):
    # https://forums.passmark.com/performancetest/4599-formula-cpu-mark-memory-mark-and-disk-mark
    #  version 10 the updated numbers (2020):
    # all_ingredients_PassM = 1 / (((1 / (CPU_Mark * 0.396566187)) + (1 / (twoD_Mark * 3.178718116))
    #                               + (1 / (threeD_Mark * 2.525195879)) + (1 / (Memory_Mark * 1.757085479))
    #                               + (1 / (Disk_Mark * 1.668158805))) / 5)
    print('number_threads', number_threads)
    cpu_performance = min(round(number_threads * CPU_Mark_single), CPU_Mark)
    # cpu_perf = round((number_threads * (CPU_Mark / total_threads)))
    print('cpu_performance', cpu_performance)
    # reset total cpus to 1, because passmark value incorp all cores already
    if total_gpus != 0:
        GPU_passmark = round((cpu_performance + ((0.5 * threeD_Mark) + (0.5 * twoD_Mark))) / 2)
        print('GPU_passmark', GPU_passmark)
        print('GPU_BASE_REF', GPU_BASE_REF)
    else:
        GPU_passmark = None
    # print('CPU_overall_PassMark (not used)', round(1 / (1 / (additional_cpus * (cpu_perf * 0.396566187)))))
    CPU_passmark = cpu_performance  # round(1 / (1 / (additional_cpus * (cpu_perf * 0.396566187))))
    print('overall CPU passMark: ', CPU_passmark)
    print('CPU_BASE_REF', CPU_BASE_REF_SINGLE if number_threads == 1 else CPU_BASE_REF_MULTI)

    return GPU_passmark, CPU_passmark


def get_overall_PassMark_v1(CPU_Mark: int, CPU_Mark_single: int, threeD_Mark: int, twoD_Mark: int, number_threads: int,
                            total_threads: int, total_cpus: int = 1, total_gpus: int = 1):
    print('CPU_Mark_single', CPU_Mark_single)
    print('threeD_Mark', threeD_Mark)
    print('number_threads', number_threads)
    print('total_cpus', total_cpus)
    print('total_gpus', total_gpus)
    # compute cpu performance
    cpu_performance = min(round(number_threads * CPU_Mark_single), CPU_Mark)  # CPU_Mark is Mark for whole processor
    print('cpu_performance', cpu_performance)
    if total_gpus == 0:
        gpu_performance = 0
    else:
        gpu_performance = (0.5 * threeD_Mark) + (0.5 * twoD_Mark)
    print('gpu_performance', gpu_performance)
    # overall_passmark = round(cpu_performance + (0.5 * (total_gpus * threeD_Mark)))
    overall_passmark = round((cpu_performance + (total_gpus * gpu_performance)) / 2)
    print('overall passMark (V1): ', overall_passmark)
    return overall_passmark


def set_passMark(cfg, device, number_threads=1, passmark_version=PASSMARK_VERSION):
    if cfg.run_type in ["val", "test"]:
        global NUMBER_THREADS_USED
        NUMBER_THREADS_USED = number_threads
        # get cpu info
        cpu_name, threads_per_cpu, total_cpus = get_cpu_specs()
        # cpu_mark, cpu_s_mark, n_cores, n_threads
        print('cpu_name', cpu_name)
        CPUMark, CPUMark_single, cpu_cores, total_threads = get_machine_info(which_type="cpu",
                                                                             machine_name=cpu_name)
        print(f'CPUMark {CPUMark}, CPUMark_single, {CPUMark_single}, '
              f'cpu_cores {cpu_cores}, total_cpus {total_cpus}, total_threads {total_threads}')
        # CPUMark = CPU_MACHINES[cpu_name][0]
        # CPUMark_single = CPU_MACHINES[cpu_name][1]
        # cpu_cores = CPU_MACHINES[cpu_name][2]
        # total_threads = CPU_MACHINES[cpu_name][3]
        assert cpu_cores == total_cpus, f"total amount of cores does not match. Different machine?"
        if total_threads != threads_per_cpu * cpu_cores:
            warnings.warn(f"Number of Threads does not match. Different machine?")
        if not device == torch.device("cpu"):
            try:
                print('torch.cuda.device_count()', torch.cuda.device_count())
                gpu_device_count = torch.cuda.device_count()
                gpu_device_name = torch.cuda.get_device_name()
                logger.info(f"GPU Device Name: {gpu_device_name}")
            except AssertionError:
                gpu_device_count = 1
                gpu_device_name = 'Apple M2 10-Core GPU'
            # if gpu_device_name not in GPU_MACHINES.keys():
            #     warnings.warn(f"Getting G3DMark and G2DMark from config.")
            #     G3DMark = cfg.G3DMark
            #     G2DMark = cfg.G2DMark
            # else:
            # G3DMark = GPU_MACHINES[gpu_device_name][0]
            # G2DMark = GPU_MACHINES[gpu_device_name][1]
            G3DMark, G2DMark = get_machine_info(which_type="gpu",
                                                machine_name=gpu_device_name)
        else:
            G3DMark, G2DMark = None, None
            gpu_device_count = 0
        if passmark_version == "v1":
            passMark = get_overall_PassMark_v1(CPUMark, CPUMark_single, G3DMark, G2DMark, number_threads,
                                               total_threads, total_cpus, gpu_device_count)
            cpu_perf = passMark
        else:
            passMark, cpu_perf = get_seperate_PassMarks(CPUMark, CPUMark_single, G3DMark, G2DMark, number_threads,
                                                        total_threads, total_cpus, gpu_device_count)
            if passMark is None:
                passMark = cpu_perf
    else:
        passMark, cpu_perf = None, None
    return passMark, cpu_perf


def set_device(cfg):
    logger.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")
    logger.info(f"cfg.cuda {cfg.cuda}")
    if torch.cuda.is_available() and not cfg.cuda:
        warn(f"Cuda GPU is available but not used! Specify <cuda=True> in config file.")
    elif torch.backends.mps.is_available() and not cfg.cuda:
        warn(f"MPS GPU is available but not used! Specify <cuda=True> in config file.")
    if torch.cuda.is_available():
        device = torch.device("cuda" if cfg.cuda else "cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps" if cfg.cuda else "cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Running {cfg.run_type}-run on {device}")

    # raise error on strange CUDA/MPS warnings which are not caught
    if (
            cfg.run_type == "train") and cfg.cuda and not torch.cuda.is_available() and not torch.backends.mps.is_available():
        e = "..."
        try:
            torch.zeros(10, device=device)
        except Exception as e:
            pass
        raise RuntimeError(f"specified training run on GPU but running on CPU! ({str(e)})")

    # set global device to access in saving solution
    global DEVICE
    DEVICE = str(device)

    return device


def _adjust_time_limit(original_TL, pass_mark, device, nr_threads=1, passmark_version=PASSMARK_VERSION):
    # TODO: Tmax = n × 240/100 seconds (used in HGS-CVRP)
    # Therefore, the smallest instance with 100 clients
    # is run for 4 minutes, whereas the largest instance containing 1000 clients is run for 40 minutes."

    print(f'IN ADJUST TIME LIMIT: device: {device}, pass_mark: {pass_mark}, nr_threads: {nr_threads}, '
          f'passmark_v: {passmark_version}, ')
    if passmark_version == "v1":
        print(f"MACHINE_BASE_REF_v1: {MACHINE_BASE_REF_v1}")
        return np.round(original_TL / (pass_mark / MACHINE_BASE_REF_v1))
    else:
        if device == torch.device("cuda"):
            print(f"GPU_BASE_REF: {GPU_BASE_REF}")
            return np.round(original_TL / (pass_mark / GPU_BASE_REF))
            # return np.round(original_TL / (pass_mark / COMBINED_BASE_REF))
        elif device == torch.device("cpu"):
            CPU_BASE_REF = CPU_BASE_REF_SINGLE if nr_threads == 1 else CPU_BASE_REF_MULTI
            # nr_threads*CPU_BASE_REF_SINGLE
            print(f"CPU_BASE_REF: {CPU_BASE_REF}")
            return np.round(original_TL / (pass_mark / CPU_BASE_REF))
        else:
            logger.info(f"Device {device} not known - specify 'cuda' or 'cpu' to adjust Time Limit for Evaluation.")


def merge_sols(sols_search, sols_construct):
    return [sol.update(running_sols=[sols_construct[
                                         i].solution] + sol.running_sols if sol is not None else sol.running_sols,
                       running_times=[sols_construct[i].run_time] + [
                           t + sols_construct[i].run_time for t in
                           sol.running_times] if sol is not None else sol.running_times,
                       run_time=sols_construct[
                                    i].run_time + sol.run_time if sol is not None else sol.run_time,
                       method_internal_cost=sols_construct[i].method_internal_cost)
            for i, sol in enumerate(sols_search)]


def print_summary_stats(all_stats, number_runs):
    average_costs = [stat['avg_cost'] for stat in all_stats]
    average_pi = [stat['avg_pi'] for stat in all_stats]
    average_wrap = [stat['avg_wrap'] for stat in all_stats]
    average_rt_best = [stat['avg_run_time_best'] for stat in all_stats]
    average_rt_total = [stat['avg_run_time_total'] for stat in all_stats]
    logger.info(f"\n\nSummary Stats of the {number_runs} Runs: \n"
                f"\n"
                f"Average Obj. Costs over {number_runs} Runs: {sum(average_costs) / number_runs}\n"
                f"Std. Dev. of Avg Costs over {number_runs} Runs: {np.std(average_costs)}\n"
                f"Average PI Score over {number_runs} Runs: "
                f"{np.mean(average_pi) if not average_pi.count(None) == len(average_pi) else None}\n"
                f"Average WRAP Score over {number_runs} Runs: "
                f"{np.mean(average_wrap) if not average_wrap.count(None) == len(average_wrap) else None}\n"
                f"Average Runtime (until best cost found) over {number_runs} Runs: "
                f"{sum(average_rt_best) / number_runs}\n"
                f"Average Total Runtime of method over {number_runs} Runs: "
                f"{sum(average_rt_total) / number_runs}")


def get_stats(sols, logger_, model_name, debug_flag, ranking_updates=None, N_runs=None, run_i=None):
    """print out some general statistical information. (Eventually get summary stats)"""

    if debug_flag:
        logger_.info(f"First and Last solution; {sols[0]} {sols[-1]}")

    avg_cost, avg_time, avg_total_time, avg_PI, avg_WRAP, not_solved, avg_cost_int = [], [], [], [], [], [], []
    run_avg_cost = 0
    run_avg_cost_int = 0
    for i, sol_ in enumerate(sols):
        avg_PI.append(sol_.pi_score)
        avg_WRAP.append(sol_.wrap_score)
        if sol_.cost != float('inf'):
            avg_cost.append(sol_.cost)
            run_avg_cost += sol_.cost
            run_avg_cost_int += sol_.method_internal_cost if sol_.method_internal_cost is not None else 1
            if sol_.method_internal_cost is not None:
                avg_cost_int.append(sol_.method_internal_cost)
            avg_total_time.append(sol_.run_time)
            # print('sol_.running_times', sol_.running_times)
            avg_time.append(sol_.running_times[-1] if sol_.running_times is not None else sol_.run_time)
        else:
            not_solved.append(i)

    if not avg_PI.count(10) == 0:
        logger_.info(f"PI = 10 for {avg_PI.count(10)} instances out of {len(sols)}.")
        if not_solved:
            logger_.info(f"{model_name} did not solve the following instances: {not_solved}")

    if not avg_WRAP.count(1) == 0:
        logger_.info(f"WRAP = 1 for {avg_WRAP.count(1)} instances out of {len(sols)}.")
        if not_solved:
            logger_.info(f"{model_name} did not solve the following instances: {not_solved}")

    # if self.debug:
    if run_i is not None and N_runs is not None:
        logger_.info(f"Stats for run {run_i}/{N_runs}:")
        logger_.info("=================================")
    else:
        logger_.info(f"Stats for run 1/1:")
        logger_.info("====================")

    print(f"Stats per instance (up to first 5 instances):")
    print(f"--------------------")
    for sol in sols[:5]:
        logger_.info(
            f"\nInstance {sol.instance.instance_id} Cost: {sol.cost}, \n"
            f"Instance {sol.instance.instance_id} Method Internal-Cost: {sol.method_internal_cost}, \n"
            f"Instance {sol.instance.instance_id} PI: {sol.pi_score}, \n"
            f"Instance {sol.instance.instance_id} WRAP: {sol.wrap_score}, \n"
            f"Instance {sol.instance.instance_id} Run Time (best sol found): "
            f"{sol.running_times[-1] if sol.running_times and sol.running_times is not None else sol.run_time}, \n"
            f"Instance {sol.instance.instance_id} Run Time (total): {sol.run_time}")

    print(f"Average Stats")
    print(f"--------------")
    if not len(avg_cost) == 0:
        avg_cost_print = run_avg_cost / len(avg_cost)
        std_cost_print = np.std(avg_cost)
    else:
        avg_cost_print = float('inf')
        std_cost_print = None
    if not len(avg_cost_int) == 0:
        run_avg_cost_int_print = run_avg_cost_int / len(avg_cost_int)
        run_std_cost_int_print = np.std(run_avg_cost_int)
    else:
        run_avg_cost_int_print = None
        run_std_cost_int_print = None
    logger_.info(
        f"\nAverage cost  : {avg_cost_print} +/- {std_cost_print}, \n"
        f"Average cost-intern (Method)  : {run_avg_cost_int_print} +/- {run_std_cost_int_print}, \n"
        f"Average PI: {np.mean(avg_PI) if not avg_PI.count(None) == len(avg_PI) else None} "
        f"+/- {np.std(avg_PI) if not avg_PI.count(None) == len(avg_PI) else None}, \n"
        f"Average WRAP: {np.mean(avg_WRAP) if not avg_WRAP.count(None) == len(avg_WRAP) else None} "
        f"+/- {np.std(avg_WRAP) if not avg_WRAP.count(None) == len(avg_WRAP) else None}, \n"
        f"Average Run Time (best sol found): {np.mean(avg_time)} +/- {np.std(avg_time)}, \n"
        f"Average Run Time (total): {np.mean(avg_total_time)} +/- {np.std(avg_total_time)}")

    return {
        "avg_cost": avg_cost_print,
        "std_cost": std_cost_print,
        "avg_pi": np.mean(avg_PI) if not avg_PI.count(None) == len(avg_PI) else None,
        "std_pi": np.std(avg_PI) if not avg_PI.count(None) == len(avg_PI) else None,
        "avg_wrap": np.mean(avg_WRAP) if not avg_WRAP.count(None) == len(avg_WRAP) else None,
        "std_wrap": np.std(avg_WRAP) if not avg_WRAP.count(None) == len(avg_WRAP) else None,
        "avg_run_time_best": np.mean(avg_time),
        "std_run_time_best": np.std(avg_time),
        "avg_run_time_total": np.mean(avg_total_time),
        "std_run_time_total": np.std(avg_total_time),
    }


def log_info(logger_, cfg, run, number_of_runs, model):
    logger_.info(f"Starting {cfg.test_cfg.eval_type} Evaluation for run {run}/{number_of_runs} "
                 f"with time limit {cfg.test_cfg.time_limit} for {model}")


def update_ranking(sols, data_file_path, acronym):
    """update the Top 8 solvers list based on the achieved PI score (incl. the one producing BKS)
        for the instance in the respective test or validation dataset"""
    if update_ranking:
        pth_rank = os.path.join(os.path.dirname(data_file_path),
                                "rank_val.pkl")
        ranking_registry = torch.load(pth_rank)
        # check if ranking of PI values has changed and update in case it did
        for id_ in range(len(sols)):
            # get values and ranking
            curr_pi = sols[id_].pi_score
            top_8_tuples = ranking_registry[str(id_)][1]
            # (check if model exists in ranking and if pi value has changed)
            if sum([1 for tup in top_8_tuples if tup[1] == acronym]) == 1:
                tuple_pi = [(j, pi) for j, (pi, model) in enumerate(top_8_tuples) if model == acronym][0]
                if curr_pi < tuple_pi[1]:
                    top_8_tuples[tuple_pi[0]] = (curr_pi, acronym)
                    top_8_tuples.sort(key=lambda y: y[0])
                    ranking_registry[str(id_)][1] = top_8_tuples[:8]
                # else: nothing gets changed
            # else: just add and sort
            else:
                top_8_tuples.append((curr_pi, acronym))
                top_8_tuples.sort(key=lambda y: y[0])
                ranking_registry[str(id_)][1] = top_8_tuples[:8]
            # top_8_pi = [top_8[0] for top_8 in top_8_tuples]
            # save back the update registry of BKS and Ranking
            torch.save(ranking_registry, pth_rank)

        example_ranking = ranking_registry[str(0)][1]
        # save back the update registry of BKS
        torch.save(ranking_registry, pth_rank)

        return example_ranking


def plot_traj(c_lists, t_lists, model_names_list, TL, path_to_plt):
    fig, ax = plt.subplots()
    for cs, ts, model_name in zip(c_lists, t_lists, model_names_list):
        ax.plot(ts, cs, label=model_name)
        # plt.plot(t, c, label=model_name)
    plt.xlabel('cumulative runtime (seconds) ')
    plt.ylabel('objective value (total cost)')
    plt.title('Trajectories for Time Limit ' + str(TL))
    save_name = os.path.join(path_to_plt, 'plttd_trajectories')
    plt.legend()
    plt.savefig(save_name + '.pdf')
    plt.show()


def plot_multiple_trajectories(model_names_list, path_to_plot, inst_id, TL=None):
    # infer the TL for evaluation run
    TL = [word[3:] for word in path_to_plot.split("/") if word[:2] == 'TL'][0] if TL is None else TL
    times_lists = [torch.load(path_to_plot + '/trajectory_times_' + str(inst_id) + '_' + model + '.pt') for model in
                   model_names_list]
    costs_lists = [torch.load(path_to_plot + '/trajectory_costs_' + str(inst_id) + '_' + model + '.pt') for model in
                   model_names_list]
    plot_traj(costs_lists, times_lists, model_names_list, TL, path_to_plot)


def plot_metrics(model_names_list,
                 results_path,
                 TL: int = None,
                 plot_over_TL: bool = True,
                 plot_per_instance: bool = True,
                 plot_over_sets: bool = True,
                 plot_sets_over_TL: bool = True,
                 plot_PI: bool = True,
                 plot_WRAP: bool = True,
                 inst_type_set: str = None,
                 run_id: int = 1):
    inst_type_set = results_path.split("/")[2] if inst_type_set is None else inst_type_set
    #     if plot_over_sets:
    #         # results_path needs to lead to main results ("outputs/saved_results/")
    #         # list_sets = os.listdir(results_path)
    #         for set in os.listdir(results_path):
    #             pth = os.path.join(results_path, set, 'TL_' + str(TL), 'run_' + str(run_id) + '_results_')
    #             results_dicts = [torch.load(pth + model + '.pkl') for model in model_names_list]
    #             solution_dicts = [res_dicts["solutions"] for res_dicts in results_dicts]
    #             set_ids = [sol.instance.instance_id for sol in solution_dicts[0]]
    #             metric_values = {}
    #             for sols, model in zip(solution_dicts, model_names_list):
    #                 metric_values[model] = ([sol.pi_score for sol in sols], [sol.wrap_score for sol in sols])
    if plot_per_instance:
        # results_path needs to lead already to TL for which to plot ("outputs/saved_results/XE_1/TL_10/")
        pth = results_path + 'run_' + str(run_id) + '_results_'
        results_dicts = [torch.load(pth + model + '.pkl') for model in model_names_list]
        solution_dicts = [res_dicts["solutions"] for res_dicts in results_dicts]
        instance_ids = [sol.instance.instance_id for sol in solution_dicts[0]]
        metric_values = {}
        for sols, model in zip(solution_dicts, model_names_list):
            metric_values[model] = ([sol.pi_score for sol in sols], [sol.wrap_score for sol in sols])
        if plot_PI:
            plot_scatter_per_instance(metric_values, "PI", instance_ids, inst_type_set, results_path, idx_=0)
        if plot_WRAP:
            plot_scatter_per_instance(metric_values, "WRAP", instance_ids, inst_type_set, results_path, idx_=1)
    if plot_over_TL:
        # results_path needs to lead only toinst type ("outputs/saved_results/XE_1/")
        TLs = []
        for TL_folder in os.listdir(results_path):
            if os.path.isdir(os.path.join(results_path, TL_folder)):
                TLs.append(int(TL_folder[3:]))
        TLs.sort()
        metric_means = {}
        for model in model_names_list:
            metric_means[model] = [[]] * len(TLs)
        # print(metric_means)
        # print('TLs', TLs)
        for i, tl in enumerate(TLs):
            pth = os.path.join(results_path, 'TL_' + str(tl), 'run_' + str(run_id) + '_results_')
            results_dicts = [torch.load(pth + model + '.pkl') for model in model_names_list]
            solution_dicts = [res_dicts["solutions"] for res_dicts in results_dicts]
            for sols, model in zip(solution_dicts, model_names_list):
                # print('tl', tl)
                # print('model', model)
                metric_means[model][i] = (
                    np.mean([sol.pi_score for sol in sols]), np.mean([sol.wrap_score for sol in sols]))
        for model in metric_means:
            all_i, all_j = [], []
            for i, j in metric_means[model]:
                all_i.append(i)
                all_j.append(j)
            metric_means[model] = [all_i, all_j]
        if plot_PI:
            plot_plot_across_TL(metric_means, "PI", inst_type_set, results_path, TLs, idx_=0)
        if plot_WRAP:
            plot_plot_across_TL(metric_means, "WRAP", inst_type_set, results_path, TLs, idx_=1)


def plot_plot_across_TL(metric_values, metric_name, instance_type_set, path, time_limits, idx_):
    fig, ax = plt.subplots()
    for i, model_name in enumerate(metric_values):
        ax.plot(np.arange(len(time_limits)), metric_values[model_name][idx_], label=model_name)
    plt.xticks(np.arange(len(time_limits)), time_limits)
    plt.xlabel('Time Limits')
    plt.ylabel(metric_name)
    plt.title('Average_' + metric_name + ' values for Instance Set ' + instance_type_set)
    save_name = os.path.join(path, 'Average_' + metric_name + '_across_Time_Limits')
    plt.legend()
    plt.savefig(save_name + '.pdf')
    plt.show()


def plot_scatter_per_instance(metric_values, metric_name, instance_ids, instance_type_set, path, idx_):
    fig, ax = plt.subplots()
    for model_name, metrics in metric_values.items():
        ax.scatter(np.arange(len(metrics[idx_])), metrics[idx_], label=model_name)
    plt.xticks(np.arange(len(instance_ids)), instance_ids)
    plt.xlabel('Instance IDs ')
    plt.ylabel(metric_name)
    plt.title(metric_name + ' values for Instance Set ' + instance_type_set)
    save_name = path + metric_name + '_across_' + instance_type_set + '_instances'
    plt.legend()
    plt.savefig(save_name + '.pdf')
    plt.show()


def plot_scatter_across_sets(metric_values, metric_name, set_ids, type_sets, path, idx_):
    fig, ax = plt.subplots()
    for model_name, metrics in metric_values.items():
        ax.scatter(np.arange(len(metrics[idx_])), metrics[idx_], label=model_name)
    plt.xticks(np.arange(len(set_ids)), set_ids)
    plt.xlabel('Instance Sets')
    plt.ylabel(metric_name)
    plt.title(metric_name + ' values for ' + type_sets)
    save_name = path + metric_name + '_across_' + type_sets
    plt.legend()
    plt.savefig(save_name + '.pdf')
    plt.show()


def converting_sol_files_to_BKS_dcts(path_to_sol_files):
    # "data/test_data/cvrp/Vrp-Set-XML100/solutions"
    empty_bks_dct = {}
    for file_name in os.listdir(path_to_sol_files):
        print('file_name', file_name)
        instance_id = file_name[:-4]
        sol_path = path_to_sol_files + "/" + file_name
        tours = []
        with open(sol_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):  # read out solution tours
                if line[:5] == 'Route':
                    l = line.strip().split()
                    tours.append([int(idx) for idx in l[2:]])
                if line[:4] == 'Cost':
                    l = line.strip().split()
                    cost = int(l[1])
        empty_bks_dct[instance_id] = (cost, tours, "solver_name", 'opt')

    return empty_bks_dct


def subsample_XML(instances_path, subsample_path=None, nr_inst_per_group=1, nr_groups=376):
    nr_groups = 376
    choices = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]]
    subsample_path = 'data/test_data/cvrp/XML100/subsampled/instances' if subsample_path is None else subsample_path
    os.makedirs(subsample_path)
    shutil.copyfile(instances_path + "/BKS_XML100.pkl", subsample_path + "/BKS_XML100.pkl")
    inst_ids_group = []
    for instance_group in list(itertools.product(*choices)):
        # make str
        group_code = ''
        for i in list(instance_group):
            # print('i', i)
            group_code += str(i)
        # how many instances per group
        for i in range(1, nr_inst_per_group + 1):
            if len(str(i)) == 1:
                inst_ids_group.append(group_code + '_' + str(0) + str(i))
            else:
                inst_ids_group.append(group_code + '_' + str(i))
        # print('inst_ids_group', inst_ids_group)
    for file_name in os.listdir(instances_path):
        if file_name[7:-4] in inst_ids_group:
            shutil.copyfile(instances_path + "/" + file_name, subsample_path + "/" + file_name)


def update_bks_file(bks_local: dict, bks: dict):
    assert len(bks_local.keys()) == len(bks.keys()), f"Length mismatch, both bks files do not have the same length."
    new_bks_dct = {}
    for inst_id in bks.keys():
        local_bks = bks_local[inst_id][0]
        new_bks = bks[inst_id][0]
        if new_bks < local_bks:
            new_bks_dct[inst_id] = bks[inst_id]
        else:
            new_bks_dct[inst_id] = bks_local[inst_id]
    assert bks_local.keys() == bks.keys() == new_bks_dct.keys()

    return new_bks_dct

def merging_base_sols(idx_start_actuals, all_base_sol_parts, base_sol_0):
    # all_base_sol_parts --> list contains all part dct solution files except first
    # base_sol_0 --> dict of first part of the solutions to be merged
    # idx_start_actuals = [1000, 2000, 4000, 6000, 8000, 10000]
    base_sol_merged = base_sol_0.copy()
    for l_idx, dct in enumerate(all_base_sol_parts):
        i_actual_start = idx_start_actuals[l_idx]
        i_actual_end = idx_start_actuals[l_idx + 1]
        print('i_act_start, i_act_end', i_actual_start, i_actual_end)
        i_dct = 0 # assume all dict sub files start with instance ID 0
        for i_act in range(i_actual_start, i_actual_end):
            base_sol_merged[str(i_act)] = dct[str(i_dct)]
    return base_sol_merged

# def get_overall_PassMark_v2(CPU_Mark: int, CPU_Mark_single: int, threeD_Mark: int, twoD_Mark: int, number_threads: int,
#                         total_threads: int, total_cpus: int = 1, total_gpus: int = 1):
# print('CPU_Mark', CPU_Mark)
# print('number_threads', number_threads)
# # cpu_performance = max(round(number_threads * CPU_Mark_single), CPU_Mark)  # CPU_Mark is Mark for whole processor
# if twoD_Mark is None and threeD_Mark is None:
#     overall_passmark = round(1 / (1 / (CPU_Mark * 0.396566187)))
#     # overall_passmark = round(1 / (1 / (CPU_Mark * 1)))  # 11980
# else:
#     print('counting GPU')
#     overall_passmark = round(1 / ((1 / (CPU_Mark * 0.396566187)) +
#                                   (1 / (twoD_Mark * 3.178718116)) +
#                                   (1 / (threeD_Mark * 1.525195879)) / 3))
#
# print('overall passMark (V2): ', overall_passmark)
# return overall_passmark

# MACHINE_BASE_REF_v2 = round(1 / (((1 / (CPU_BASE_REF_SINGLE * 0.396566187))
#                                   + (1 / (GPU_2D_BASE_REF * 3.178718116)) + (
#                                               1 / (GPU_3D_BASE_REF * 2.525195879))) / 3))


# def get_overall_PassMark_v3(CPU_Mark: int, CPU_Mark_single: int, threeD_Mark: int, twoD_Mark: int,
# number_threads: int, total_threads: int, total_cpus: int = 1, total_gpus: int = 1):
#     # https://forums.passmark.com/performancetest/4599-formula-cpu-mark-memory-mark-and-disk-mark
#     #  version 10 the updated numbers (2020):
#     # all_ingredients_PassM = 1 / (((1 / (CPU_Mark * 0.396566187)) + (1 / (twoD_Mark * 3.178718116))
#     #                               + (1 / (threeD_Mark * 2.525195879)) + (1 / (Memory_Mark * 1.757085479))
#     #                               + (1 / (Disk_Mark * 1.668158805))) / 5)
#     print('CPU_Mark', CPU_Mark)
#     print('number_threads', number_threads)
#     # CPU_Mark is Mark for whole processor
#     cpu_performance = min(round(number_threads * CPU_Mark_single), CPU_Mark)
#     print('cpu_performance', cpu_performance)
#     if twoD_Mark is None and threeD_Mark is None:
#         # overall_passmark = round(1 / (1 / (CPU_Mark * 0.396566187)))
#         overall_passmark = round(1 / (1 / (cpu_performance * 1)))
#     else:
#         overall_passmark = round(1 / (((1 / (cpu_performance * 0.396566187)) +
#                                        (1 / (total_gpus * (twoD_Mark * 3.178718116))) +
#                                        (1 / (total_gpus * (threeD_Mark * 1.525195879)))) / 3))
#
#     print('overall passMark (V3): ', overall_passmark)
#     return overall_passmark

# MACHINE_BASE_REF_v3 = round(1 / (((1 / (std_nr_threads * CPU_BASE_REF_SINGLE * 0.396566187))
#                                   + (1 / (std_nr_gpus * GPU_2D_BASE_REF * 3.178718116))
#                                   + (1 / (std_nr_gpus * GPU_3D_BASE_REF * 2.525195879))) / 3))
