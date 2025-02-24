import pandas as pd
import warnings
import os
from typing import Tuple
import torch as pytorch

# from d2l import torch


def get_machine_info(which_type: str, machine_name: str) -> Tuple:
    current_dir = os.getcwd()
    root_dir = os.path.join(current_dir.split("Routing-Arena")[0], "Routing-Arena")
    if which_type == "cpu":
        score_registry_path = os.path.join(root_dir, 'machine_scores/cpu_scores.md')
    elif which_type == "gpu":
        score_registry_path = os.path.join(root_dir, 'machine_scores/gpu_scores.md')
    else:
        warnings.warn(f"Not sure which type of machine! Needs to be either 'cpu' or 'gpu'.")
        score_registry_path = None

    df_md = pd.read_table(score_registry_path, sep="|", header=0, index_col=1, skipinitialspace=True)  # import md
    df_md = df_md.dropna(axis=1, how='all')  # drop NaN info
    df_md = df_md.iloc[1:]  # drop seperator header
    df_md = df_md.rename(columns=lambda x: x.strip())  # strip column names
    df = df_md.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # strip rest of cells
    # device_name = machine_name
    if which_type == "cpu":
        name, cpu_mark, cpu_s_mark, n_cores, n_threads, _ = df.query("device_name == @machine_name").values.tolist()[0]
        # name, cpu_mark, cpu_s_mark, n_cores, n_threads = [int(x) for x in [name, cpu_mark, cpu_s_mark,
        # n_cores, n_threads]]
        return int(cpu_mark), int(cpu_s_mark), int(n_cores), int(n_threads)
    else:
        name, g3d_m, g2d_m, _ = df.query("device_name == @machine_name").values.tolist()[0]  # get g3d, g2d marks
        return int(g3d_m), int(g2d_m)


def merge_bks(source_path_1: str, source_path_2: str, save_merged_bks: bool, save_path: str = None):
    bks_1_all = pytorch.load(source_path_1)
    bks_2_all = pytorch.load(source_path_2)
    assert len(bks_1_all) == len(bks_2_all), print(f"both BKS registries do not have same length...")

    merged_registry = {}
    for i in range(len(bks_1_all)):
        bks_1 = bks_1_all[str(i)][0]
        bks_2 = bks_2_all[str(i)][0]
        if bks_1 <= bks_2:
            merged_registry[str(i)] = bks_1_all[str(i)]
        else:
            merged_registry[str(i)] = bks_2_all[str(i)]
    assert len(merged_registry) == len(bks_1_all) == len(bks_2_all)
    if save_merged_bks:
        pytorch.save(merged_registry, save_path)
    return merged_registry


def get_node_ids_of_sli(path_sli="data/train_data/cvrp_10000/uchoa_R_R_1/single_large_instance.pt",
                        path_test="data/test_data/cvrp/sub_uchoa/cvrp100_RR1/val_R_R_1_seed1234_size128.pt", ):
    node_ids_uch_rr1 = []
    sli_uch100_rr1 = torch.load(path_sli)
    test_uch100_rr1 = torch.load(path_test)
    count = 0
    test_data_len = len(test_uch100_rr1)
    for i in range(len(test_uch100_rr1)):
        # print(base_unf40[str(i)][0])
        inst_i_node_ids = []
        count_i = 0
        for j, (coords_, dems_) in enumerate(
                zip(sli_uch100_rr1[0].coords[1:], sli_uch100_rr1[0].node_features[1:, -1])):
            for k, (coord_i, demand_i) in enumerate(zip(test_uch100_rr1[i].coords, test_uch100_rr1[i].node_features)):
                if (coord_i == coords_).all() and (demand_i == dems_).all():
                    print('sample i=', i)
                    print(f'sli coord_j={j} found in coord_k={k}')
                    # print('dems_', dems_)
                    inst_i_node_ids.append(j)
                    count_i += 1
        print('count_i', count_i)
        # print('node instance ids for ', i, inst_i_node_ids)
        node_ids_uch_rr1.append(inst_i_node_ids)
        print('len(node_ids_uch_rr1)', len(node_ids_uch_rr1))
        count += 1
    if count == test_data_len:
        print("done")

    return node_ids_uch_rr1