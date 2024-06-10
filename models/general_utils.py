import pandas as pd
import warnings
import os
from typing import Tuple


def get_machine_info(which_type: str, machine_name: str) -> Tuple:
    current_dir = os.getcwd()
    root_dir = os.path.join(current_dir.split("routing-arena")[0], "routing-arena")
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