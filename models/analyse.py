import numpy as np
import torch
import logging
import os
from warnings import warn
import matplotlib.pyplot as plt
from formats import RPSolution, CVRPInstance
from data import CVRPDataset, TSPDataset
from models import runner_utils
from metrics.metrics import Metrics
from typing import Optional, Dict, Union, List
import glob
import warnings

INSTANCE_SET_TYPES = ["X", "XE", "uniform", "uchoa"]
INSTANCE_SET_NAMES = ["X", "XE_1", "XE_2", "XE_3", "XE_4", "XE_5" "uniform", "uchoa"]


class Analyser:
    """Sampler implementing different options to generate data for RPs."""

    def __init__(self,
                 results_dir: str,
                 problem: str = "cvrp",
                 model_names_list: List[str] = None,
                 TL: Union[int, List[int]] = None,
                 load_instances: bool = True
                 ):
        """

        Args:
            results_dir: directory in outputs/saved_results that should to have different instance_sets with
                         different TL directories
            model_names_list: Optional List of models in results_dir that should be incorp. in analysis
            TL: List or Integer of the TL results to be analysed
        """
        self.results_dir = results_dir
        self.ds_class = CVRPDataset if problem.upper() == "CVRP" else TSPDataset
        self.model_names_list = model_names_list
        if TL is not None:
            # get TL for which to analyse results
            self.TL = [TL] if isinstance(TL, int) else TL
        else:
            self.TL = None

        # extract attributes from folder (all info is stores in dicts: each sub-dir is a dict)
        # for ex. when results_dir = "outputs/saved_results/XE":
        # then self.info_dct.keys() = dict_keys(['XE_3', 'XE_6', 'XE_7', 'XE_2', ... , 'XE_4', 'XE_1', 'XE_10'])
        # and self.info_dct['XE_3'].keys() = dict_keys(['TL_5', 'TL_20', 'TL_10']) dep. on the nr of res folders
        self.info_dct = self.get_info_dct()
        self.insts_dct = self.get_instances() if load_instances else {}

    def plot_trajectories(self, inst_id_list: Union[str, List[str]] = None, set_type_list: Union[str, List[str]] = None,
                          model_names_list: List = None, TLs_list: List = None, plot_with_inst=False, save_dir_=None,
                          full_trajectories: bool = True, plot_gaps=True):
        # infer specs for trajectory plotting

        set_type_list = self.info_dct.keys() if set_type_list is None else set_type_list
        print('set_type_lsit', set_type_list)
        for set_type in set_type_list:
            print('plotting trajectories for set_type', set_type)
            set_TL_list = self.info_dct[set_type].keys() if TLs_list is None else TLs_list
            TLs = [TL for TL in set_TL_list if TL in self.info_dct[set_type].keys()]
            for TL in TLs:
                set_model_list = self.info_dct[set_type][TL].keys() if model_names_list is None else model_names_list
                models_list = [modl for modl in set_model_list if modl in self.info_dct[set_type][TL].keys()]
                models_list = [modl for modl in models_list if
                               "running_costs" in self.info_dct[set_type][TL][modl].keys()]
                # for model in models_list:
                if inst_id_list is not None:
                    set_ID_list = [inst_id_list] if isinstance(inst_id_list, int) else inst_id_list
                else:
                    set_ID_list = [id_ for id_ in self.info_dct[set_type][TL][models_list[0]]['running_costs'].keys()]
                if save_dir_ is None:
                    print('self.results_dir.split("/")[-1]', self.results_dir.split("/")[-1])
                    save_dir = os.path.join(self.results_dir, set_type, TL) if self.results_dir.split("/")[
                                                                                   -1] != set_type \
                        else os.path.join(self.results_dir, TL)
                else:
                    save_dir = save_dir_
                assert os.path.exists(save_dir)
                # print('save_dir', save_dir)
                self.plot_traj(c_lists=[self.info_dct[set_type][TL][model]['running_costs'] for model in models_list],
                               t_lists=[self.info_dct[set_type][TL][model]['running_times'] for model in models_list],
                               model_names_list=models_list,
                               TL=TL,
                               save_dir=save_dir,
                               set_name=set_type,
                               inst_ids=set_ID_list,
                               final_costs=[self.info_dct[set_type][TL][model]['final_costs'] for model in models_list],
                               final_runtimes=[self.info_dct[set_type][TL][model]['total_runtimes'] for model in
                                               models_list],
                               plot_inst=plot_with_inst,
                               full_trajectories=full_trajectories,
                               plot_gaps=plot_gaps)

    def plot_metrics_mean(self, metric: str = "wrap", set_type_list: Union[str, List[str]] = None,
                          across_timeLimit: bool = True, across_sets: bool = False, model_names_list: List = None,
                          TLs_list: List = None, save_dir_=None):

        info_dct = self.info_dct["XE"] if list(self.info_dct.keys()) == ["XE"] else self.info_dct

        set_type_list = info_dct.keys() if set_type_list is None else set_type_list
        metric_name = metric.lower() + "_mean"
        across_timeLimit = False if across_sets else across_timeLimit
        set_metric_means = {}
        for set_type in set_type_list:
            set_TL_list = info_dct[set_type].keys() if TLs_list is None else TLs_list
            TLs = [TL for TL in set_TL_list if TL in info_dct[set_type].keys()]
            TLs_sort = [int(TL[3:]) for TL in TLs]
            TLs_sort.sort()
            TLs_sort = [TLs_sort[0]] if across_sets else TLs_sort
            metric_means = {}
            for TL in TLs_sort:
                set_model_list = info_dct[set_type]["TL_" + str(TL)].keys() \
                    if model_names_list is None else model_names_list
                models_list = [modl for modl in set_model_list if
                               modl in info_dct[set_type]["TL_" + str(TL)].keys()]
                models_list = [modl for modl in models_list if metric_name
                               in info_dct[set_type]["TL_" + str(TL)][modl].keys()]

                for modl in models_list:
                    if modl not in metric_means.keys():
                        metric_means[modl] = [(TL, info_dct[set_type]["TL_" + str(TL)][modl][metric_name])]
                    else:
                        metric_means[modl].append((TL, info_dct[set_type]["TL_" + str(TL)][modl][metric_name]))
            if save_dir_ is not None:
                save_dr = save_dir_
            else:
                if self.results_dir.split("/")[-1][:3] == "XE_":
                    save_dr = os.path.join(self.results_dir)
                else:
                    save_dr = os.path.join(self.results_dir, set_type)
            if across_timeLimit:
                plot_metric_across_TL(metric_means, metric.upper(), set_type,
                                      save_dr, TLs_sort)
            else:
                set_metric_means[set_type] = metric_means
        if across_sets:
            plot_metric_across_sets(set_metric_means, metric.upper(),
                                    self.results_dir)

    def plot_metrics_instances(self, metric: str, inst_id_list: Union[str, List[str]] = None,
                               set_type_list: Union[str, List[str]] = None, model_names_list: List = None,
                               TLs_list: List = None, save_dir_=None):

        set_type_list = self.info_dct.keys() if set_type_list is None else set_type_list
        metric_name = metric.lower() + "_scores"
        sorted_ids = []
        for set_type in set_type_list:
            set_TL_list = self.info_dct[set_type].keys() if TLs_list is None else TLs_list
            TLs = [TL for TL in set_TL_list if TL in self.info_dct[set_type].keys()]
            TLs_sort = [int(TL[3:]) for TL in TLs]
            TLs_sort.sort()
            metric_values = {}
            for TL in TLs_sort:
                set_model_list = self.info_dct[set_type]["TL_" + str(TL)].keys() \
                    if model_names_list is None else model_names_list
                models_list = [modl for modl in set_model_list if
                               modl in self.info_dct[set_type]["TL_" + str(TL)].keys()]
                models_list = [modl for modl in models_list if metric_name
                               in self.info_dct[set_type]["TL_" + str(TL)][modl].keys()]
                for modl in models_list:
                    if inst_id_list is None:
                        sorted_ids = [int(i) for i in
                                      self.info_dct[set_type]["TL_" + str(TL)][modl][metric_name].keys()]
                    else:
                        sorted_ids = [int(i) for i in
                                      self.info_dct[set_type]["TL_" + str(TL)][modl][metric_name].keys() if
                                      i in inst_id_list]
                    sorted_ids.sort()
                    metric_values[modl] = [self.info_dct[set_type]["TL_" + str(TL)][modl][metric_name][str(i)] for i in
                                           sorted_ids]
                if save_dir_ is None:
                    if self.results_dir.split("/")[-1][:3] == "XE_":
                        save_dr = os.path.join(self.results_dir, "TL_" + str(TL))
                    else:
                        save_dr = os.path.join(self.results_dir, set_type, "TL_" + str(TL))
                else:
                    save_dr = save_dir_
                plot_scatter_per_instance(metric_values, metric.upper(), sorted_ids, set_type,
                                          save_dir=save_dr)

    def avg_gap_over_time(self, timelimit: int, save_dir, model_name_list: List = None,
                          set_name: str = None, inst_ids: List = None):
        fig, ax = plt.subplots()
        set_name = set_name if set_name is not None else list(self.info_dct.keys())[0]
        inst_ids = list(self.insts_dct.keys()) if inst_ids is None else inst_ids
        TL_results = self.info_dct[set_name].keys()
        TL_results = [int(TL[3:]) for TL in TL_results]
        TL_results.sort()
        TL_results = ["TL_" + str(t_l) for t_l in TL_results]
        model_name_list = self.info_dct[set_name][
            list(TL_results)[-1]].keys() if model_name_list is None else model_name_list
        means_across_time_all = []
        for model_n in model_name_list:
            model_means_across_time, tl_list = [], []
            for tl in TL_results:
                if int(tl[3:]) <= timelimit:
                    model_mean_tl = np.mean(list(self.info_dct[set_name][tl][model_n]['gap_to_bks'].values()))
                    model_means_across_time.append(model_mean_tl)
                    result_tl = int(tl[3:])
                    percentage_time = round((result_tl / timelimit) * 100)
                    tl_list.append(percentage_time)
            means_across_time_all.append(model_means_across_time)
            xi_ticks = list(range(len(tl_list)))
            ax.plot(xi_ticks, model_means_across_time, label=model_n)
        save_name = os.path.join(save_dir, 'gaps_over_time')

        plt.xticks(xi_ticks, tl_list)
        plt.xlabel('time (%) of total budget (' + str(timelimit) + ' sec.)')
        plt.ylabel(' % Gap to BKS')
        set_name = set_name if set_name[-3:] not in ["avg", "upd"] else set_name[:-4]
        title = 'GAP Over % of Time Budget for ' + set_name
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_name + '.pdf')
        plt.show()

    def plot_traj(self, c_lists, t_lists, model_names_list, TL, save_dir, set_name, inst_ids: List,
                  final_costs, final_runtimes, plot_inst=False, full_trajectories=True, plot_gaps=True):
        for inst_id in inst_ids:
            if plot_gaps:
                try:
                    instance_bks = self.insts_dct[inst_id].BKS
                except KeyError:
                    instance_bks = self.insts_dct[int(inst_id)].BKS
            fig, ax = plt.subplots() if not plot_inst else plt.subplots(1, 2, figsize=(8, 4))
            for cs, ts, model_name, final_cost, final_runtime in zip(c_lists, t_lists, model_names_list, final_costs,
                                                                     final_runtimes):
                if cs[inst_id] is not None:
                    if full_trajectories:
                        # append final run_time and final cost
                        if ts[inst_id][-1] < int(TL[3:]):
                            if final_cost[inst_id] <= cs[inst_id][-1]:
                                cs[inst_id].append(final_cost[inst_id])
                            else:
                                cs[inst_id].append(cs[inst_id][-1])
                            ts[inst_id].append(final_runtime[inst_id])
                    if len(cs[inst_id]) > 1 and inst_id in cs.keys():
                        if not plot_inst:
                            if plot_gaps:
                                ax.plot(ts[inst_id], [gap_bks(c, instance_bks) for c in cs[inst_id]],
                                        label=model_name)
                            else:
                                ax.plot(ts[inst_id], cs[inst_id], label=model_name)
                                # ax.set_yscale('log')
                        else:
                            if plot_gaps:
                                ax[1].plot(ts[inst_id], [gap_bks(c, instance_bks) for c in cs[inst_id]], label=model_name)
                            else:
                                ax[1].plot(ts[inst_id], cs[inst_id], label=model_name)
                    # plt.plot(t, c, label=model_name)
                    torch.save((ts[inst_id], cs[inst_id]), save_dir+"traj_"+model_name+".pt")
            if "cvrp_100_uniform" in set_name:
                title_set_name = "Uniform-100"
            if not plot_gaps:
                title = 'Trajectory for Instance ' + inst_id + ' of set ' + title_set_name + ' (TL=' + str(TL)[3:] + ' sec.)'
                plt.xlabel('cumulative runtime (seconds) ')
                plt.ylabel('objective value (total cost)')
            else:
                # title = 'Gaps over Time for Instance ' + inst_id + ' of set ' + title_set_name + ' (TL=' + str(TL)[
                                                                                                       #3:] + ' sec.)'
                title = ''
                plt.xlabel('cumulative runtime (seconds) ')
                plt.ylabel('Gap to BKS (%)')
            if plot_inst:
                self.plot_with_instance(title, inst_id, save_dr=save_dir, fig=fig, ax=ax)
            else:
                if plot_gaps:
                    save_name = os.path.join(save_dir, 'gaps_over_running_time_' + inst_id)
                else:
                    save_name = os.path.join(save_dir, 'plttd_trajectories_' + inst_id)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_name + '.pdf')
                plt.show()

    def plot_with_instance(self, title: str, instance_id: str, save_dr: str, fig=None, ax=None,
                           add_idx=True, inst_set: str = None):
        # fig2, ax = plt.subplots()  # (ax1, ax2) # 1, 2
        # geometry = (1,1,1)
        # ax = fig.add_subplot(index)
        cmap = plt.get_cmap("tab20")
        fig.suptitle(title)
        fig.tight_layout()
        if fig is None:
            fig = plt.figure()
        # print('fig', fig)
        # ax = fig.add_subplot(111)
        try:
            instance = self.insts_dct[instance_id] if inst_set is None else self.insts_dct[inst_set][instance_id]
        except KeyError:
            instance = self.insts_dct[int(instance_id)] if inst_set is None \
                else self.insts_dct[inst_set][int(instance_id)]
        locations = instance.coords if instance.original_locations is None else instance.original_locations
        # print('ax', ax)
        ax[0].scatter(locations[:, 0], locations[:, 1], c='k')
        ax[0].scatter(locations[0, 0], locations[0, 1], c='r', s=7 ** 2, marker='s')  # depot/start node
        if add_idx:
            # add node indices
            for i in range(1, locations.shape[0]):
                ax[0].annotate(i, (locations[i, 0], locations[i, 1]),
                               xytext=(locations[i, 0] + 0.012, locations[i, 1] + 0.012),
                               fontsize='medium', fontweight='roman')
        save_name = os.path.join(save_dr, 'subplot_plttd_trajectories_' + instance_id)
        plt.legend()
        plt.savefig(save_name + '.pdf')
        plt.show()

    def get_info_dct(self) -> Dict:
        main_dir_name = self.results_dir.split("/")[-1]
        sub_main_dir_name = self.results_dir.split("/")[-2]
        make_final_dct, make_sub_final_dct = False, False
        if main_dir_name not in INSTANCE_SET_TYPES:
            make_final_dct = True
        if main_dir_name[:2] == "TL":
            make_sub_final_dct = True
        results_dct = {}
        dir_entry_names = [str(dir_n).split(" ")[1][:-1][1:-1] for dir_n in list(os.scandir(self.results_dir))]
        if self.TL is not None and "TL_" + str(self.TL[0]) in dir_entry_names:
            filter_dir_list = [dir_n for dir_n in list(os.scandir(self.results_dir)) if
                               str(dir_n).split(" ")[1][:-1][1:-1] in ["TL_" + str(tl) for tl in self.TL]]
        else:
            filter_dir_list = os.scandir(self.results_dir)
        for dir_ in filter_dir_list:
            if not dir_.is_file():
                results_dct[dir_.name] = {}
                sub_dir_entry_names = [str(dir_n).split(" ")[1][:-1][1:-1] for dir_n in list(os.scandir(dir_))]
                if self.TL is not None and "TL_" + str(self.TL[0]) in sub_dir_entry_names:
                    filter_sub_dir_list = [dir_n for dir_n in list(os.scandir(dir_)) if
                                           str(dir_n).split(" ")[1][:-1][1:-1] in ["TL_" + str(tl) for tl in self.TL]]
                else:
                    filter_sub_dir_list = os.scandir(dir_)
                # print('["TL_" + str(tl) for tl in self.TL]', ["TL_" + str(tl) for tl in self.TL])
                # print('[str(dir_n).split(" ")[1][:-1][1:-1] for dir_n in list(os.scandir(sub_dir))]',
                #       [str(dir_n).split(" ")[1][:-1][1:-1] for dir_n in list(os.scandir(dir_))])
                # filter_sub_dir_list = [dir_n for dir_n in list(os.scandir(dir_)) if
                #                        str(dir_n).split(" ")[1][:-1][1:-1] in ["TL_" + str(tl) for tl in self.TL]]

                # print('filter_sub_dir_list', filter_sub_dir_list)
                # for sub_dir in os.scandir(dir_):
                for sub_dir in filter_sub_dir_list:
                    if not sub_dir.is_file():
                        results_dct[dir_.name][sub_dir.name] = {}
                        sub_sub_dir_entry_names = [str(dir_n).split(" ")[1][:-1][1:-1] for dir_n in
                                                   list(os.scandir(sub_dir))]
                        if self.TL is not None and "TL_" + str(self.TL[0]) in sub_sub_dir_entry_names:
                            filter_sub_sub_dir_list = [dir_n for dir_n in list(os.scandir(sub_dir)) if
                                                       str(dir_n).split(" ")[1][:-1][1:-1] in ["TL_" + str(tl) for tl in
                                                                                               self.TL]]
                        else:
                            filter_sub_sub_dir_list = os.scandir(sub_dir)
                        # filter_sub_sub_dir_list = [dir_n for dir_n in list(os.scandir(sub_dir)) if
                        #                            not str(dir_n) in ["TL_" + str(tl) for tl in self.TL]]
                        # print('filter_sub_sub_dir_list', filter_sub_sub_dir_list)
                        # for sub_sub_dir in os.scandir(sub_dir):
                        for sub_sub_dir in filter_sub_sub_dir_list:
                            # print('sub_sub_dir', sub_sub_dir)
                            if not sub_sub_dir.is_file():
                                results_dct[dir_.name][sub_dir.name][sub_sub_dir.name] = {}
                            elif sub_sub_dir.name[-4:] == ".pdf":
                                pass
                            elif "avg_info" in sub_sub_dir.name or "models_summary" in sub_sub_dir.name:
                                pass
                            else:
                                # print('sub_sub_dir.name', sub_sub_dir.name)
                                model_name = sub_sub_dir.name.split("_")[-1].split(".")[0]
                                if model_name not in results_dct[dir_.name][sub_dir.name].keys():
                                    results_dct[dir_.name][sub_dir.name][model_name] = extract_data(
                                        os.path.join(self.results_dir, dir_.name, sub_dir.name, sub_sub_dir.name))
                                else:
                                    results_dct[dir_.name][sub_dir.name][model_name].update(
                                        extract_data(os.path.join(self.results_dir, dir_.name, sub_dir.name,
                                                                  sub_sub_dir.name)))
                    elif sub_dir.name[-4:] == ".pdf":
                        pass
                    elif "avg_info" in sub_dir.name or "models_summary" in sub_dir.name:
                        pass
                    else:
                        # print('sub_dir.name', sub_dir.name)
                        model_name = sub_dir.name.split("_")[-1].split(".")[0]
                        if model_name not in results_dct[dir_.name].keys():
                            results_dct[dir_.name][model_name] = extract_data(
                                os.path.join(self.results_dir, dir_.name, sub_dir.name))
                        else:
                            results_dct[dir_.name][model_name].update(
                                extract_data(os.path.join(self.results_dir, dir_.name, sub_dir.name)))
            elif dir_.name[-4:] == ".pdf":
                pass
            elif "avg_info" in dir_.name or "models_summary" in dir_.name:
                pass
            else:
                # print('dir_.name', dir_.name)
                model_name = dir_.name.split("_")[-1].split(".")[0]
                if model_name not in results_dct.keys():
                    results_dct[model_name] = extract_data(os.path.join(self.results_dir, dir_.name))
                else:
                    results_dct[model_name].update(extract_data(os.path.join(self.results_dir, dir_.name)))
        if make_sub_final_dct:
            sub_final_dct = {main_dir_name: results_dct}
            return {
                sub_main_dir_name: sub_final_dct
            }
        elif make_final_dct:
            return {
                main_dir_name: results_dct
            }
        else:
            return results_dct

    def get_instances(self):
        data_type = self.results_dir.split("saved_results")[-1]
        if data_type == "XE_avg":
            # update data_type
            data_type = "XE"
        data_type_set_name = data_type.split("/")[-1] if data_type.split("/")[1] in ['XE', 'XE_avg'] else None
        try:
            # print('DATA_STORE_PATHS[data_type.split("/")[1]]', DATA_STORE_PATHS[data_type.split("/")[1]])
            data_store_path = DATA_STORE_PATHS[data_type.split("/")[1]]
        except KeyError:
            updated_key = data_type.split("/")[1][:-4]
            data_store_path = DATA_STORE_PATHS[updated_key]
        if data_type_set_name is not None:
            data_store_path = os.path.join(data_store_path, data_type_set_name)
        dataset = self.ds_class(
            store_path=data_store_path,
            re_evaluate=True,
            load_bks=True
        )
        # for instance in dataset.data
        return {
            instance.instance_id: instance
            for instance in dataset.data
        }

    def print_scores(self, instance_ids: List[str], set_type_name: str = None, model_n: str = "HGS", tl: str = "TL_10",
                     print_average: bool = True):
        set_type_name = list(self.info_dct.keys())[0] if set_type_name is None else set_type_name
        costs_all, gaps_all, wrap_all, pi_all, bks_all = [], [], [], [], []
        for instance_id in instance_ids:
            print(f"INSTANCE : {instance_id}: \n "
                  f"COST: {self.info_dct[set_type_name][tl][model_n]['final_costs'][instance_id]}, "
                  f"GAP: {round(self.info_dct[set_type_name][tl][model_n]['gap_to_bks'][instance_id], 3)}%, "
                  f"WRAP: {round(self.info_dct[set_type_name][tl][model_n]['wrap_scores'][instance_id], 4)}, "
                  f"PI: {round(self.info_dct[set_type_name][tl][model_n]['pi_scores'][instance_id], 4)}")
            costs_all.append(self.info_dct[set_type_name][tl][model_n]['final_costs'][instance_id])
            gaps_all.append(round(self.info_dct[set_type_name][tl][model_n]['gap_to_bks'][instance_id], 3))
            wrap_all.append(round(self.info_dct[set_type_name][tl][model_n]['wrap_scores'][instance_id], 4))
            pi_all.append(round(self.info_dct[set_type_name][tl][model_n]['pi_scores'][instance_id], 4))
            try:
                bks_all.append(self.insts_dct[instance_id].BKS)
            except KeyError:
                bks_all.append(self.insts_dct[int(instance_id)].BKS)
        if print_average:
            print(f"AVERAGES: \n "
                  f"COST: {np.mean(costs_all)}, "
                  f"GAP: {np.mean(gaps_all)}%, "
                  f"WRAP: {np.mean(wrap_all)}, "
                  f"PI: {np.mean(pi_all)}, "
                  f"BKS COST: {np.mean(bks_all)}")


DATA_STORE_PATHS = {
    "XE": "data/test_data/cvrp/uchoa/XE",
    "X": "data/test_data/cvrp/uchoa/X",
    "cvrp_500_uchoa": "data/test_data/cvrp/uchoa/cvrp50/val_seed123_size512.pkl",
    "cvrp_100_uchoa": "data/test_data/cvrp/uchoa/cvrp100/val_seed123_size512.pt",
    "cvrp_50_uchoa": "data/test_data/cvrp/uchoa/cvrp50/val_seed123_size512.pt",
    "cvrp_100_uniform": "data/test_data/cvrp/uniform/cvrp100/val_seed4321_size512.pkl",
    "cvrp_50_uniform": "data/test_data/cvrp/uniform/cvrp50",
    "XML": "data/test_data/XML",
    "cvrp_100_XML": "data/test_data/cvrp/XML100/subsampled/instances",
}

C_METHODS = ["AM", "MDAM", "POMO", "SGBS", "Savings"]


def extract_data(path):
    if "trajectory" in path:
        inst_id = path.split("/")[-1].split("_")[2]
        value_list = torch.load(path)
        if "times" in path:
            return {"running_times_" + inst_id: value_list}
        if "costs" in path:
            return {"running_costs_" + inst_id: value_list}
    elif "run_avg_results" in path:
        # if "run_avg" in path:
        res_dct = torch.load(path, map_location=torch.device('cpu'))
        print('Getting Problem Instances from: ', path)
        if isinstance(res_dct, dict):
            sol_tuples = res_dct["solutions"]
        else:
            sol_tuples = res_dct
        # if None not in [sol.pi_score for sol in sol_tuples] and None not in [sol.wrap_score for sol in sol_tuples]:
        pi_list = [sol.pi_score for sol in sol_tuples if sol.pi_score is not None]
        wrap_list = [sol.wrap_score for sol in sol_tuples if sol.wrap_score is not None]
        return {
            "final_costs": {str(sol.instance.instance_id): sol.cost for sol in sol_tuples},
            "total_runtimes": {str(sol.instance.instance_id): sol.run_time for sol in sol_tuples},
            "gap_to_bks": {str(sol.instance.instance_id): gap_bks(sol.cost, sol.instance.BKS) for sol in sol_tuples},
            "running_costs": {str(sol.instance.instance_id): sol.running_costs for sol in sol_tuples},
            "running_times": {str(sol.instance.instance_id): sol.running_times for sol in sol_tuples},
            "pi_scores": {str(sol.instance.instance_id): sol.pi_score for sol in sol_tuples},
            "pi_mean": np.mean(pi_list) if pi_list else None,
            "wrap_scores": {str(sol.instance.instance_id): sol.wrap_score for sol in sol_tuples},
            "wrap_mean": np.mean(wrap_list) if wrap_list else None
        }
    else:
        print("Unknown file... returning None")
        return None


def gap_bks(z, z_bks):
    return 100 * (z - z_bks) / z_bks


def plot_metric_across_sets(metric_values: Dict, metric_name: str, save_dir: str):
    fig, ax = plt.subplots()
    set_name_lst = metric_values.keys()
    model_set_metrics = {}
    for i, set_name in enumerate(metric_values):
        for model_name in metric_values[set_name]:
            if model_name not in model_set_metrics.keys() and i == 0:
                model_set_metrics[model_name] = [metric_values[set_name][model_name][0][1]]
            elif model_name not in model_set_metrics.keys() and i > 0:
                pad = [None] * i
                model_set_metrics[model_name] = pad + [metric_values[set_name][model_name][0][1]]
            # elif model_name in model_set_metrics.keys() and i != len(model_set_metrics[model_name]):
            else:
                model_set_metrics[model_name].append(metric_values[set_name][model_name][0][1])
        for model_n in model_set_metrics.keys():
            if len(model_set_metrics[model_n]) - 1 != i:
                model_set_metrics[model_n].append(None)
    for model_name in model_set_metrics.keys():
        ax.plot(np.arange(len(set_name_lst)), model_set_metrics[model_name], label=model_name)
    plt.xticks(np.arange(len(set_name_lst)), set_name_lst)
    plt.xlabel('Data Sets')
    plt.ylabel(metric_name)
    plt.title('Average ' + metric_name + ' values across Instance Sets ')
    save_name = os.path.join(save_dir, 'Average_' + metric_name + '_across_Sets')
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(save_name + '.pdf')
    plt.show()


def plot_metric_across_TL(metric_values: Dict, metric_name: str, set_type: str, save_dir: str, TL_lst: List):
    fig, ax = plt.subplots()
    for i, model_name in enumerate(metric_values):
        model_metrics_lst = []
        for tl in TL_lst:
            tls_models = [metr_val[0] for metr_val in metric_values[model_name]]
            if tl in tls_models:
                model_metrics_lst.append(
                    [metr_val[1] for metr_val in metric_values[model_name] if metr_val[0] == tl][0])
            else:
                model_metrics_lst.append(None)
        ax.plot(np.arange(len(TL_lst)), model_metrics_lst, label=model_name)
    plt.xticks(np.arange(len(TL_lst)), TL_lst)
    plt.xlabel('Time Limits')
    plt.ylabel(metric_name)
    set_type = set_type if set_type[-3:] not in ["avg", "upd"] else set_type[:-4]
    plt.title('Average ' + metric_name + ' for Instance Set ' + set_type)
    save_name = os.path.join(save_dir, 'Average_' + metric_name + '_across_Time_Limits')
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(save_name + '.pdf')
    plt.show()


def plot_scatter_per_instance(metric_values, metric_name, instance_ids, instance_type_set, save_dir):
    fig, ax = plt.subplots()
    for model_name, metrics in metric_values.items():
        ax.scatter(np.arange(len(metrics)), metrics, label=model_name)
    plt.xticks(np.arange(len(instance_ids)), instance_ids)
    plt.xlabel('Instance IDs ')
    plt.ylabel(metric_name)
    plt.title(metric_name + ' values for Instance Set ' + instance_type_set)
    save_name = save_dir + metric_name + '_across_' + instance_type_set + '_instances'
    plt.legend()
    plt.savefig(save_name + '.pdf')
    plt.show()


def average_run_results(path_to_results: str, model_name: str, number_runs: int, save_dir_sols: str = None,
                        save_dir_info: str = None, data_length=None):

    # get plain model name
    all_file_names_in_dir = [file.split("_")[-1].split(".")[-2] for file in glob.glob(path_to_results+"/*.pkl")]
    model_n = [file_n for file_n in all_file_names_in_dir if model_name in file_n][0]

    # get run_results
    run_results = [torch.load(path_to_results + "/run_" + str(
                r) + "_results_" + model_n + ".pkl") for r in np.arange(1, number_runs+1)]

    if data_length is None:
        # for cvrp_100_uniform data have subsamples --> length 128
        data_length = len(run_results[0]["solutions"])
    final_costs, num_vehicles, gaps, pi_scores, wrap_scores, final_runtime, final_runn_t = [], [], [], [], [], [], []
    cost_stds, gap_stds, pi_stds, wrap_stds, n_vehicles_std, runn_t_std = [], [], [], [], [], []
    cost_min, num_vehicles_min, gap_min, pi_min, wrap_min, final_rt_min, runn_t_min = [], [], [], [], [], [], []
    running_costs, running_normed_times = [], []
    run_solutions = [run_res["solutions"] for run_res in run_results]

    # averages over 3 runs for set
    average_run_cost, average_run_gaps, average_run_pi, average_run_wrap = [], [], [], []
    for run_sol in run_solutions:
        average_run_cost.append(np.mean([run_sol[i].cost for i in range(data_length) if run_sol[i].cost != float('inf')]))
        average_run_gaps.append(
            np.mean([gap_bks(run_sol[i].cost, run_sol[i].instance.BKS) for i in range(data_length) if run_sol[i].cost != float('inf')]))
        average_run_pi.append(np.mean([run_sol[i].pi_score for i in range(data_length)]))
        average_run_wrap.append(np.mean([run_sol[i].wrap_score for i in range(data_length)]))

    # averages over 3 runs per instance
    for i in range(data_length):
        final_costs.append(np.mean([r_sol[i].cost for r_sol in run_solutions if r_sol[i].cost != float('inf')]))
        cost_stds.append(np.std([r_sol[i].cost for r_sol in run_solutions if r_sol[i].cost != float('inf')]))
        cost_min.append(np.min([r_sol[i].cost for r_sol in run_solutions if r_sol[i].cost != float('inf')]))
        final_runtime.append(np.mean([r_sol[i].run_time for r_sol in run_solutions if r_sol[i].run_time is not None]))
        final_rt_min.append(np.min([r_sol[i].run_time for r_sol in run_solutions if r_sol[i].run_time is not None]))
        final_runn_t.append(np.mean([r_sol[i].running_times[-1] for r_sol in run_solutions if r_sol[i].running_times is not None]))
        runn_t_std.append(np.std([r_sol[i].running_times[-1] for r_sol in run_solutions if r_sol[i].running_times is not None]))
        runn_t_min.append(np.min([r_sol[i].running_times[-1] for r_sol in run_solutions if r_sol[i].running_times is not None]))
        num_vehicles.append(np.mean([r_sol[i].num_vehicles for r_sol in run_solutions])) # if r_sol[i].num_vehicles != float('inf')
        num_vehicles_min.append(np.min([r_sol[i].num_vehicles for r_sol in run_solutions]))
        n_vehicles_std.append(np.std([r_sol[i].num_vehicles for r_sol in run_solutions ]))
        pi_scores.append(np.mean([r_sol[i].pi_score for r_sol in run_solutions]))
        pi_stds.append(np.std([r_sol[i].pi_score for r_sol in run_solutions]))
        pi_min.append(np.min([r_sol[i].pi_score for r_sol in run_solutions]))
        wrap_scores.append(np.mean([r_sol[i].wrap_score for r_sol in run_solutions]))
        wrap_stds.append(np.std([r_sol[i].wrap_score for r_sol in run_solutions]))
        wrap_min.append(np.min([r_sol[i].wrap_score for r_sol in run_solutions]))
        gaps.append(np.mean([gap_bks(r_sol[i].cost, r_sol[i].instance.BKS) for r_sol in run_solutions if r_sol[i].cost != float('inf')]))
        gap_stds.append(np.std([gap_bks(r_sol[i].cost, r_sol[i].instance.BKS) for r_sol in run_solutions if r_sol[i].cost != float('inf')]))
        gap_min.append(np.min([gap_bks(r_sol[i].cost, r_sol[i].instance.BKS) for r_sol in run_solutions if r_sol[i].cost != float('inf')]))

    avergage_info_sols = [
        RPSolution(
            solution=None,
            cost=final_costs[i],
            num_vehicles=num_vehicles[i],
            pi_score=pi_scores[i],
            wrap_score=wrap_scores[i],
            run_time=final_runtime[i],
            problem="cvrp",
            running_costs=run_solutions[0][i].running_costs,
            running_times=run_solutions[0][i].running_times,
            instance=run_solutions[0][i].instance,
        )
        for i in range(data_length)]

    # average info dict has average value, std and min value
    #          'std_cost': cost_stds[i],
    #          'min_cost': ,
    avergage_info_dicts = [
        {'cost': [final_costs[i], cost_stds[i], cost_min[i]],
         'vehicles': [num_vehicles[i], n_vehicles_std[i], num_vehicles_min[i]],
         'pi_score': [pi_scores[i], pi_stds[i], pi_min[i]],
         'wrap_score': [wrap_scores[i], wrap_stds[i], wrap_min[i]],
         'running_time': [final_runn_t[i], runn_t_std[i], runn_t_min[i]],
         'instance': run_solutions[0][i].instance,
         }
        for i in range(data_length)]

    # average_run_cost, average_run_gaps, average_run_pi, average_run_wrap
    summary_runs = {'average_cost': np.mean(average_run_cost),
                    'std_cost': np.std(average_run_cost),
                    'average_gaps': np.mean(average_run_gaps),
                    'std_gaps': np.std(average_run_gaps),
                    'pi_score': np.mean(average_run_pi),
                    'std_pi': np.std(average_run_pi),
                    'wrap_score': np.mean(average_run_wrap),
                    'std_wrap': np.std(average_run_wrap),
                    }

    if save_dir_info is not None:
        torch.save(avergage_info_dicts, save_dir_info)
    if save_dir_sols is not None:
        torch.save(avergage_info_sols, save_dir_sols)

    return summary_runs


def re_evaluate_results(time_budgets: List[int],
                        model_name_list: list,
                        path_to_res: str,
                        path_to_data: str,
                        save_path: str):
    for time_lim in [240, 100, 50, 10]:
        for model in model_name_list:
            run_res = torch.load("outputs/saved_results/cvrp_100_uchoa/TL_240/run_1_results_" + model + ".pkl")
            re_evaluate("data/test_data/cvrp/uchoa/cvrp100/val_seed123_size512.pt", run_res, time_lim, model, "uchoa",
                        "outputs/saved_results/cvrp_100_uchoa_upd/TL_" + str(
                            time_lim) + "/run_1_results_" + model + ".pkl")


def re_evaluate(data_path: str, run_results: Dict, original_TL, model_name: str, data_coord_dist: str,
                save_path: str = None, wrap_eval: bool = True, normalized_runtimes: bool = True,
                normalized_ds: bool = True, same_tl: bool = True, data_length: int = 128, not_saving=False):
    # normalized runtimes means that the running_times and the final runtime are normalized as used in the
    # PI and WRAP evaluation (so not the actual time seen on respective machine, but normalized to the std machines)

    # get machine specs
    is_add_ls, adjusted_inst_timelimit_ls, bas_ref_ls, normalizing_factor_ls = False, None, None, None
    try:
        run_machine_specs = run_results["machine"]
        number_cpus = int(run_machine_specs["CPU"].split(":")[1])
        print('run_machine_specs["GPU"].split(":")[0] == None', run_machine_specs["GPU"].split(":")[0] is None)
        print('run_machine_specs["GPU"].split(":")[0] == None', run_machine_specs["GPU"].split(":")[0] == 'None')
        used_cpu_only = True if run_machine_specs["GPU"].split(":")[0] == 'None' else False
        print('used_cpu_only', used_cpu_only)
        pass_mark_for_eval, cpu_mark = set_eval_passmark(run_machine_specs["CPU"], run_machine_specs["GPU"])
        # get time_limits
        is_c_method = [c_m for c_m in C_METHODS if c_m in model_name]
        is_add_ls, adjusted_inst_timelimit_ls, bas_ref_ls, normalizing_factor_ls = False, None, None, None
        if is_c_method:
            is_add_ls = True if model_name[-3:] in ["-SA", "TBS", "GLS"] else False
        print('is_c_method', is_c_method)
        print('is_add_ls', is_add_ls)
        device_ = torch.device("cpu") if used_cpu_only else torch.device("cuda")
        adjusted_inst_timelimit, base_ref = adjust_time_limit(original_TL, pass_mark_for_eval, device_, number_cpus)
        normalizing_factor = (pass_mark_for_eval / base_ref)
        if is_add_ls:
            adjusted_inst_timelimit_ls, bas_ref_ls = adjust_time_limit(original_TL, cpu_mark, torch.device("cpu"),
                                                                       number_cpus)
            normalizing_factor_ls = (cpu_mark / bas_ref_ls)
    except TypeError:
        print("already normalized times")
        pass_mark_for_eval, cpu_mark = runner_utils.CPU_BASE_REF_SINGLE, runner_utils.CPU_BASE_REF_SINGLE
        device_ = torch.device("cpu")
        adjusted_inst_timelimit, base_ref = adjust_time_limit(original_TL, pass_mark_for_eval, device_, 1)
        print('adjusted_inst_timelimit', adjusted_inst_timelimit)
        normalizing_factor = (pass_mark_for_eval / base_ref)
        print('normalizing_factor', normalizing_factor)
    # initiate classes
    try:
        solutions = run_results["solutions"]
    except TypeError:
        solutions = run_results
    print('NUMBER OF SOLS IN RES:', len(solutions))
    ds_size = len(solutions)
    ds_class = CVRPDataset(
        store_path=data_path,
        distribution=data_coord_dist,
        dataset_size=ds_size,
        TimeLimit=original_TL,
        load_bks=True,
        load_base_sol=wrap_eval,
        normalize=normalized_ds,
    )
    metric_class = Metrics(BKS=ds_class.bks,
                           passMark=pass_mark_for_eval,
                           TimeLimit_=original_TL,
                           passMark_cpu=cpu_mark,
                           is_cpu_search=is_add_ls,
                           base_sol_results=ds_class.BaseSol if ds_class.BaseSol else None,
                           scale_costs=10000 if os.path.basename(
                               data_path) in runner_utils.NORMED_BENCHMARKS else None,
                           single_thread=True if number_cpus <= 1 else False,
                           cpu=used_cpu_only)

    ds_class.metric = metric_class
    ds_class.adjusted_time_limit = adjusted_inst_timelimit if adjusted_inst_timelimit_ls is None \
        else adjusted_inst_timelimit_ls
    print('ds_class.adjusted_time_limit', ds_class.adjusted_time_limit)
    updated_sols = []
    for sol in solutions[:data_length]:
        print(f'RE-EVALUATE {model_name} FOR INSTANCE: {sol.instance.instance_id}')
        print('SOL.RUNNING_TIMES: ', sol.running_times)
        # check if invalid times (like 0.0)
        if float(0) in sol.running_times:
            print('sol.running_times', sol.running_times)
            upd_vals = []
            for value in sol.running_times:
                if value == float(0):
                    new_val = 0.1
                else:
                    new_val = value
                upd_vals.append(new_val)
            print('updated vals', upd_vals)
            sol = sol.update(running_times=upd_vals)
        if sol.instance.BKS != ds_class.bks[str(sol.instance.instance_id)][0]:
            # update BKS, time_limit in stored CVRPInstance
            sol_inst = sol.instance
            sol_inst_upd = sol_inst.update(BKS=ds_class.bks[str(sol.instance.instance_id)][0],
                                           time_limit=original_TL)
            sol = sol.update(instance=sol_inst_upd)
        final_runtime_adj = min(sol.run_time, ds_class.adjusted_time_limit)
        # updated_final_rt_ = min(sol.run_time, ds_class.adjusted_time_limit)
        # print('sol.running_sols', sol.running_sols)
        if sol.running_sols is None:
            # update running costs, running times acc. to (new) Time Limit
            # costs_up_to_tl, rt_up_to_tl = [], []
            # print('len(sol.running_costs)', len(sol.running_costs))
            # print('len(sol.running_times)', len(sol.running_times))
            assert len(sol.running_costs) == len(sol.running_times), f"Length for running_costs and " \
                                                                     f"running_times is not equal"
            # number_costs_orig = len(sol.running_costs)
            # for i in range(number_costs_orig):
            #     if sol.running_times[i] <= ds_class.adjusted_time_limit:
            #         costs_up_to_tl.append(sol.running_costs[i])
            #         print('costs_up_to_tl', costs_up_to_tl)
            #         rt_up_to_tl.append(sol.running_times[i])
            print('sol before update cost=None', sol)
            # print('sol after update cost=None', sol)
            print('len(sol.running_costs)', len(sol.running_costs))
            print('sol.running_costs[-1]', sol.running_costs[-1])
            print('sol.running_costs[-1]', sol.running_costs[-1])
            print('sol.instance.BKS', sol.instance.BKS)
            print('sol.cost', sol.cost)
            if sol.running_costs[-1] < sol.instance.BKS:
                print('SMALLER RUNN COSTS THAN BKS --> ROUNDING ERROR IN ', model_name)
                # if sol.running_costs[-2] > sol.cost >= sol.instance.BKS:
                #     print('Replace last running cost with final cost')
                #     updated_runn_costs = sol.running_costs
                #     updated_runn_costs[-1] = sol.cost
                #     updated_runn_times = sol.running_times
                # else:
                # print('remove last running costs smaller than final cost.')
                updated_runn_costs, updated_runn_times = [], []
                for runn_c, runn_t in zip(sol.running_costs, sol.running_times):
                    if runn_c > sol.cost:
                        updated_runn_costs.append(runn_c)
                        updated_runn_times.append(runn_t)
                updated_runn_costs.append(sol.cost)
                updated_runn_times.append(sol.running_times[-1])
                sol = sol.update(running_costs=updated_runn_costs, running_times=updated_runn_times)
            # placeholder solution for eval -> final sol stays None, cost will be changed to last running cost
            sol = sol.update(running_costs=sol.running_costs, running_times=sol.running_times, running_sols=None,
                             cost=None, run_time=final_runtime_adj, solution=sol.solution if same_tl else None)
            print('sol.running_costs[-1] AFTER', sol.running_costs[-1])
            print('sol.instance.BKS AFTER', sol.instance.BKS)
            print('sol.cost', sol.cost)
        else:
            # print('sol.running_sols', sol.running_sols)
            sols_up_to_tl = []
            if not same_tl:
                for i in range(len(sol.running_times)):
                    # print('sol.running_times[i]', sol.running_times[i])
                    # print('ds_class.adjusted_time_limit', ds_class.adjusted_time_limit)
                    if sol.running_times[i] <= ds_class.adjusted_time_limit:
                        sols_up_to_tl.append(sol.running_sols[i])
                last_sol = None if not sols_up_to_tl else sols_up_to_tl[-1]
                # print("sols_up_to_tl[-1]", sols_up_to_tl[-1])
            # print('sol.solution', sol.solution)
            sol = sol.update(running_costs=None, running_times=sol.running_times, running_sols=sol.running_sols,
                             cost=None, run_time=final_runtime_adj,
                             solution=sol.solution if same_tl else last_sol)
            # sol.running_sols[-1]
        # update sol with new running values and set final cost, solution and runtime to None before re-eval:
        # print('sols_up_to_tl[0] == sols_up_to_tl[1] == sols_up_to_tl[2]',
        #       sols_up_to_tl[0] == sols_up_to_tl[1] == sols_up_to_tl[2])
        # print('len(sols_up_to_tl)', len(sols_up_to_tl))
        # print('len(costs_up_to_tl)', len(costs_up_to_tl))
        # print('len(rt_up_to_tl)', len(rt_up_to_tl))
        # print('sol.running_costs[-1]', sol.running_costs[-1])
        # print('costs_up_to_tl[-1]', costs_up_to_tl[-1])
        # print('sol.cost', sol.cost)
        # print('sol.running_times[-1]', sol.running_times[-1])
        # print('rt_up_to_tl[-1]', rt_up_to_tl[-1])
        # print('sol.run_time', sol.run_time)
        # print('final_runtime_adj', final_runtime_adj)
        # print('sol.running_sols', sol.running_sols)
        # print('sol.cost after update to None', sol.cost)
        # print('sol.run_time after update to None', sol.run_time)
        # print('before sol.wrap', sol.wrap_score)
        # print('before sol.pi', sol.pi_score)
        updated_sol, summary, new_bks = ds_class.eval_solution(model_name=model_name,
                                                               solution=sol,
                                                               eval_mode="pi" if not wrap_eval else ["pi", "wrap"],
                                                               save_trajectory=False,
                                                               save_trajectory_for=None,
                                                               place_holder_final_sol=True if sol.running_sols is None
                                                               else False)
        print('updated_sol.cost after EVAL', updated_sol.cost)
        print('updated_sol.run_time after EVAL', updated_sol.run_time)
        # print('updated sol after eval', updated_sol)
        if sol.running_sols is not None:
            if sol.running_costs is not None:
                print('len(sol.running_sols)', len(sol.running_sols))
                print('len(updated_sol.running_sols)', len(updated_sol.running_sols))
        # print('len(updated_sol.running_costs)', len(updated_sol.running_costs))
        # print('len(updated_sol.running_times)', len(updated_sol.running_times))
        if sol.running_sols is None:
            # updated_final_rt_ = min(updated_sol.run_time, ds_class.adjusted_time_limit)
            print('same_tl', same_tl)
            # print('updated_sol.running_costs[-1]', updated_sol.running_costs[-1])
            updated_sol = updated_sol.update(cost=updated_sol.running_costs[-1] if updated_sol.running_costs is
                                                                                   not None else float('inf'),
                                             solution=sol.solution if same_tl else None,
                                             run_time=min(updated_sol.run_time, ds_class.adjusted_time_limit))


        new_r_times, is_first = [], True
        if normalized_runtimes:
            if updated_sol.running_times is not None:
                for i in range(len(updated_sol.running_times)):
                    if is_first:
                        normalizing_factor_ = normalizing_factor
                    else:
                        normalizing_factor_ = normalizing_factor if normalizing_factor_ls is None \
                            else normalizing_factor_ls
                    normalized_rt = int(((updated_sol.running_times[i] * normalizing_factor_) * 1000) + .5) / 1000.0
                    if normalized_rt > original_TL:
                        # print('normed_rt bigger than original TL')
                        normalized_rt = round(normalized_rt)
                    new_r_times.append(normalized_rt)
                    is_first = False
                print('SOL_UPDATED.running_times: ', updated_sol.running_times)
                print('new_r_times: ', new_r_times)
                # update final run_time:
                print('updated_sol.run_time', updated_sol.run_time)
                print('AJUSTED PER INST TL: ', ds_class.adjusted_time_limit)
                if updated_sol.run_time <= ds_class.adjusted_time_limit:
                    norm_factor = normalizing_factor if normalizing_factor_ls is None else normalizing_factor_ls
                    updated_final_rt = round(int(((updated_sol.run_time * norm_factor) * 1000) + .5) / 1000.0)
                    updated_final_rt = updated_final_rt if (
                            updated_final_rt < (original_TL * (5 / 100))) else original_TL
                    print('normed UPDATE FINAL RT:', updated_final_rt)
                else:
                    # if normalized_runtimes:
                    updated_final_rt = original_TL
                    print('normed UPDATE FINAL RT where r_t is bigger than RT:', updated_final_rt)

                updated_sol = updated_sol.update(running_times=new_r_times, run_time=updated_final_rt)
                # print('updated sol after normalized runtimes', updated_sol)
            else:
                print("NO SOLUTION FOUND IN TIME LIMIT")
                updated_sol = updated_sol.update(running_times=None, run_time=None)
        updated_sols.append(updated_sol)
        print('BEFORE sol.wrap       ', sol.wrap_score)
        print('AFTER updated_sol.wrap', updated_sol.wrap_score)
        print('BEFORE sol.pi       ', sol.pi_score)
        print('AFTER updated_sol.pi', updated_sol.pi_score)
        print('BEFORE sol.cost       ', sol.cost)
        print('AFTER updated_sol.cost', updated_sol.cost)
        print('BEFORE sol.running_costs       ', sol.running_costs)
        print('AFTER updated_sol.running_costs', updated_sol.running_costs)
        print('BEFORE sol.running_times       ', sol.running_times)
        print('AFTER updated_sol.running_times', updated_sol.running_times)
        if solutions[0].wrap_score is not None:
            print('BEFORE AVERAGE WRAP:', np.mean([sol_.wrap_score for sol_ in solutions]))
        print('AFTER AVERAGE WRAP: ', np.mean([sol_.wrap_score for sol_ in updated_sols]))
        print('UPDATED_SOL', updated_sol)
    run_res_updated = {"machine": run_machine_specs, "solutions": updated_sols}
    if save_path is not None and not not_saving:
        torch.save(run_res_updated, save_path)
    return updated_sols


def set_eval_passmark(cpu_device, gpu_device=None, passmark_version=runner_utils.PASSMARK_VERSION):
    # CPU Mark:
    cpu_machine, cpu_count = cpu_device.split(":")[0], int(cpu_device.split(":")[1])
    cpu_mark = runner_utils.CPU_MACHINES[cpu_machine][0]
    cpu_mark_single = runner_utils.CPU_MACHINES[cpu_machine][1]
    number_cpu_cores = runner_utils.CPU_MACHINES[cpu_machine][2]
    total_threads = runner_utils.CPU_MACHINES[cpu_machine][3]
    # GPU Marks:
    gpu_machine = gpu_device.split(":")[0]
    gpu_count = int(gpu_device.split(":")[1]) if gpu_device.split(":")[1] != 'None' else None
    print('gpu_machine', gpu_machine)
    print('gpu_count', gpu_count)
    if gpu_count is not None:
        g3d_mark = runner_utils.GPU_MACHINES[gpu_machine][0]
        g2d_mark = runner_utils.GPU_MACHINES[gpu_machine][1]
    else:
        g3d_mark, g2d_mark, gpu_count = None, None, 0

    if passmark_version == "v1":
        passMark = runner_utils.get_overall_PassMark_v1(cpu_mark, cpu_mark_single, g3d_mark, g2d_mark, cpu_count,
                                                        total_threads, number_cpu_cores, gpu_count)
        cpu_perf = passMark
    else:
        passMark, cpu_perf = runner_utils.get_seperate_PassMarks(cpu_mark, cpu_mark_single, g3d_mark, g2d_mark,
                                                                 cpu_count, total_threads, number_cpu_cores, gpu_count)
        if passMark is None:
            passMark = cpu_perf

    return passMark, cpu_perf


def adjust_time_limit(original_TL, pass_mark, device, nr_threads=1, passmark_version=runner_utils.PASSMARK_VERSION):
    # TODO: Tmax = n × 240/100 seconds (used in HGS-CVRP)
    #  "Therefore, the smallest instance with 100 clients
    # is run for 4 minutes, whereas the largest instance containing 1000 clients is run for 40 minutes."

    print(f'IN ADJUST TIME LIMIT: device: {device}, pass_mark: {pass_mark}, nr_threads: {nr_threads}, '
          f'passmark_v: {passmark_version}, ')
    if passmark_version == "v1":
        print(f"MACHINE_BASE_REF_v1: {runner_utils.MACHINE_BASE_REF_v1}")
        return np.round(original_TL / (pass_mark / runner_utils.MACHINE_BASE_REF_v1)), runner_utils.MACHINE_BASE_REF_v1
    else:
        if device == torch.device("cuda"):
            print(f"GPU_BASE_REF: {runner_utils.GPU_BASE_REF}")
            return np.round(original_TL / (pass_mark / runner_utils.GPU_BASE_REF)), runner_utils.GPU_BASE_REF
            # return np.round(original_TL / (pass_mark / COMBINED_BASE_REF))
        elif device == torch.device("cpu"):
            CPU_BASE_REF = runner_utils.CPU_BASE_REF_SINGLE if nr_threads == 1 else runner_utils.CPU_BASE_REF_MULTI
            # nr_threads*CPU_BASE_REF_SINGLE
            print(f"CPU_BASE_REF: {CPU_BASE_REF}")
            return np.round(original_TL / (pass_mark / CPU_BASE_REF)), CPU_BASE_REF
        else:
            print(f"Device {device} not known - specify 'cuda' or 'cpu' to adjust Time Limit for Evaluation.")


def feasibility_check(instance: CVRPInstance, solution: List[List], is_running: bool = False,
                      is_denormed: bool = False):
    depot = instance.depot_idx[0]
    coords = instance.coords.astype(int) if is_denormed and isinstance(instance.coords[0][0], np.int64) \
        else instance.coords
    # if self.scale_factor is None else (instance.coords * self.scale_factor).astype(int)
    demands = instance.node_features[:, instance.constraint_idx[0]] if is_denormed \
        else instance.node_features[:, instance.constraint_idx[0]]
    # demands = np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity)
    # print('demands[:10]', demands[:10])
    # * instance.original_capacity).astype(int)
    # np.round(instance.node_features[:, instance.constraint_idx[0]] * instance.original_capacity, 3).astype(int)
    routes = solution if solution else None  # check if solution list is not empty - if empty set to None
    # capacity = instance.original_capacity
    capacity = instance.original_capacity if is_denormed else instance.vehicle_capacity
    # print('capacity', capacity)
    if routes is not None:  # or len(solution) == 0:
        k, cost = 0, 0  # .0
        for r in routes:
            if r:
                if r[0] != depot:
                    r = [depot] + r
                if r[-1] != depot:
                    r.append(depot)
                transit = 0
                source = r[0]
                cum_d = 0
                for target in r[1:]:
                    transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                    cum_d += demands[target]
                    source = target
                if cum_d > capacity + 0.01:
                    if is_running:
                        warnings.warn(f"One of the solutions in the trajectory for instance {instance.instance_id} "
                                      f"is infeasible: {cum_d}>{capacity + 0.01}. Setting cost and k to 'inf'.")

                    else:
                        warnings.warn(f"Final CVRP solution {solution} is infeasible for instance "
                                      f"with ID {instance.instance_id}. Setting cost and k to 'inf'.")
                        warnings.warn(f"cumulative demand {cum_d} surpasses (normalized) capacity "
                                      f"{capacity} for instance with ID {instance.instance_id}.")
                    cost = float("inf")
                    k = float("inf")
                    break
                cost += transit
                k += 1
    else:
        warnings.warn(f"No CVRP solution specified (None). setting cost and k to 'inf'")
        cost = float("inf")
        k = float("inf")

    return cost, k
