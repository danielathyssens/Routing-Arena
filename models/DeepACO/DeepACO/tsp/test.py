import time
import torch
from torch.distributions import Categorical, kl
from d2l.torch import Animator

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_test_dataset

torch.manual_seed(12345)

EPS = 1e-10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    if model:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO(
            n_ants=n_ants,
            heuristic=heu_mat,
            distances=distances,
            device=device
        )

    else:
        aco = ACO(
            n_ants=n_ants,
            distances=distances,
            device=device
        )
        if k_sparse:
            aco.sparsify(k_sparse)

    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    results_all_it = []
    for i, t in enumerate(t_aco_diff):
        best_cost, results_per_it = aco.run(t)
        results[i] = best_cost
        results_all_it.append(results_per_it)
    assert len(results) == len(results_all_it) == len(t_aco_diff)
    return results, results_all_it


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    results_per_inst = []
    start = time.time()
    for pyg_data, distances in dataset:
        results, res_per_it = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        results_per_inst.append(res_per_it[0])
        sum_results += results
    end = time.time()

    return sum_results / len(dataset), end - start, results_per_inst


def main():
    n_ants = 20  #48   # 20  # for TSP100  # 50 for TSP500
    n_node = 20  # 500
    k_sparse = 10  # 20 for TSP100 and n_node//10 for rest
    t_aco = [10]   # 10, 20, 30, 40, 50, 100]
    test_list = load_test_dataset(n_node, k_sparse, device)
    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(f'../pretrained/tsp/tsp{n_node}.pt', map_location=device))
    avg_aco_best, duration, results_all = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    torch.save(results_all, "results/instance_res_it_" + str(t_aco[0]) + ".pt")
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))


##########################################################################################

if __name__ == "__main__":
    main()
