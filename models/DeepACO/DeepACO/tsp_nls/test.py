import time
import torch

from net import Net
from aco import ACO
from utils import load_test_dataset
from tqdm import tqdm

EPS = 1e-10
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu(),
        distances=distances.cpu(),
        device='cpu',
        local_search='nls',
    )
    
    results = torch.zeros(size=(len(t_aco_diff),))
    results_all_it = []
    for i, t in enumerate(t_aco_diff):
        best_cost, results_per_it = aco.run(t, inference = True)
        results[i] = best_cost
        results_all_it.append(results_per_it)
    assert len(results) == len(results_all_it) == len(t_aco_diff)
    return results, results_all_it
        
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),))
    results_per_inst = []
    start = time.time()
    for pyg_data, distances in tqdm(dataset):
        results, res_per_it = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        # print('res_per_it[0].size()', res_per_it[0].size())
        # print('res_per_it[0]', res_per_it[0])
        results_per_inst.append(res_per_it[0])
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start, results_per_inst

def main(n_node, model_file, k_sparse = None, n_ants=48, t_aco = None):
    k_sparse = k_sparse or n_node//10
    # t_aco = None or list(range(1,11))
    test_list = load_test_dataset(n_node, k_sparse, device, start_node = 0)
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if device == 'cpu' else device+"+cpu" )

    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(model_file, map_location=device))
    avg_aco_best, duration, results_all = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    torch.save(results_all, "results/instance_res_it_"+str(t_aco[0])+".pt")
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to checkpoint file, default to '../pretrained/tsp_nls/tsp{nodes}.pt'")
    opt = parser.parse_args()
    n_nodes = opt.nodes

    filepath = opt.model or f'../pretrained/tsp_nls/tsp{n_nodes}.pt'
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found!")
        exit(1)
    
    main(n_nodes, filepath, t_aco=[4], k_sparse = 50, n_ants=50)
