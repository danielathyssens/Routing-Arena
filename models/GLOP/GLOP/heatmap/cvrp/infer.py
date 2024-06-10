import sys
import torch
sys.path.insert(0, './')
from models.GLOP.GLOP.heatmap.cvrp.inst import gen_pyg_data
from models.GLOP.GLOP.heatmap.cvrp.train import infer_heatmap
from models.GLOP.GLOP.nets.partition_net import Net

EPS = 1e-10

K_SPARSE = {
    100: 10,  # added
    1000: 100,
    2000: 200,
    5000: 200,
    7000: 200
}
def load_partitioner(n, device, ckpt_path, k_sparse=None, depth=None):
    print('n before', n)
    n_ = n if n in K_SPARSE.keys() else 100
    print('n_', n_)
    k_sparse = K_SPARSE[n_] if k_sparse is None else k_sparse
    depth = 12 if depth is None else depth
    net = Net(48, 3, k_sparse, 2, depth=depth)
    ckpt_path = f'./pretrained/Partitioner/cvrp/cvrp-{n_}.pt' if ckpt_path == '' else ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    print('  [*] Loading model from {}'.format(ckpt_path))
    if ckpt.get('model_state_dict', None):
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    return net.to(device)

@torch.no_grad()
def infer(model, coors, demand, capacity, k_sparse=None, is_cvrplib=False):
    model.eval()
    n = demand.size(0)-1 if demand.size(0)-1 in K_SPARSE.keys() else 100
    k_sparse = K_SPARSE[n] if k_sparse is None else k_sparse
    pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse, cvrplib=is_cvrplib)
    heatmap = infer_heatmap(model, pyg_data)
    return heatmap
