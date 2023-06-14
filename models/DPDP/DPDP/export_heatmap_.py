import os

import torch
from torch.autograd import Variable

# Remove warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

import time
from datetime import timedelta

# from models.DPDP.DPDP.problems.vrp.vrp_reader import VRPReader
from models.DPDP.DPDP.heatmap_utils import VRPReader
from models.DPDP.DPDP.problems.tsp.tsp_reader import TSPReader
from models.DPDP.DPDP.problems.tsptw.tsptw_reader import TSPTWReader
from tqdm import tqdm


# parser = argparse.ArgumentParser(description='Export heatmap')
# parser.add_argument('-c','--config', type=str)
# parser.add_argument('--problem', type=str, default='tsp')
# parser.add_argument('--checkpoint', type=str, required=True)
# parser.add_argument('--instances', type=str, required=True)
# parser.add_argument('-o', '--output_filename', type=str)
# parser.add_argument('--batch_size', type=int, default=10)
# parser.add_argument('--no_prepwrap', action='store_true', help='For backwards compatibility')
# parser.add_argument('-f', action='store_true', help='Force overwrite existing results')
# args = parser.parse_args()

def generate_heatmap(problem, net, config, instance, batch_size=1, do_prepwrap=False):
    # Export heatmaps
    if torch.cuda.is_available():
        # print("CUDA available, using GPU")
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        # print("CUDA not available")
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    # Set evaluation mode
    net.eval()

    batch_size = batch_size
    # print('config ', config)
    num_nodes = len(instance['loc'][0])  # config.num_nodes
    num_neighbors = config.num_neighbors
    beam_size = config.beam_size

    # Heatmaps can make sense for clusters as well if we simply want to cache the predictions
    # assert config.variant == "routes", "Heatmaps only make sense for routes"
    # instance_filepath = args.instance
    if problem == 'cvrp':
        reader = VRPReader(num_nodes, num_neighbors, batch_size, instance)
    else:
        DataReader = DataReader = TSPTWReader if problem == 'tsptw' else TSPReader
        reader = DataReader(num_nodes, num_neighbors, batch_size, instance, do_prep=not do_prepwrap)

    assert len(
        reader.filedata) % batch_size == 0, f"Number of instances {len(reader.filedata)} must be divisible by batch size {batch_size}"

    dataset = iter(reader)

    all_prob_preds = []
    start = time.time()
    # tqdm(
    # total=reader.max_iter)
    for i, batch in enumerate(dataset):

        with torch.no_grad():
            if problem in ('tsp', 'tsptw') and do_prepwrap:
                # Convert batch to torch Variables
                x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
                x_nodes_timew = Variable(torch.FloatTensor(batch.nodes_timew).type(dtypeFloat),
                                         requires_grad=False) if problem == 'tsptw' else None

                # Forward pass
                with torch.no_grad():
                    y_preds, loss, _ = net.forward(x_nodes_coord, x_nodes_timew)
            else:
                # Convert batch to torch Variables
                x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
                x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
                x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
                x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)

                # Forward pass
                with torch.no_grad():
                    y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord)

            prob_preds = torch.log_softmax(y_preds, -1)[:, :, :, -1]

            all_prob_preds.append(prob_preds.cpu())
    end = time.time()
    duration = end - start
    # device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # print("Took", timedelta(seconds=int(duration)), "s on ", device_count, "GPUs")
    heatmaps = torch.cat(all_prob_preds, 0)
    # os.makedirs(heatmap_dir, exist_ok=True)
    # save_dataset((heatmaps.numpy(), {'duration': duration, 'device_count': device_count, 'args': args}), heatmap_filename)
    # print("Saved", len(heatmaps), "heatmaps to", heatmap_filename)
    return heatmaps.numpy(), duration
