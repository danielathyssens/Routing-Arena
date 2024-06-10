
--boundary_.oOo._8FS1UVS+WXiqrI3JVgR+ASygd5QJzR90
Content-Length: 2083
Content-Type: application/octet-stream
X-File-MD5: 4194d659662111bdfca5ba5bab93870a
X-File-Mtime: 1683731590
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/config/meta/run.yaml

# @package _global_

# tensorboard --host localhost --port 8080 --logdir=./outputs

run_type: "val"
number_runs: 1
debug_lvl: 0  # 0 disables debugging and verbosity completely, >1 activates additional debugging functionality
global_seed: 1234
cuda: True
# eGPU GeForce RTX 3060 (G3DMark Mark: 17177, G2DMark: 975)
G3DMark: 17177
G2DMark: 975
# DELL Machine (Intel® Core™ i7-10850H - 12 Threads): 11980
# DELL Machine (Intel® Core™ i7-10850H - 1 Threads): 2714
CPU_Mark_single: 2714
CPU_Mark: 11980
number_threads: 12
number_cpus: 6

test_cfg:
  eval_type: ['pi', 'wrap']  # 'simple', 'wrap', 'pi', ['pi', 'wrap']
  time_limit: 10
  save_solutions: False   # either save solution in log dir of run output dir
  save_for_analysis: True # or save solutions for further analysis in saved results folder
  saved_res_dir: outputs/saved_results # where results for analysis should be saved
  out_name: ${out_name}
  save_trajectory: True   # will always be saved in log dir of run output dir
  save_traj_for:  # IDs of Instances to plot in list - if None, first instance in dataset is plotted
  fixed_dataset: True
  dataset_size:   # to overwrite the number of instances to be tested in fixed test set (first 5 instances)
  data_file_path: ${data_file_path} # depends on distribution --> env config
  checkpoint_load_path: ${checkpoint_load_path}
  batch_size: 1
  decode_type: 'greedy'  # 'sample'
  sample_size: 0         # if select sample as decode strategy, default is 1280 - else disable and set to 0
  eval_batch_size: 1
  add_ls: True  # boolean to add local search (spec in 'ls_policy_cfg') after construction
  ls_policy_cfg:
    local_search_strategy: 'SIMULATED_ANNEALING' #'SIMULATED_ANNEALING' #'GUIDED_LOCAL_SEARCH'
    solution_limit: #None
    verbose: False
    log_search: False
    batch_size: 1
    search_workers: 1  # only possible to parallelize search if cp_sat solver is enabled in GORT LS

#    time_limit: ${test_cfg.time_limit} #4
# global paths and logging
log_lvl: INFO
tb_log_path: 'logs/tb/'
log_path: 'logs/'
checkpoint_save_path: 'checkpoints/'



--boundary_.oOo._8FS1UVS+WXiqrI3JVgR+ASygd5QJzR90
Content-Length: 2582
Content-Type: application/octet-stream
X-File-MD5: a84c7220489012081a1e8a89e53a6e4f
X-File-Mtime: 1682683315
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/config/meta/train.yaml

# @package _global_

# tensorboard --host localhost --port 8080 --logdir=./outputs

run_type: "train"
debug_lvl: 0  # 0 disables debugging and verbosity completely, >1 activates additional debugging functionality
global_seed: 1234
cuda: True

val_size: 10000                # Number of instances used for reporting validation performance
val_dataset: # None            # Dataset file to use for validation
generator_args: {}


test_cfg:
  data_file_path: # None         # data/test_data/cvrp/uniform/cvrp20_test_seed1234.pkl
  checkpoint_load_path: #None

train_opts_cfg:
  run_name: 'train_run_uchoa100'       # Name to identify the run
  graph_size: ${graph_size}              # The size of the problem graph
  batch_size: 512             # Number of instances per batch during training
  epoch_size: 1280000 # 0 #0         # Number of instances per epoch during training
  n_epochs: 100               # The number of epochs to train
  epoch_start: 0
  checkpoint_epochs: 1        # Save checkpoint every n epochs (default 1), 0 to save no checkpoints
  lr_model: 1e-4              # Set the learning rate for the actor network
  lr_critic: 1e-4             # Set the learning rate for the critic network
  lr_decay: 1.0               # Learning rate decay per epoch
  eval_only: False            # Set this value to only evaluate model
  seed: 1234                  # Random seed to use
  max_grad_norm: 1.0          # Max. L2 norm for grad-clipping,default 1.0 (0 to disable clipping)
  exp_beta: 0.8               # Exponential moving average baseline decay (default 0.8)
  baseline: rollout           # Baseline: 'rollout','critic','exponential'. Default: no baseline.
  bl_alpha: 0.05              # Significance in the t-test for updating rollout baseline
  bl_warmup_epochs: 1         # Number epochs to warmup the baseline, default None = 1 (rollout)
  eval_batch_size: 1024       # Batch size to use during (baseline) evaluation
  checkpoint_encoder: False   # Set to decrease memory usage by checkpointing encoder
  shrink_size: None           # Shrink bs if at least this many instances in batch are finished
  data_distribution: None     # Data distribution to use during training (depends on problem)
  no_progress_bar: False      # Disable progress bar
  log_step: 50                # Log info every log_step steps'
  no_tensorboard: True        # Disable logging TensorBoard files

#    time_limit: ${test_cfg.time_limit} #4
# global paths and logging
log_lvl: INFO
tb_logging: False
tb_log_path: 'logs/tb/'
log_path: 'logs/'
checkpoint_save_path: 'checkpoints/'



--boundary_.oOo._8FS1UVS+WXiqrI3JVgR+ASygd5QJzR90
Content-Length: 495
Content-Type: application/octet-stream
X-File-MD5: 483335969f359be63c9309bdfb637014
X-File-Mtime: 1665590755
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/config/model/attn.yaml

# @package _global_
# model parameters

# POLICY model
# policy_cfg:
#  embedding_dim: 128
model: "AM"

# model params
model_cfg:
  model_type: "Attention"
  model_args:
    hidden_dim: 128
    num_layers: 3
    tanh_clipping: 10.0
    norm_type: "batch"

# eval options
eval_opts_cfg:
  o: None
  width: ${test_cfg.sample_size}
  decode_strategy: ${test_cfg.decode_type}
  eval_batch_size: ${test_cfg.eval_batch_size}
  max_calc_batch_size: 10000
  compress_mask: True
  softmax_temperature: 1

--boundary_.oOo._8FS1UVS+WXiqrI3JVgR+ASygd5QJzR90
Content-Length: 975
Content-Type: application/octet-stream
X-File-MD5: db39082b486c18050a00f0e71669721d
X-File-Mtime: 1644835842
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/nets/critic_network.py

from torch import nn
from models.AM.nets.graph_encoder import GraphAttentionEncoder


class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = GraphAttentionEncoder(
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            normalization=encoder_normalization
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        _, graph_embeddings = self.encoder(inputs)
        return self.value_head(graph_embeddings)

--boundary_.oOo._8FS1UVS+WXiqrI3JVgR+ASygd5QJzR90
Content-Length: 22354
Content-Type: application/octet-stream
X-File-MD5: 50813690fa23e628e5ee7c1781523988
X-File-Mtime: 1642775620
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/nets/attention_model.py

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from models.AM.utils.tensor_functions import compute_in_batches

from models.AM.nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from models.AM.utils.beam_search import CachedLookup
from models.AM.utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows fo