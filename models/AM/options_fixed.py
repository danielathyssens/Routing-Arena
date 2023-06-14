import argparse
import os
import time
import argparse
import torch
from models.AM.utils.functions import parse_softmax_temperature


class fixed_eval_opts():
    def __init__(self,
                 o = None,
                 width: int = 0,
                 decode_strategy: str ='greedy',
                 eval_batch_size: int = 1,
                 max_calc_batch_size: int = 10000,
                 compress_mask: bool = True,
                 softmax_temperature: int = 1):
        self.o = o
        self.width = width
        self.decode_strategy = decode_strategy
        self.eval_batch_size = eval_batch_size     # original default 1024
        self.max_calc_batch_size = max_calc_batch_size
        self.compress_mask = compress_mask        # only when decode_strategy is 'bs' (i.e. beam search)
        self.softmax_temperature = softmax_temperature


class fixed_train_opts():
    def __init__(self,
                 run_name: str = 'run',             # Name to identify the run
                 graph_size: int = 20,              # The size of the problem graph
                 batch_size: int = 512,             # Number of instances per batch during training
                 epoch_size: int = 1280000,         # Number of instances per epoch during training
                 lr_model: float = 1e-4,            # Set the learning rate for the actor network
                 lr_critic: float = 1e-4,           # Set the learning rate for the critic network
                 lr_decay: float = 1.0,             # Learning rate decay per epoch
                 eval_only: bool = False,           # Set this value to only evaluate model
                 seed: int = 1234,                  # Random seed to use
                 max_grad_norm: float = 1.0,        # Max. L2 norm for grad-clipping,default 1.0 (0 to disable clipping)
                 exp_beta: float = 0.8,             # Exponential moving average baseline decay (default 0.8)
                 baseline: str = None,              # Baseline: 'rollout','critic','exponential'. Default: no baseline.
                 bl_alpha: float = 0.05,            # Significance in the t-test for updating rollout baseline
                 bl_warmup_epochs: int = 1,      # Number epochs to warmup the baseline, default None = 1 (rollout)
                 eval_batch_size: int = 1024,       # Batch size to use during (baseline) evaluation
                 checkpoint_encoder: bool = False,  # Set to decrease memory usage by checkpointing encoder
                 shrink_size: int = None,           # Shrink bs if at least this many instances in batch are finished
                 data_distribution: str = None,     # Data distribution to use during training (depends on problem)
                 no_progress_bar: bool = False,     # Disable progress bar
                 log_step: int = 50                 # Log info every log_step steps'
                 ):
        self.run_name = run_name
        self.graph_size = graph_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        assert self.epoch_size % self.batch_size == 0, "Epoch size must be integer multiple of batch size!"
        self.lr_model = lr_model
        self.lr_critic = lr_critic
        self.lr_decay = lr_decay
        self.eval_only = eval_only
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.exp_beta = exp_beta
        self.baseline = baseline
        self.bl_alpha = bl_alpha
        if bl_warmup_epochs is None:
            self.bl_warmup_epochs = 1 if baseline == 'rollout' else 0
        assert (self.bl_warmup_epochs == 0) or (self.baseline == 'rollout')
        self.eval_batch_size = eval_batch_size
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.checkpoint_encoder = checkpoint_encoder
        self.data_distribution = data_distribution
        self.no_progress_bar = no_progress_bar
        self.log_step = log_step

