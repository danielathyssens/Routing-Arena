from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any
import math
import os
from tqdm import tqdm
import warnings
import time
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim

from models.base_model import BaseModel
from models.AM.nets.graph_encoder import GraphAttentionEncoder
from models.AM.utils.beam_search import CachedLookup
from models.AM.utils.functions import sample_many
from models.AM.utils.tensor_functions import compute_in_batches
from models.AM.utils.log_utils import log_values
from models.AM.utils import load_model, load_problem, move_to, load_args
from formats import TSPInstance, CVRPInstance, RPSolution


def make_cvrp_instance(args: Union[CVRPInstance, List[CVRPInstance]], distribution_args=None, offset=0):
    # depot, loc, demand, capacity, *args = args
    if not isinstance(args, List):
        depot = args.coords[args.depot_idx[0]]
        loc = args.coords[1:, :]
        demand = args.node_features[1:, args.constraint_idx[0]]
        capacity = args.vehicle_capacity

        grid_size = 1
        if distribution_args is not None:
            depot_types, customer_types, grid_size = distribution_args
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'demand': torch.tensor(demand, dtype=torch.float),   # / capacity -> demands already normalized
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size
        }
    else:
        return [make_cvrp_instance(args_) for args_ in args[offset:offset + len(args)]]

class AttentionModel(BaseModel):
    """Attention Model by Kool et al. (2018)"""

    def __init__(self,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 problem: str = 'tsp',
                 model_dir: str = 'models/AM/',
                 n_encode_layers: int = 3,
                 tanh_clipping: float = 10.,
                 mask_inner: bool = True,
                 mask_logits: bool = True,
                 normalization: str = 'batch',
                 n_heads: int = 8,
                 checkpoint_encoder: bool = False,
                 shrink_size: int = None,
                 is_train: bool = True,  # added by me
                 train_data: Optional[str] = None,  # added by me
                 eval_data: Optional[str] = None,  # added by me
                 val_data_size: int = 10000,  # added by me
                 ckpt_save_dir: Optional[str] = 'models/checkpoints/AM/',  # added by me
                 device: Union[str, int, torch.device] = "cpu",  # added by me
                 n_epochs: int = 100,
                 model_opts: Optional[Dict] = None
                 ):
        super(AttentionModel, self).__init__(model_dir=model_dir,
                                             problem=problem,
                                             is_train=is_train,
                                             train_data=train_data,
                                             eval_data=eval_data,
                                             val_data_size=val_data_size,
                                             ckpt_save_dir=ckpt_save_dir,
                                             device=device,
                                             n_epochs=n_epochs)

        if not self.is_train:
            # from models.AM.options_fixed import fixed_eval_opts
            # fixed_eval = fixed_eval_opts()
            self.eval_opts = model_opts
            print('self.eval_opts', self.eval_opts)
            # if self.decode_type is None:
            # self.set_decode_type("greedy" if self.eval_opts.decode_strategy in ('bs', 'greedy') else
            #                      "sampling", temp=self.eval_opts.softmax_temp)
        else:
            # from models.AM.options_fixed import fixed_train_opts
            self.n_epochs = n_epochs
            self.ckpt_save_dir = ckpt_save_dir
            # fixed_train = fixed_train_opts()
            self.train_opts = model_opts

        # print('self.train_opts', self.train_opts)
        # print('self.device', self.device)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping

        if isinstance(problem, str):
            self.problem = load_problem(problem)
        # else problem already initiated
        else:
            self.problem = problem

        self.allow_partial = self.problem.NAME == 'sdvrp'
        self.is_vrp = self.problem.NAME == 'cvrp' or self.problem.NAME == 'sdvrp'
        self.is_orienteering = self.problem.NAME == 'op'
        self.is_pctsp = self.problem.NAME == 'pctsp'

        self.decode_type = None
        self.temp = 1.0

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert self.problem.NAME == "tsp", "Unsupported problem: {}".format(self.problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y

            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    # New methods
    def _preprocess(self, is_eval, opts):
        if os.path.exists(os.path.join(self.model_path, 'args.json')):
            args = load_args(os.path.join(self.model_path, 'args.json'))
        else:
            print("No args saved for this model checkpoint")
            args = None

        # preliminaries for evaluating on test
        # if is_eval:
        #    self.set_decode_type("greedy" if opts.decode_strategy in ('bs', 'greedy') else
        #                         "sampling", temp=opts.softmax_temp)

    # From Kool et al. (2018)
    def _load(self, state_dict_path):
        # problem = load_problem(self.problem)
        # problem = load_problem(args['problem'])

        # load for testing
        if state_dict_path is not None and not self.is_train:
            # Load the model parameters from a saved state
            load_optimizer_state_dict = None
            print(f'  [*] Loading AM model for Evaluation from {state_dict_path} for {self.problem.NAME}')

            load_params = torch.load(
                os.path.join(
                    os.getcwd(),
                    state_dict_path
                ), map_location=lambda storage, loc: storage)  # Load on CPU

            if isinstance(load_params, dict):
                load_optimizer_state_dict = load_params.get('optimizer', None)
                load_model_state_dict = load_params.get('model', load_params)
            else:
                load_model_state_dict = load_params.state_dict()

            state_dict = self.state_dict()
            # print('state_dict', state_dict.keys())
            state_dict.update(load_model_state_dict)

            self.load_state_dict(state_dict)
            self.set_decode_type("greedy" if self.eval_opts.decode_strategy in ('bs', 'greedy') else
                                 "sampling", temp=self.eval_opts.softmax_temperature)

        # load for training
        else:
            # Load data from load_path
            load_params = {}
            if state_dict_path is not None:
                print(f'  [*] Loading data to resume Training Process from {state_dict_path}')
                load_params = torch.load(
                    os.path.join(
                        os.getcwd(),
                        state_dict_path
                    ), map_location=lambda storage, loc: storage)  # Load on CPU

            if not self.no_cuda and torch.cuda.device_count() > 1:
                # model = torch.nn.DataParallel(model)
                torch.nn.DataParallel(self)

            # Overwrite model parameters by parameters to load
            if isinstance(load_params, dict):
                load_optimizer_state_dict = load_params.get('optimizer', None)
                load_model_state_dict = load_params.get('model', load_params)
            else:
                load_model_state_dict = load_params.state_dict()
            if isinstance(self, DataParallel):
                state_dict = self.module.state_dict()
            else:
                state_dict = self.state_dict()
            # print('state_dict', state_dict.keys())
            state_dict.update(load_model_state_dict)
            self.load_state_dict(state_dict)

            return load_params

    def get_inner(self):
        return self.module if isinstance(self, DataParallel) else self

    def train_model(self, train_dataset, val_dataset, ckpt_save_path, opts, tb_logger, coords_distribution, **sampling_args):

        print('sampling_args', sampling_args)
        optimizer, baseline, lr_scheduler = self.train_prep(distribution=coords_distribution,
                                                            **sampling_args)
        epoch_rewards = []
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_data_samples = train_dataset.sample(sample_size=opts.epoch_size)
            step, avg_epoch_reward = self.train_epoch(
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                train_data_samples,
                val_dataset,
                tb_logger,
                ckpt_save_path
            )
            epoch_rewards.append((step, avg_epoch_reward))

        return epoch_rewards

    def train_prep(self, model_data=None, resume_path=None, distribution=None, **kwargs):
        print('distribution', distribution)
        if model_data is None:
            model_data = {}

        # init baseline
        from models.AM.nets.critic_network import CriticNetwork
        from models.AM.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
        from models.AM.reinforce_baselines import (
            NoBaseline,
            ExponentialBaseline,
            CriticBaseline,
            RolloutBaseline,
            WarmupBaseline)


        if self.train_opts.baseline == 'exponential':
            baseline = ExponentialBaseline(self.train_opts.exp_beta)
        elif self.train_opts.baseline == 'critic' or self.train_opts.baseline == 'critic_lstm':
            assert self.problem == 'tsp', "Critic only supported for TSP"
            baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(  # critic net. has same parameters for embedding and hidden dim
                        2,
                        self.embedding_dim,
                        self.hidden_dim,
                        self.n_encode_layers,
                        self.tanh_clipping
                    )
                    if self.train_opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        self.embedding_dim,
                        self.hidden_dim,
                        self.n_encode_layers,
                        self.normalization
                    )
                ).to(self.device)
            )
        elif self.train_opts.baseline == 'rollout':
            baseline = RolloutBaseline(self, self.problem.NAME, self.train_opts, distribution=distribution, **kwargs)
        else:
            assert self.train_opts.baseline is None, "Unknown baseline: {}".format(self.train_opts.baseline)
            baseline = NoBaseline()

        if self.train_opts.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, self.train_opts.bl_warmup_epochs,
                                      warmup_exp_beta=self.train_opts.exp_beta)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in model_data:
            baseline.load_state_dict(model_data['baseline'])

        # Initialize optimizer
        # print('self.parameters', self.parameters)
        optimizer = optim.Adam(
            [{'params': self.parameters(), 'lr': self.train_opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': self.train_opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

        # Load optimizer state
        if 'optimizer' in model_data:
            optimizer.load_state_dict(model_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: self.train_opts.lr_decay ** epoch)

        # resume training
        if resume_path is not None:
            epoch_resume = int(os.path.splitext(os.path.split(resume_path)[-1])[0].split("-")[1])

            torch.set_rng_state(model_data['rng_state'])
            if not self.no_cuda and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(model_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(self, epoch_resume)

        return optimizer, baseline, lr_scheduler

    def train_epoch(self, optimizer, baseline, lr_scheduler, epoch, train_dataset, val_dataset, tb_logger, save_dir):
        opts = self.train_opts

        print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
        step = epoch * (opts.epoch_size // opts.batch_size)
        start_time = time.time()
        # if not opts.no_tensorboard:
        if tb_logger is not None:
            tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

        # Generate new training data for each epoch
        # training_dataset = baseline.wrap_dataset(self.problem.make_dataset(
        #     size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
        # training_dataset = baseline.wrap_dataset(train_dataset)
        # train_data = self.prep_data(train_dataset)
        training_dataloader = DataLoader(baseline.wrap_dataset(train_dataset),
                                         batch_size=opts.batch_size, num_workers=8)

        # Put model in train mode!
        self.train()
        self.set_decode_type("sampling")

        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
            self.train_batch(
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger
            )

            step += 1

        epoch_duration = time.time() - start_time
        print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

        print('opts.checkpoint_epochs', opts.checkpoint_epochs)
        if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            print('Saving model and state...')
            torch.save(
                {
                    'model': self.get_inner().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': baseline.state_dict()
                },
                os.path.join(save_dir, 'epoch-{}.pt'.format(epoch))
            )

        # val_data = self.prep_data(val_dataset)
        avg_reward = self.validate(val_dataset)

        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)

        # print('baseline', baseline)
        baseline.epoch_callback(self, epoch)

        # lr_scheduler should be called at end of epoch
        lr_scheduler.step()

        return step, avg_reward

    def train_batch(self, optimizer, baseline, epoch, batch_id, step, batch, tb_logger):
        x, bl_val = baseline.unwrap_batch(batch)
        x = move_to(x, self.device)
        bl_val = move_to(bl_val, self.device) if bl_val is not None else None

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = self.forward(x)

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        loss = reinforce_loss + bl_loss

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, self.train_opts.max_grad_norm)
        optimizer.step()

        # Logging
        if step % int(self.train_opts.log_step) == 0:
            log_values(cost, grad_norms, epoch, batch_id, step,
                       log_likelihood, reinforce_loss, bl_loss, tb_logger, self.train_opts.baseline)

    def validate(self, dataset):
        # Validate
        print('Validating...')
        cost = self.rollout(dataset)
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))

        return avg_cost

    def rollout(self, dataset):
        # Put in greedy evaluation mode!
        self.set_decode_type("greedy")
        self.eval()

        def eval_model_bat(bat):
            with torch.no_grad():
                # call model forward pass
                cost, _ = self.forward(move_to(bat, self.device))
            return cost.data.cpu()

        return torch.cat([
            eval_model_bat(bat)
            for bat
            in tqdm(DataLoader(dataset, batch_size=self.train_opts.eval_batch_size),
                    disable=self.train_opts.no_progress_bar)
        ], 0)

    def _eval_all(self, instance_loader):
        # assert self.eval_opts.o is None or (len(self.eval_opts.datasets) == 1 and len(self.eval_opts.width) <= 1), \
        # assert self.eval_opts.o is None or len(self.eval_opts.width) <= 1, \
        #     "Cannot specify result filename with more than one dataset or more than one width"

        widths = self.eval_opts.width if self.eval_opts.width is not None else [0]
        costs_per_width = []
        costs = None
        for width in widths:
            costs = [self._eval_instance(instance, width) for instance in instance_loader]
            costs_per_width.append(costs)
        if len(costs_per_width) == 1 and costs is not None:
            return costs
        else:
            return costs_per_width

    def prep_data(self, dat: Union[List[TSPInstance], List[CVRPInstance]], offset=0):
        """preprocesses data format for AttentionModel (i.e. from List[NamedTuple] to List[torch.Tensor])"""
        if self.problem.NAME == "tsp":
            return [torch.FloatTensor(row.coords) for row in (dat[offset:offset+len(dat)])]
        elif self.problem.NAME == "cvrp":
            return [make_cvrp_instance(args) for args in dat[offset:offset + len(dat)]]
        else:
            raise NotImplementedError

    def _make_RPSolution(self, sols, costs, times, instances) -> List[RPSolution]:
        """Parse model solution back to RPSolution for consistent evaluation"""
        # transform solution torch.Tensor -> List[List]
        sol_list = [self._get_sep_tours(sol_) for sol_ in sols]

        return [
            RPSolution(
                solution=sol_list[i],
                cost=costs[i],
                num_vehicles=len(sol_list[i]),
                run_time=times[i],
                problem=self.problem.NAME,
                instance=instances[i],
            )
            for i in range(len(sols))
        ]

    def _get_sep_tours(self, tours):

        if self.problem.NAME == 'tsp':
            # if problem is TSP - only have single tour
            return tours.tolist()[0]

        elif self.problem.NAME == 'cvrp':
            # print('tours: ', tours)
            it = iter(tours[0])
            tours_list_k = [[0, next(it).item()]]
            for ele in it:
                if ele != 0:
                    tours_list_k[-1].append(ele.item())
                    # print(tours_list_k[-1])
                else:
                    tours_list_k[-1].append(0)
                    tours_list_k.append([ele.item()])
                # print(tours_list_k)
            tours_list_k[-1].append(0)
            # tours_list_k = tours_list_k[:-1]
            # tours_list_k[-1] = tours_list_k[-1][:-1]
            # print(f'tours_list_k: {tours_list_k}')
            return tours_list_k

    def _eval_instance(self, instance, width=0):
        instance = move_to(instance, self.device)
        if self.eval_opts.width != 0:
            width = self.eval_opts.width
        # model, _ = load_model(opts.model, device=self.device, is_eval=True, opts=opts)
        # self.set_decode_type("greedy")
        with torch.no_grad():
            if self.eval_opts.decode_strategy in ('sample', 'greedy'):
                if self.eval_opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert self.eval_opts.eval_batch_size <= self.eval_opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * self.eval_opts.eval_batch_size > self.eval_opts.max_calc_batch_size:
                    assert self.eval_opts.eval_batch_size == 1
                    assert width % self.eval_opts.max_calc_batch_size == 0
                    batch_rep = self.eval_opts.max_calc_batch_size
                    iter_rep = width // self.eval_opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, cost = self.sample_many(instance, batch_rep=batch_rep, iter_rep=iter_rep)
                # , ds
                batch_size = len(cost)
                ids = torch.arange(batch_size, dtype=torch.int64, device=cost.device)
            else:
                assert self.eval_opts.decode_strategy == 'bs'

                cum_log_p, sequences, cost, ids, batch_size = self.beam_search(
                    instance, beam_size=width,
                    compress_mask=self.eval_opts.compress_mask,
                    max_calc_batch_size=self.eval_opts.max_calc_batch_size
                )
            return sequences, cost

    # original model methods
    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand',)
            elif self.is_orienteering:
                features = ('prize',)
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP

            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2,
                                                                                       embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        if self.is_vrp and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )
