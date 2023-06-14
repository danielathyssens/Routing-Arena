#
import os
import time
import logging
import shutil
from warnings import warn
from typing import Optional, Dict, Union, List
from operator import attrgetter
from omegaconf import DictConfig, OmegaConf as oc
from copy import deepcopy
from tqdm import tqdm

import math
import random
import numpy as np
import hydra
import torch
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy, IQNPolicy

from models.runner_utils import _adjust_time_limit, print_summary_stats, set_device, set_passMark, eval_inference, \
    get_time_limit
from formats import CVRPInstance, RPSolution
from metrics.metrics import Metrics
from models.runner_utils import NORMED_BENCHMARKS

from data.cvrp_dataset import CVRPDataset
from data.tsp_dataset import TSPDataset
from models.NeuroLS.neuro_ls import to_RPSolution, to_RPInstance

from models.NeuroLS.lib.utils import recursive_str_lookup
from models.NeuroLS.lib.env import VecEnv, RPGraphVecEnv
from models.NeuroLS.lib.routing import eval_rp, RP_TYPES
from models.NeuroLS.lib.routing import RPSolution as RPSol_neurols
from models.NeuroLS.lib.scheduling import eval_jssp, JSSP_TYPES, JSSPSolution
from models.NeuroLS.lib.networks import Model
from models.NeuroLS.lib.utils.tianshou_utils import (
    CheckpointCallback,
    MonitorCallback,
    offpolicy_trainer,
    tester,
    TestCollector
)

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'TSP': TSPDataset,
    'CVRP': CVRPDataset
}


#
class Runner:
    """
    Wraps all setup, training and testing functionality
    of the respective experiments configured by cfg.
    """

    def get_acronym(self, model_name: str):
        acronym = model_name
        if self.cfg.test_cfg.env_kwargs.operator_mode == 'SET':
            acronym = model_name + '_SET'
        elif self.cfg.test_cfg.env_kwargs.operator_mode == 'SELECT_LS':
            acronym = model_name + '_LS'
        elif self.cfg.test_cfg.env_kwargs.operator_mode == 'SELECT_LS+':
            acronym = model_name + '_LS+'

        return acronym

    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)
        oc.set_struct(self.cfg, False)

        self.acronym = self.get_acronym("NeuroLS")
        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # set device
        self.device = set_device(self.cfg)

        # set PassMark for eval
        self.passMark, self.CPU_passMark = set_passMark(self.cfg, self.device, number_threads=1)

        if cfg.run_type in ["val", "test"]:
            # get Time Budget
            self.time_limit = get_time_limit(self.cfg)
            if self.time_limit is not None:
                # get normalized per instance Time Limit
                self.per_instance_time_limit = _adjust_time_limit(self.time_limit, self.passMark, self.device)
                self.machine_info = None
                logger.info(f"Eval PassMark for {self.acronym}: {self.passMark}. "
                            f"Adjusted Time Limit per Instance: {self.per_instance_time_limit}.")
            else:
                self.per_instance_time_limit = None
                self.machine_info = (self.passMark, self.CPU_passMark, self.device, 1, False)
                logger.info(f"Per Instance Time Limit is set for each instance separately after loading data.")

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self._build_problem()
        self._build_env()
        self._build_model()
        self._build_policy()
        self.seed_all(self.cfg.global_seed)
        self._build_collectors()
        self._build_callbacks()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        os.makedirs(self.cfg.checkpoint_save_path, exist_ok=True)
        # log dir
        self.cfg.log_path = os.path.join(self._cwd, self.cfg.log_path)
        os.makedirs(self.cfg.log_path, exist_ok=True)
        # val log dir
        self.cfg.val_log_path = os.path.join(self._cwd, self.cfg.val_log_path)
        os.makedirs(self.cfg.val_log_path, exist_ok=True)

    def init_metrics(self, cfg):
        self.metric = Metrics(BKS=self.ds.bks,
                              passMark=self.passMark,
                              TimeLimit_=self.time_limit,
                              passMark_cpu=self.CPU_passMark,
                              base_sol_results=self.ds.BaseSol if self.ds.BaseSol else None,
                              scale_costs=10000 if os.path.basename(
                                  cfg.data_file_path) in NORMED_BENCHMARKS else None,
                              cpu=False if self.device != torch.device("cpu") else True,
                              verbose=self.debug >= 1)

        self.ds.metric = self.metric
        self.ds.adjusted_time_limit = self.per_instance_time_limit

    def _build_problem(self):
        """Load dataset and create environment (problem state and data)."""

        """Load dataset (problem data)."""
        cfg = self.cfg.copy()

        if cfg.run_type in ["val", "test"]:
            self.ds = self.get_test_set(cfg.test_cfg, cfg.env_kwargs)
        elif cfg.run_type in ["train", "resume"]:
            val_cfg = cfg.val_env_cfg
            val_env_kwargs = cfg.env_kwargs
            self.ds, self.ds_val = self.get_train_val_set(val_cfg, val_env_kwargs)
        else:
            raise NotImplementedError(f"Unknown run_type: '{self.cfg.run_type}' for model {self.acronym}"
                                      f"Must be ['val', 'test', 'train', 'resume']")

    def _build_env(self):
        """Create and wrap the problem environments."""
        self.policy_type = self.cfg.policy.upper()
        env_cfg = self.cfg.env_cfg.copy()
        env_kwargs = self.cfg.env_kwargs.copy()
        env_kwargs['debug'] = self.debug
        clamp = self.cfg.policy_cfg.pop("clamp_reward", False)
        env_kwargs['clamp'] = clamp
        self.env = self._get_env_cl()(
            num_envs=self.cfg.train_batch_size,
            RA_dataset_class=self.ds,
            problem=self.cfg.problem,
            env_kwargs=env_kwargs,
            **env_cfg
        )
        self.env.seed(self.cfg.global_seed)
        # overwrite cfg for validation env
        val_env_cfg = deepcopy(self.cfg.env_cfg)
        val_env_cfg.update(self.cfg.get('val_env_cfg', {}))
        val_env_kwargs = deepcopy(env_kwargs)
        val_env_kwargs.update(self.cfg.get('val_env_kwargs', {}))
        render = self.cfg.get('render_val', False)
        if render:
            assert self.cfg.val_batch_size == 1, f"can only render for test_batch_size=1"
            val_env_kwargs['enable_render'] = True
            val_env_kwargs['plot_save_dir'] = self.cfg.val_log_path

        self._val_env_kwargs = val_env_kwargs.copy()
        self.val_env = self._get_env_cl()(
            num_envs=self.cfg.val_batch_size,
            RA_dataset_class=self.ds_val,
            problem=self.cfg.problem,
            env_kwargs=val_env_kwargs,
            dataset_size=self.cfg.val_dataset_size,
            **val_env_cfg
        )
        self.val_env.seed(self.cfg.global_seed + 1)

    def _get_env_cl(self):
        # dont need additional graph support when using feed forward encoder
        if (
                self.cfg.problem.upper() == "JSSP" or
                "FF" in "".join(recursive_str_lookup(self.cfg.model_cfg.encoder_args))
        ):
            return VecEnv
        return RPGraphVecEnv

    def _build_model(self):
        """Initialize the model and the corresponding learning algorithm."""
        # make sure decoder works with specified env mode
        a_mode = self.cfg.env_kwargs.acceptance_mode
        o_mode = self.cfg.env_kwargs.operator_mode
        p_mode = self.cfg.env_kwargs.position_mode

        env_modes = {
            'acceptance': a_mode,
            'operator': o_mode,
            'position': p_mode,
        }
        self.model = Model(
            problem=self.cfg.problem,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            env_modes=env_modes,
            device=self.device,
            policy_type=self.policy_type,
            **self.cfg.model_cfg
        )
        logger.info(self.model)

    def _build_policy(self):
        """Infer and set the policy arguments provided to the learning algorithm."""
        policy_cfg = self.cfg.policy_cfg.copy()
        lr = self.cfg.optimizer_cfg.get('lr')
        self.optim = self.model.get_optimizer(**self.cfg.optimizer_cfg)
        # policy
        self.train_fn = None
        self.test_fn = None
        policy = None
        self.exploration_noise = policy_cfg.pop("exploration_noise", False)

        if self.policy_type in ["DQN", "IQN"]:
            eps_train = policy_cfg.pop("epsilon", 1.0)
            eps_test = policy_cfg.pop("epsilon_test", 0.0)
            eps_final = policy_cfg.pop("epsilon_final", 0.01)
            frac_epoch_final = policy_cfg.pop("frac_epoch_final", 0.7)
            max_epoch = self.cfg.trainer_cfg.max_epoch
            epoch_final = math.ceil(frac_epoch_final * max_epoch)

            if self.policy_type == "DQN":
                policy = DQNPolicy(model=self.model, optim=self.optim, **policy_cfg)
            elif self.policy_type == "IQN":
                policy = IQNPolicy(model=self.model, optim=self.optim, **policy_cfg)

            def train_fn(num_epoch: int, env_step: int):
                """A hook called at the beginning of training in each epoch."""
                # linear epsilon decay in the first epoch_final epochs
                if num_epoch <= epoch_final:
                    # eps = eps_train - env_step / 1e6 * (eps_train - eps_final)
                    eps = eps_train - num_epoch / epoch_final * (eps_train - eps_final)
                elif num_epoch == epoch_final:
                    eps = eps_final
                    # in final (late intermediate) epoch once reduce lr
                    for pg in policy.optim.param_groups:
                        pg['lr'] = lr * 0.1
                else:
                    eps = eps_final
                policy.set_eps(eps)

            def test_fn(num_epoch: int, env_step: int):
                """A hook called at the beginning of testing in each epoch."""
                policy.set_eps(eps_test)

            self.policy = policy
            self.train_fn = train_fn
            self.test_fn = test_fn

        else:
            raise ValueError(f"unknown policy: '{self.cfg.policy}'")

        # replay buffer
        replay_buffer_cfg = self.cfg.replay_buffer_cfg.copy()
        self.prioritized = replay_buffer_cfg.pop("prioritized", False)
        if self.prioritized:
            self.rp_buffer = PrioritizedVectorReplayBuffer(buffer_num=self.cfg.train_batch_size,
                                                           **replay_buffer_cfg)
        else:
            self.rp_buffer = VectorReplayBuffer(buffer_num=self.cfg.train_batch_size,
                                                **replay_buffer_cfg)

    def _build_collectors(self):
        """Create necessary collectors."""
        self.train_collector = Collector(
            policy=self.policy,
            env=self.env,
            buffer=self.rp_buffer,
            exploration_noise=self.exploration_noise
        )

        size = self.cfg.val_dataset_size + 2 * self.cfg.val_batch_size  # * test_env_kwargs.num_steps
        buf = VectorReplayBuffer(size, self.cfg.val_batch_size)
        # create collector
        self.val_collector = TestCollector(
            policy=self.policy,
            env=self.val_env,
            buffer=buf,
        )

    def _build_callbacks(self):
        """Create necessary callbacks."""
        self.callbacks = {"save_checkpoint_fn": CheckpointCallback(
            exp=self,
            save_dir=self.cfg.checkpoint_save_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.checkpoint_cfg
        ), "monitor": MonitorCallback(
            tb_log_path=self.cfg.tb_log_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.monitor_cfg
        )}

    def save_results(self, result: Dict, run_id: int = 0):
        pth = os.path.join(self.cfg.log_path, "run_" + str(run_id) + "_results.pkl")
        torch.save(result, pth)

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.seed(seed)
        self.val_env.seed(seed + 1)

    def _eval(self,
              solutions: List[Union[RPSol_neurols, JSSPSolution]],
              time_limit: Union[int, float] = None):
        if self.cfg.problem.upper() in RP_TYPES:
            return eval_rp(
                solutions,
                problem=self.cfg.problem,
                strict_feasibility=self.cfg.get("strict_max_num", True),
                time_limit=time_limit  # to update time limit from default 10 seconds in instance
            )
        else:
            raise NotImplementedError(f"evaluation for {self.cfg.problem} not implemented.")

    def train(self, **kwargs):
        """Train the specified model."""
        logger.info(f"start training on {self.device}...")
        results, solutions = offpolicy_trainer(
            problem=self.cfg.problem,
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.val_collector,
            episode_per_test=self.cfg.val_dataset_size,
            batch_size=self.cfg.update_batch_size,
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            verbose=self.debug,
            render_val=self.cfg.render_val,
            **self.callbacks,
            **self.cfg.trainer_cfg,
            **kwargs
        )
        logger.info(f"training finished.")
        logger.info(results)
        solutions, summary = self._eval(solutions)
        self.callbacks['monitor'].save_results({
            "solutions": solutions,
            "summary": summary
        }, 'val_results')
        logger.info(summary)

    def test(self, additional_test_cfg: Optional[Union[DictConfig, Dict]] = None, **kwargs):
        """Test (evaluate) the provided trained model."""

        print('additional_test_cfg', additional_test_cfg)

        # build dataset to get PI / WRAP evaluation capability
        self._build_problem()

        # init metrics class
        if self.cfg.test_cfg.test_env_cfg.data_file_path is not None and self.passMark is not None \
                and self.cfg.test_cfg.eval_type != "simple":
            assert self.device in [torch.device("cpu"), torch.device("cuda")], \
                f"Device {self.device} unknown - set to torch.device() for metric Evaluation " \
                f"or set test_cfg.eval_type to 'simple'"
            self.init_metrics(self.cfg)

        # load model checkpoint
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        logger.info(f"loading model checkpoint: {ckpt_pth}")
        state_dict = torch.load(ckpt_pth, map_location=self.device)

        # get tester_cfg from current (test) cfg file
        test_cfg = self.cfg.test_cfg.copy()
        tb_log_path = os.path.join(os.getcwd(), self.cfg.tb_log_path)
        test_coord_dist = self.cfg.coords_dist
        # get checkpoint cfg and update
        self.cfg.update(state_dict["cfg"])
        # update cfg with additionally provided args
        if additional_test_cfg is not None:
            add_test_cfg = oc.to_container(additional_test_cfg, resolve=True) \
                if isinstance(additional_test_cfg, DictConfig) else additional_test_cfg
            test_cfg.update(add_test_cfg.get('tester_cfg', {}))
            self.cfg.update(add_test_cfg)
        self.cfg.coords_dist = test_coord_dist
        print('self.cfg.coords_dist', self.cfg.coords_dist)

        # check if same size instances
        graph_sizes = [self.ds.data[i].graph_size for i in range(len(self.ds.data))]
        same_size = True if len(set(graph_sizes)) == 1 else False

        if same_size:
            ds_iter = [self.ds]
            ds_size = len(self.ds.data)
        else:
            ds_iter = []
            for i in range(len(self.ds.data)):
                ds = deepcopy(self.ds)
                ds.data = [ds.data[i]]
                ds.data_transformed = [ds.data_transformed[i]]
                ds.size = 1
                ds_iter.append(ds)
            ds_size = 1
            disable_progress_bar = True

        _solutions, results = [], []
        for ds in tqdm(ds_iter, disable=same_size):
            # build env
            test_env_cfg = self.cfg.env_cfg.copy()
            test_env_cfg['fixed_dataset'] = True
            test_env_cfg.update(test_cfg.get('test_env_cfg', {}))
            test_env_kwargs = self.cfg.env_kwargs.copy()
            test_env_kwargs.update(test_cfg.get('env_kwargs', {}))
            clamp = self.cfg.policy_cfg.pop("clamp_reward", False)
            test_env_kwargs['clamp'] = clamp
            render = test_cfg.get('render', False)
            if render:
                assert test_cfg.test_batch_size == 1, f"can only render for test_batch_size=1"
                test_env_kwargs['enable_render'] = True
                test_env_kwargs['plot_save_dir'] = self.cfg.test_log_path
            # if num_iters is not None:
            #     test_env_kwargs['num_steps'] = num_iters
            if test_cfg.time_limit is not None:
                adj_run_time = self.per_instance_time_limit if self.per_instance_time_limit is not None \
                    else ds.data[0].time_limit
                print('ADJUSTED TIME LIMIT', adj_run_time)
                test_env_kwargs['time_limit'] = float(adj_run_time) * 1000  # in milliseconds
                test_env_kwargs['save_trajectory'] = test_cfg.save_trajectory
                test_env_kwargs['report_on_improvement'] = test_cfg.report
            # if isinstance(self.ds, List):
            test_cfg['dataset_size'] = ds_size # ds.size  # self.ds.size

            # build env
            test_env_cfg['data_file_path'] = None
            if test_env_kwargs.sampling_args['graph_size'] != ds.data[0].graph_size:
                print('test_env_kwargs', test_env_kwargs)
                print('ds.data', ds.data)
                print("test_env_kwargs.sampling_args['graph_size']", test_env_kwargs.sampling_args['graph_size'])
                test_env_kwargs.sampling_args['graph_size'] = ds.data[0].graph_size
                test_env_kwargs.sampling_args['k'] = None
            print('test_env_kwargs', test_env_kwargs)
            self.env = self._get_env_cl()(
                num_envs=test_cfg.test_batch_size,
                RA_dataset_class=ds,  # self.ds,
                problem=self.cfg.problem,
                env_kwargs=test_env_kwargs,
                dataset_size=ds_size,  # self.ds.size,
                **test_env_cfg
            )
            # self.env.seed(self.cfg.global_seed + 2)

            # load checkpoint model
            self.policy_type = self.cfg.policy.upper()
            self._build_model()
            try:
                self.model.load_state_dict(state_dict["model"])
            except RuntimeError as e:
                raise RuntimeError(
                    f"modes specified in tester_cfg are different from modes specified during training "
                    f"and action dimensionality is not compatible: \n   {e}")
            self._build_policy()

            # size = self.ds.size + 3 * test_cfg.test_batch_size  # * test_env_kwargs.num_steps
            assert test_cfg.test_batch_size == 1
            size = ds_size # ds.size + 3 * test_cfg.test_batch_size  # * test_env_kwargs.num_steps
            buf = VectorReplayBuffer(size, test_cfg.test_batch_size)
            # create collector
            test_collector = TestCollector(
                policy=self.policy,
                env=self.env,
                buffer=buf,
            )

            # create callback
            monitor = MonitorCallback(
                tb_log_path=tb_log_path,
                metric_key=self.cfg.eval_metric_key,
                **self.cfg.monitor_cfg
            )

            # default to a single run if number of runs not specified
            number_of_runs = self.cfg.number_runs if self.cfg.number_runs is not None else 1
            results_all, stats_all = [], []
            for run in range(1, number_of_runs + 1):
                # run test inference
                logger.info(f"running inference {run}/{number_of_runs}...")
                solutions_ = self.run_inference(test_collector, monitor, render, ds)
                logger.info(f"Starting Evaluation for run {run}/{number_of_runs} "
                            f"with time limit {self.time_limit} for {self.acronym}")
                if same_size:
                    results, summary_per_instance, stats = self.eval_inference(run, number_of_runs, solutions_)
                    results_all.append(results)
                    stats_all.append(stats)
                else:
                    assert number_of_runs == 1, f"Can only evaluate NeuroLS for different sized dataset with 1 run."
                    print('type(solutions_)', type(solutions_))
                    _solutions.extend(solutions_)
            if number_of_runs > 1:
                print_summary_stats(stats_all, number_of_runs)
                # save overall list of results (if just one run - single run is saved in eval_inference)
                if self.cfg.test_cfg.save_solutions:
                    logger.info(
                        f"Storing Overall Results for {number_of_runs} runs in {os.path.join(self.cfg.log_path)}")
                    self.save_results(
                        result={
                            "solutions": results_all,
                            "summary": stats_all,
                        })
        if not same_size:
            self.eval_inference(1, 1, _solutions)

    def run_inference(self, test_collector, monitor, render, dataset):
        logger.info(f"Run-time dependent parameters: {self.device} Device, "
                    f"Adjusted Time Budget: {self.per_instance_time_limit} / instance.")
        results, solutions_, running_sols, running_ts = tester(
            problem=self.cfg.problem,
            policy=self.policy,
            test_collector=test_collector,
            episode_per_test=dataset.size,  # self.ds.size,
            monitor=monitor,
            render=0.0001 if render else 0,  # rendering is deactivated for render=0
            num_render_eps=self.cfg.test_cfg.get('num_render_eps', 1)
        )
        logger.info(results)
        tl = self.per_instance_time_limit if self.per_instance_time_limit is not None else dataset.data[0].time_limit
        solutions_internal, summary = self._eval(solutions_,
                                                 time_limit=tl)
        monitor.save_results({
            "solutions": solutions_internal,
            "summary": summary
        }, 'test_results')
        logger.info(list(summary.values()))
        logger.info(summary)
        solutions_RP = to_RPSolution(solutions_internal, running_sols, running_ts, dataset)  # self.ds.data)
        return solutions_RP

    def eval_inference(self, curr_run: int, number_of_runs: int, RP_solutions: List[RPSolution]):
        return eval_inference(
            curr_run,
            number_of_runs,
            RP_solutions,
            self.ds,
            self.cfg.log_path,
            self.acronym,
            self.cfg.test_cfg,
            self.debug
        )

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        self._build_problem()
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        # print('self.cfg.trainer_cfg', self.cfg.trainer_cfg)
        new_max_epoch = self.cfg.trainer_cfg.max_epoch
        new_start_epoch = self.cfg.trainer_cfg.start_epoch
        assert ckpt_pth is not None
        state_dict = torch.load(ckpt_pth, map_location=self.device)
        self.load_state_dict(state_dict)
        if new_max_epoch > self.cfg.trainer_cfg.max_epoch:
            # print('update max epoch')
            self.cfg.trainer_cfg.max_epoch = new_max_epoch
        if new_start_epoch != 0:
            self.cfg.trainer_cfg.start_epoch = new_start_epoch

        # remove the unnecessary new directory hydra creates
        # new_hydra_dir = os.getcwd()
        # if "resume" in new_hydra_dir:
        #     remove_dir_tree("resume", pth=new_hydra_dir)

        # print('self.cfg.trainer_cfg', self.cfg.trainer_cfg)
        logger.info(f"resuming training from: {ckpt_pth}")
        # self.train(resume_from_log=True, **kwargs) # doesn't work with version from tensorboard logger
        self.train(**kwargs)

    def state_dict(self, *args, **kwargs) -> Dict:
        """Save states of all experiment components
        in PyTorch like state dictionary."""
        return {
            "cfg": self.cfg.copy(),
            "model": self.model.state_dict(*args, **kwargs),
            "optim": self.optim.state_dict(),
            # "rp_buffer": self.rp_buffer.__dict__.copy(),
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load a previously saved state_dict and
        reinitialize all required components.

        Examples:
            state_dict = torch.load(PATH)
            experiment.load_state_dict(state_dict)
        """
        self.cfg.update(state_dict["cfg"])
        self._dir_setup()
        self._build_env()
        self._build_model()
        self.model.load_state_dict(state_dict["model"])
        self._build_policy()
        self.optim.load_state_dict(state_dict['optim'])
        # self.rp_buffer.__dict__.update(state_dict["rp_buffer"])
        self._build_collectors()
        self._build_callbacks()

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup()
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['test', 'val']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'test', 'val', 'debug']")

    def get_test_set(self, test_cfg, env_kwargs):
        if self.cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[self.cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")

        if self.cfg.test_cfg.eval_type != "simple":
            load_bks = True
            if self.cfg.test_cfg.eval_type == "wrap" or "wrap" in self.cfg.test_cfg.eval_type:
                load_base_sol = True
            else:
                load_base_sol = False
        else:
            load_bks, load_base_sol = False, False

        ds = dataset_class(
            store_path=test_cfg.test_env_cfg.data_file_path if 'data_file_path' in
                                                               list(test_cfg.test_env_cfg.keys()) else None,
            distribution=self.cfg.coords_dist,
            graph_size=self.cfg.graph_size,
            dataset_size=test_cfg.dataset_size,
            seed=self.cfg.global_seed,
            verbose=self.debug > 1,
            transform_func=to_RPInstance,
            TimeLimit=self.time_limit if self.cfg.data_file_path is not None else None,
            machine_info=self.machine_info,
            load_bks=load_bks,
            load_base_sol=load_base_sol,
            sampling_args=env_kwargs.sampling_args,
            generator_args=env_kwargs.generator_args
        )
        return ds

    def get_train_val_set(self, val_cfg, val_env_kwargs):
        if self.cfg.problem.upper() in DATA_CLASS.keys():
            dataset_class = DATA_CLASS[self.cfg.problem.upper()]
        else:
            raise NotImplementedError(f"Unknown problem class: '{self.cfg.problem.upper()}' for model {self.acronym}"
                                      f"Must be ['TSP', 'CVRP']")

        ds = dataset_class(
            is_train=True,
            graph_size=self.cfg.graph_size,
            seed=self.cfg.global_seed,
            verbose=self.debug > 1,
            transform_func=to_RPInstance,
            distribution=self.cfg.coords_dist
        )
        ds_val = dataset_class(
            is_train=True,
            store_path=val_cfg.data_file_path if 'data_file_path' in list(val_cfg.keys()) else None,
            num_samples=self.cfg.val_dataset_size,
            distribution=val_env_kwargs.generator_args.coords_sampling_dist,
            graph_size=self.cfg.graph_size,
            transform_func=to_RPInstance,
            device=self.device,
            seed=self.cfg.global_seed,
            verbose=self.debug > 1,
            **val_env_kwargs.generator_args
        )
        return ds, ds_val


def update_path(cfg: DictConfig, fixed_dataset: bool = True):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()
    if fixed_dataset:
        if cfg.val_env_cfg.data_file_path is not None:
            cfg.val_env_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.val_env_cfg.data_file_path)
            )
        if cfg.test_cfg.test_env_cfg.data_file_path is not None:
            cfg.test_cfg.test_env_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.test_env_cfg.data_file_path)
            )
    if cfg.checkpoint_load_path is not None:
        cfg.checkpoint_load_path = os.path.normpath(
            os.path.join(cwd, cfg.checkpoint_load_path)
        )
    if cfg.test_cfg.saved_res_dir is not None:
        cfg.test_cfg.saved_res_dir = os.path.normpath(
            os.path.join(cwd, cfg.test_cfg.saved_res_dir)
        )
    # if cfg.checkpoint_save_path is not None:
    #     cfg.checkpoint_save_path = os.path.normpath(
    #         os.path.join(cwd, cfg.checkpoint_save_path)
    #     )
    return cfg


def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i + len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
