import math
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
# from tensorboard_logger import Logger as TbLogger
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, NamedTuple, Any
from data.tsp_dataset import TSPDataset
from data.cvrp_dataset import CVRPDataset
from data.cvrptw_dataset import CVRPTWDataset
from formats import RPSolution


class BaseModel(nn.Module):
    """Model wrapper which is inherited by the benchmark models to unify processes such as saving checkpoints,
        loading models, initialising training and evaluation

        Args:
        model: model to train or evaluate - given is either the model class to train or the model checkpoint arguments
        problem: String value of the problem the model should solve ('tsp', 'VRP, ...)
        train: Boolean value to indicate whether the model is evaluated (if False) or trained (if True)
        data_path: String value giving the path to the train or evaluation data respectively
    """

    def __init__(self,
                 graph_size: int = 20,
                 model_dir: Optional[str] = None,
                 problem: Optional[str] = None,
                 is_train: bool = True,
                 train_data: Optional[list] = None,
                 eval_data:  Optional[list] = None,
                 val_data_size: int = None,
                 ckpt_save_dir: str = None,
                 device: Union[str, int, torch.device] = None,
                 no_cuda: bool = False,
                 n_epochs: int = 100,
                 tensor_board_logging: bool = False):
        super(BaseModel, self).__init__()

        # model: Optional[dict] = None,
        self.graph_size = graph_size
        self.model_dir = model_dir
        self.problem = problem
        self.is_train = is_train
        if self.is_train:
            self.data_path = train_data
            self.val_data = eval_data
            self.val_size = val_data_size
        else:
            self.data_path = eval_data
            self.val_data = None
        self.ckpt_save_dir = 'models/checkpoints/' if ckpt_save_dir is None else ckpt_save_dir
        self.start_epoch_train = 0  # relevant for learning rate decay
        self.n_epochs = n_epochs  # how many epochs to train
        self.tensor_board_logging = tensor_board_logging
        self.no_cuda = no_cuda

        if device is not None:
            self.device = device
        elif not torch.cuda.is_available() or self.no_cuda:
            self.device = torch.device("cpu")
        # elif torch.cuda.is_available() and not self.no_cuda:
        #    self.device = 0
        else:
            self.device = torch.device("cuda:0")

        if problem == "tsp":
            self.dataset_class = TSPDataset
        elif problem == "cvrp":
            self.dataset_class = CVRPDataset
        elif problem == "cvrptw":
            self.dataset_class = CVRPTWDataset
        else:
            print("There exists no dataset class yet for this problem!")
            self.dataset_class = None

    def preprocess(self):
        """can be called by model class to perform model-specific preprocessing: _preprocess"""
        self._preprocess()
        pass

    def load(self, model_path, epoch=None):
        """load model parameters (pytorch state_dict object)"""
        if os.path.isfile(model_path):
            model_filename = model_path
            model_path = os.path.dirname(model_filename)
        elif os.path.isdir(model_path):
            if epoch is None:
                epoch = max(
                    int(os.path.splitext(filename)[0].split("-")[1])
                    for filename in os.listdir(model_path)
                    if os.path.splitext(filename)[1] == '.pt'
                )
            model_filename = os.path.join(model_path, 'epoch-{}.pt'.format(epoch))
        else:
            assert False, "{} is not a valid directory or file".format(model_path)

        # Load model parameters to respective model class
        _ = self._load(model_filename)
        self.to(self.device)

        # in evaluation set model class to specified device and in eval() mode (same for all models)
        if not self.is_train:
            self.eval()

    def save(self):
        """save model parameters after training (pytorch state_dict object)"""
        pass

    def _train(self,
               train_data_path=None,
               val_data_path=None,
               val_size=None,
               model_load_path=None,
               resume=None,
               epoch_start=None,
               n_epochs=None,
               checkpoint_epochs=0,
               checkpoint_dir=None,
               tensor_board=None,
               run_name=None):

        """call training process of method

            Args:
            train_data_path: data to train on - either given in a path or as keyword --> so need to be generated
            val_data_path: data to validate on - either path or generated (default: follows same distribution as train)
            val_size (int): how many samples to validate on
            model_load_path (str): where to load model state dict from (either this or resume needs to be None)
            resume (int): from which epoch to resume training, if "None" start to train from "epoch_start"
            epoch_start (int): from which epoch to (re-)start training
            n_epochs (int): how many epochs to train
            checkpoint_epochs (int): after how many epochs to save a model checkpoint - 0: no checkpoints are saved
            checkpoint_dir (str): where checkpoints are saved
            tensor_board (bool): whether tensorboard logging is active
            run_name (str): to give a custom name to the training run (else it will be just "run_%Y%m%dT%H%M%S")
        """

        start_ep = epoch_start if epoch_start is not None else self.start_epoch_train
        train_eps = n_epochs if n_epochs is not None else self.n_epochs
        ckpt_dir = checkpoint_dir if checkpoint_dir is not None else self.ckpt_save_dir
        tb_log = tensor_board if tensor_board is not None else self.tensor_board_logging
        train_path = train_data_path if train_data_path is not None else self.data_path
        val_path = val_data_path if val_data_path is not None else self.data_path
        val_size = val_size if val_size is not None else self.val_size
        # set name for current run:
        if run_name is not None:
            # custom name for this run; "<run_name>_%Y%m%dT%H%M%S"
            run_name = "{}_{}".format(run_name, time.strftime("%Y%m%dT%H%M%S"))
        else:
            # default name for logs, outputs etc. is "run_%Y%m%dT%H%M%S"
            run_name = "{}_{}".format('run', time.strftime("%Y%m%dT%H%M%S"))

        # save dir for this run ("outputs")
        save_dir = os.path.join(self.model_dir, 'outputs', "{}_{}".format(self.problem, self.graph_size), run_name)

        # Optionally configure tensorboard
        try:
            tb_logger = TbLogger(os.path.join(self.model_dir, 'logs', "{}_{}".format(self.problem, self.graph_size),
                                              run_name)) if tb_log else None
        except:
            tb_logger = None

        # Figure out what's the problem
        # problem = load_problem(self.problem)

        # Initialize/Load model --> # Overwrite model parameters by parameters to load
        # Load model parameters and optimizer state
        assert model_load_path is None or resume is None, "Only one of load path and resume can be given"
        load_path = model_load_path if model_load_path is not None else resume
        print('load_path in _train', load_path)
        load_model_data = self._load(load_path)

        # Train data
        # --> generated on the fly for most models except self.model_dir == 'PIM'
        # init data_loader?

        # Load val data
        val_dataset = self.dataset_class(store_path=val_path, is_train=True, num_samples=val_size)
        # val_dataset = problem.make_dataset(
        #    size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
        #    distribution=opts.data_distribution)

        # FOR SOME MODELS: preprocessing for train
        # --> # Initialize baseline,
        # --> # Load baseline from data,
        # --> # Initialize optimizer,
        # --> # Load optimizer state
        # --> # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        optimizer, baseline, lr_scheduler = self._train_prep(model_data=load_model_data, resume_path=resume)

        if resume is not None:
            epoch_resume = int(os.path.splitext(os.path.split(resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            start_ep = epoch_resume + 1

        # Start the actual training loop
        for epoch in range(start_ep, start_ep + train_eps):

            # train epoch
            start_time = time.time()
            step = self.train_epoch(optimizer, baseline, epoch, tb_logger)
            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

            # save checkpoints?
            if (checkpoint_epochs != 0 and epoch % checkpoint_epochs == 0) or epoch == train_eps - 1:
                print(f'Saving model and state...')
                torch.save(
                    {
                        'model': self.state_dict() if not isinstance(self, DataParallel) else self.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(ckpt_dir, 'epoch-{}.pt'.format(epoch))
                )

            # call the models own validation function
            avg_reward = self.validate(val_dataset)

            # if not opts.no_tensorboard:
            if tb_logger is not None:
                tb_logger.log_value('val_avg_reward', avg_reward, step)

            baseline.epoch_callback(self, epoch)

            # lr_scheduler should be called at end of epoch
            lr_scheduler.step()

    def _eval(self, data_=None, eval_batch_size=1,
              eval_all=False, no_progress_bar=False) -> Tuple[Dict, List[RPSolution]]:
        # , no_progress_bar=
        """call evaluation process of method"""
        if self.dataset_class is not None and data_ is not None:
            # if isinstance(data_,class)
            # dataset = self.dataset_class(store_path=data_)
            data_rp = data_
        elif self.dataset_class is not None and self.data_path is not None:
            dataset = self.dataset_class(store_path=self.data_path)
            data_rp = dataset.data
        else:
            print("No data to evaluate - specify a dataset in _eval() or in model class initiation")
            data_rp = None
        # initiate dataloader from dataset
        # data_rp = data_rp[:2]
        data = self.prep_data(data_rp)
        dataloader = DataLoader(data, batch_size=eval_batch_size)
        # evaluate all instances in eval/test set
        if eval_all and isinstance(data, list):
            costs = self._eval_all(dataloader)
            return costs
        else:
            if isinstance(data, list) and len(data) == 1:
                cost = self._eval_instance(torch.tensor(data))
                return cost
            else:
                costs, sols, times = [], [], []
                for instance in tqdm(dataloader, disable=no_progress_bar):
                    st_time = time.time()
                    sol, cost = self._eval_instance(instance)
                    times.append(time.time() - st_time)
                    costs.append(cost.item())
                    sols.append(sol)
                solutions = self._make_RPSolution(sols, costs, times, data_rp)
                return {}, solutions
