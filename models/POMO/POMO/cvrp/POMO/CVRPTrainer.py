
import torch
from logging import getLogger

from .CVRPEnv import CVRPEnv as Env
from .CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from data.cvrp_dataset import CVRPDataset

from ...utils.utils import *

# changed input of CVRPTrainer to be the initialised env, model, optimizer instead of parameters
# added CVRPDataset to sample batches from new data generators
# added new function to transform CVRPDataset data to POMO compatible input
# source code from https://github.com/yd-kwon/POMO/tree/835c8c06248ade886856f7f5d207fca3f6c63575
class CVRPTrainer:
    def __init__(self,
                 env,
                 generate_problems,
                 model,
                 optimizer,
                 scheduler,
                 trainer_params,
                 USE_CUDA):

        # save arguments
        # self.env_params = env_params
        # self.model_params = model_params
        # self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        # self.result_folder = get_result_folder()
        self.result_folder = trainer_params.checkpoint_save_path
        self.result_log = LogData()

        # cuda
        # USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            print('cuda_device_num', cuda_device_num)
            try:
                torch.cuda.set_device(cuda_device_num)
                self.device = torch.device('cuda', cuda_device_num)
                torch.set_default_tensor_type('torch.cuda.FloatTensor')  # deprecated
                # torch.set_default_dtype(torch.cuda.FloatTensor)
            except AttributeError:
                if torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                    torch.set_default_device("mps")
                    torch.set_default_dtype(torch.float32)
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')  # deprecated
            # torch.set_default_dtype(torch.FloatTensor)

        # Main Components
        # self.model = Model(**self.model_params)
        # self.env = Env(**self.env_params)
        # self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        # self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.env = env
        self.model = model.to(device=self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generate_problems = generate_problems

        # new data generating component
        # self.dataset = CVRPDataset

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay # moved after optimizer.step() -- deprecation
            # self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = int(self.trainer_params.train_episodes)
        # ['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size
            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        # new boolean argument "generate_problems" to generate different distributions of data
        self.env.load_problems(batch_size, generate_problems=self.generate_problems, device=self.device)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        self.scheduler.step()
        return score_mean.item(), loss_mean.item()
