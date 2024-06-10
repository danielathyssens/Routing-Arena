
import torch

import os
from logging import getLogger
from typing import List
from formats import TSPInstance

from .TSPEnv import TSPEnv as Env
from .TSPModel import TSPModel as Model

from ...utils.utils import *


class TSPTester:
    def __init__(self,
                 # env_params,
                 # model_params,
                 env,
                 model,
                 tester_params,
                 USE_CUDA):

        # save arguments
        # self.env_params = env_params
        # self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        # USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            # cuda_device_num = self.tester_params['cuda_device_num']
            # torch.cuda.set_device(cuda_device_num)
            # device = torch.device('cuda', cuda_device_num)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cuda_device_num = self.tester_params['cuda_device_num']
            # print('cuda_device_num', cuda_device_num)
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
            torch.set_default_tensor_type('torch.FloatTensor')

        # ENV and MODEL
        # self.env = Env(**self.env_params)
        # self.model = Model(**self.model_params)
        self.env = env
        self.model = model.to(device=self.device)
        # Model(**self.model_params)

        # Restore --> already done in runner
        # model_load = tester_params['model_load']
        # checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        # checkpoint = torch.load(checkpoint_fullname, map_location=device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, data: List[TSPInstance]):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        # always use benchmark-generated data
        self.env.use_benchmark_problems(data_instances=data, device=self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        runtimes, costs, costs_aug, sols = [], [], [], []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            start_time = time.time()
            score, sol, aug_score, sol_aug = self._test_one_batch(batch_size)
            duration = time.time() - start_time
            runtimes.append(duration)
            costs_aug.append(aug_score)
            costs.append(score)
            sols.append(sol_aug[0] if self.tester_params['augmentation_enable'] else sol[0])

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        return sols, runtimes, costs, costs_aug

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        # print("self.tester_params['augmentation_enable']", self.tester_params['augmentation_enable'])
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        # print('state.current_node after env.pre_step', state.current_node)
        selected_all = [] # added
        while not done:
            selected, _ = self.model(state)
            # print('selected[0]', selected[0])
            # print('selected[5]', selected[5])
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            # added
            selected_all.append(selected.unsqueeze(2))

        # added
        all_routes = torch.cat(selected_all, dim=-1)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        # changed:
        max_pomo_reward, indices_pomo = aug_reward.max(dim=2)  # get best results from pomo
        # max_pomo_reward_a, indices_pomo_a = aug_reward.amax(dim=(1, 2))  # added
        # print('max_pomo_reward', max_pomo_reward)
        # print('max_pomo_reward_a', max_pomo_reward_a)
        # print('indices_pomo', indices_pomo)
        # print('indices_pomo_a', indices_pomo_a)
        # print('indices_pomo.size()', indices_pomo.size())
        # print('indices_pomo_a.size()', indices_pomo_a.size())
        # print('indices_pomo[:, 0]', indices_pomo[:, 0])
        # added:
        # print('self.env.selected_node_list==all_routes', (self.env.selected_node_list==all_routes).all())
        # print('self.env.selected_node_list.size()', self.env.selected_node_list.size())
        # best_sols_pomo = all_routes[:, indices_pomo[0], :]
        # best_sols_pomo = self.env.selected_node_list[:, indices_pomo[:, 0], :]
        indices_pomo_ex = indices_pomo.unsqueeze(2).expand(aug_factor*batch_size, 1, self.env.pomo_size)
        # print('indices_pomo_ex.size()', indices_pomo_ex.size())
        # print('indices_pomo_ex', indices_pomo_ex)
        best_sols_pomo = self.env.selected_node_list.gather(1, index=indices_pomo_ex)
        # print('best_sols_pomo[0]==best_sols_pomo[1]', best_sols_pomo[0]==best_sols_pomo[1])
        # print('best_sols_pomo.size()', best_sols_pomo.size())
        # print('best_sols_pomo', best_sols_pomo)
        # print('best_sols_pomo.tolist()', best_sols_pomo.tolist())
        best_sol_pomo = best_sols_pomo[0]
        # print('best_sols_pomo[0].size()', best_sol_pomo.size())
        # shape: (augmentation, batch)
        # print('-max_pomo_reward', -max_pomo_reward)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        # changed:
        max_aug_pomo_reward, index_augm = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # print('index_augm', index_augm)
        # shape: (batch,)
        best_sol_pomo_augm = best_sols_pomo[index_augm][0] # added
        # best_sol_pomo_augm = self.env.selected_node_list[indices_pomo[index_augm]][0]  # added
        # print('best_sol_pomo_augm.size()', best_sol_pomo_augm.size())
        # print('best_sol_pomo_augm', best_sol_pomo_augm)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
        # print(' -max_aug_pomo_reward',  -max_aug_pomo_reward)

        return no_aug_score.item(), best_sol_pomo, aug_score.item(), best_sol_pomo_augm
