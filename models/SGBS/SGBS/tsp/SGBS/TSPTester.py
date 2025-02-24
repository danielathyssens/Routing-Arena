
"""
The MIT License

Copyright (c) 2022 SGBS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import numpy as np

import os
from logging import getLogger
import pickle
import copy

# added:
from typing import List
from formats import TSPInstance

# from E_TSPEnv import E_TSPEnv as Env
# from E_TSPModel import E_TSPModel as Model

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
        # self.logger = getLogger(name='trainer')  # chaNGED
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()


        # cuda
        # if self.run_params['use_cuda']:
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = env  # Env(**self.env_params)
        # self.env = Env(**self.env_params)
        self.model = model.to(device=device)
        # self.model = Model(**self.model_params)

        # Restore
        # model_load = tester_params['model_load']
        # checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        # checkpoint = torch.load(checkpoint_fullname, map_location=device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        # Augmentation
        self.aug_factor = 8 if self.tester_params['augmentation_enable'] else 1

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, data: List[TSPInstance], time_budget=None):
        self.time_estimator.reset()

        # Load problems from problem file
        # filename = self.tester_params['test_data_load']['filename']
        # index_begin = self.tester_params['test_data_load']['index_begin']
        index_begin = self.tester_params['index_begin']
        test_num_episode = self.tester_params['test_episodes']
        # self.env.use_pkl_saved_problems(filename, test_num_episode, index_begin)
        self.env.use_benchmark_problems(data_instances=data, device=self.device)

        # prepare
        result_arr = torch.zeros(test_num_episode)

        # run
        with torch.no_grad():
            episode = 0
            runtimes, costs, costs_aug, sols = [], [], [], []  # added
            while episode < test_num_episode:
            
                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)
                start_time = time.time()  # added

                batch_score, sol = self._test_one_batch_simulation_guided_beam_search(episode, batch_size)  # changed
                duration = time.time() - start_time  # added
                runtimes.append(duration)  # added
                # costs_aug.append(aug_score)   # added
                costs.append(batch_score.item())  # added
                sols.append(sol[0])  # added

                result_arr[episode:episode+batch_size] = batch_score            
                episode += batch_size
            
                # Logs
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info("{:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, score_mean:{:.6f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, 
                    batch_score.mean().item(), result_arr[:episode].mean().item()))
                                            
        # Save Result
        # result_to_save = {
        #     'index_begin': index_begin,
        #     'num_episode': test_num_episode,
        #     'result_arr': result_arr.cpu().numpy(),
        # }
        # with open('{}/result.pkl'.format(self.result_folder), 'wb') as f:
        #     pickle.dump(result_to_save, f)

        # Done
        self.logger.info(" *** Done *** ")
        self.logger.info(" Final Score: {}".format(result_arr.mean().item()))

        # return result_arr.mean().item() # changed
        return sols, runtimes, costs

    def _get_pomo_starting_points(self, model, env, num_starting_points):
        
        # Ready
        ###############################################
        model.eval()
        env.modify_pomo_size(self.env.env_params['pomo_size'])
        env.reset()

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        
        # starting points
        ###############################################
        sorted_index = reward.sort(dim=1, descending=True).indices
        selected_index = sorted_index[:, :num_starting_points]
        # shape: (batch, num_starting_points)
        
        return selected_index    


    def _test_one_batch_simulation_guided_beam_search(self, episode, batch_size):

        beam_width = self.tester_params['sgbs_beta']     
        expansion_size_minus1 = self.tester_params['sgbs_gamma_minus1']
        rollout_width = beam_width * expansion_size_minus1
        aug_batch_size = self.aug_factor * batch_size
    
        # Ready
        ###############################################
        self.model.eval()
        self.env.load_problems_by_index(episode, batch_size, self.aug_factor)
        selected_all = [] # added
        
        reset_state, _, __ = self.env.reset()
        self.model.pre_forward(reset_state)


        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, beam_width)
        # print('starting_points.size()', starting_points.size())
        # print('starting_points', starting_points)
        # Beam Search
        ###############################################
        self.env.modify_pomo_size(beam_width)
        self.env.reset()

        # the first step, pomo starting points           
        state, _, done = self.env.step(starting_points)
        selected_all.append(starting_points.unsqueeze(2))

        # BS Step > 1
        ###############################################

        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.env)
        rollout_env.modify_pomo_size(rollout_width)

        # LOOP
        first_rollout_flag = True
        while not done:

            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :expansion_size_minus1]
                idx_selected = ordered_i[:, :, :expansion_size_minus1]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:expansion_size_minus1+1]
                idx_selected = ordered_i[:, :, 1:expansion_size_minus1+1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.env, repeat=expansion_size_minus1)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                selected, _ = self.model(rollout_state)
                # shape: (aug*batch, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)
            # rollout_reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            rollout_reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, slightly improves performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(self.env)
                rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                # rollout_reward.shape: (aug*batch, rollout_width + beam_width)
                next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            first_rollout_flag = False

            # BS Step
            ###############################################
            sorted_reward, sorted_index = rollout_reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            self.env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            selected = next_nodes.gather(dim=1, index=beam_index)
            selected_all.append(selected.unsqueeze(2)) # added
            # shape: (aug*batch, beam_width)
            state, reward, done = self.env.step(selected)

    
        # Return
        ###############################################
        tour = torch.cat(selected_all, dim=-1)  # added
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
    
        max_pomo_reward, indices_sgbs = aug_reward.max(dim=2)  # .values  # get best results from simulation guided beam search
        best_sol_sgbs = tour[:, indices_sgbs[0], :]
        # shape: (augmentation, batch)
    
        max_aug_pomo_reward, index_augm = max_pomo_reward.max(dim=0)   # .values  # get best results from augmentation
        best_sol_sgbs_augm = tour[index_augm][0]
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value
    
        return aug_score, best_sol_sgbs_augm

        

