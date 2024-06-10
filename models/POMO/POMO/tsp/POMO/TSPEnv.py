
from dataclasses import dataclass
from typing import List
import torch

from ..TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
from formats import TSPInstance

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, train_dataset, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        # added env parameters:
        self.train_dataset = train_dataset
        self.gen_args = env_params['generator_args']

        self.FLAG__use_saved_problems = False
        self.saved_problems = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, generate_problems=False, aug_factor=1, device=torch.device('cpu')):
        self.batch_size = batch_size
        # added generate_problems argument to generate problems from different distribution
        # print('generate_problems', generate_problems)
        if generate_problems:
            self.problems = self.generate_problems(batch_size=batch_size, device=device)
        elif not self.FLAG__use_saved_problems:
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.problems = self.saved_problems[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size
        # print('self.problems.shape', self.problems.shape)
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def use_benchmark_problems(self, data_instances: List[TSPInstance], device=None):
        self.FLAG__use_saved_problems = True
        self.saved_index = 0
        # print('data_instances', data_instances)
        self.saved_problems = make_pomo_instances(data_instances, device)
        # print('self.saved_problems[:3]', self.saved_problems[:3])
        # self.problem_size = self.saved_node_xy[0].size()[0] if self.problem_size is None else self.problem_size

    # added function to generate new distribution-type data from data.CVRPDataset during training
    def generate_problems(self, batch_size, device=None):
        device = torch.device("cpu") if device is None else device
        self.batch_size = batch_size
        data_instances = self.train_dataset.sample(sample_size=batch_size, graph_size=self.problem_size,
                                                   distribution=self.gen_args.coords_dist, log_info=False)

        return make_pomo_instances(data_instances.data, device)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        # print('self.problems[:5]', self.problems[:5])
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


# new function to transform CVRPInstance type data into pomo type data
def make_pomo_instances(data: List[TSPInstance], device=None):
    # needs to return (len(data), problem, 2)-shaped node instances
    # depot_xy = torch.stack([torch.FloatTensor(instance.coords[0].reshape(1, -1)) for instance in data]).to(device)
    node_xy = torch.stack([torch.FloatTensor(instance.coords) for instance in data]).to(device)
    return node_xy
