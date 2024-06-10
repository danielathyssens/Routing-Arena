
--boundary_.oOo._0qEJvRiftPGqAtVkk2e8Ngov04khU87e
Content-Length: 9
Content-Type: application/octet-stream
X-File-MD5: af63a7f3d6b1a8c4a73cb4b5c75358fb
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/tsp/.gitignore

concorde/
--boundary_.oOo._0qEJvRiftPGqAtVkk2e8Ngov04khU87e
Content-Length: 906
Content-Type: application/octet-stream
X-File-MD5: f9e783cb68d6bb289066fa4e1ce7bdff
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/tsp/install_concorde.sh

#!/bin/bash
mkdir concorde
cd concorde
mkdir qsopt
cd qsopt
# Download qsopt
if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.a
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.h
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt
else
    wget http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/centos/qsopt.a
    wget http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/centos/qsopt.h
    wget http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/centos/qsopt
fi
cd ..
wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar xf co031219.tgz
cd concorde
if [[ "$OSTYPE" == "darwin"* ]]; then
    ./configure --with-qsopt=$(pwd)/../qsopt --host=powerpc-apple-macos
else
    ./configure --with-qsopt=$(realpath ../qsopt)
fi
make
TSP/concorde -s 99 -k 100
cd ../..
--boundary_.oOo._0qEJvRiftPGqAtVkk2e8Ngov04khU87e
Content-Length: 2449
Content-Type: application/octet-stream
X-File-MD5: 75ac5fa0b347e02da8ffb1b988d9d339
X-File-Mtime: 1645816036
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/tsp/problem_tsp.py

from torch.utils.data import Dataset
import torch
import os
import pickle
from ...problems.tsp.state_tsp import StateTSP
from ...utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

--boundary_.oOo._0qEJvRiftPGqAtVkk2e8Ngov04khU87e
Content-Length: 5355
Content-Type: application/octet-stream
X-File-MD5: ce6d2c70249edc687d6846c7720fcf44
X-File-Mtime: 1642773131
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/tsp/state_tsp.py

import torch
from typing import NamedTuple
from ...utils.boolmask import mask_long2bool, mask_long_scatter


class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.s