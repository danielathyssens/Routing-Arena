
--boundary_.oOo._oOwi39EtysyA/VWuRlI5hBNKpMmHUblQ
Content-Length: 4987
Content-Type: application/octet-stream
X-File-MD5: 6e472b5d2c5558e267f105a40612a832
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/model/solution.py

# module solution.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
solution module - Implements Solution, a class that describes a solution for the problem.

"""
__version__="1.0"

import numpy as np
import copy
import sys
from random import shuffle

def random(pctsp, start_size):
    s = Solution(pctsp)
    length = len(pctsp.prize)

    # Modification: start from start_size but increase after maximum number of iterations in case no feasible solution
    # is found. When the full length is used, there should always be a feasible solution
    for size in range(start_size, length + 1):
        if size: s.size = size

        i = 0
        min_solutions = 30
        max_solutions = 1000

        while i < min_solutions or (i < max_solutions and not s.is_valid()):
            r = Solution(pctsp)
            if size: r.size = size
            cities = list(range(1, length, 1))
            shuffle(cities) # Shuffle in place
            r.route = [0] + cities # The city 0 is always the first

            if r.quality < s.quality and r.is_valid():
                s = r

            i += 1
        if s.is_valid():
            break
    assert s.is_valid()
    return s


class Solution(object):
    """
    Attributes:
       route (:obj:`list` of :obj:`int`): The list of cities in the visiting order
       size (:obj:`int`): The quantity of the first cities to be considered in the route list
       quality (:obj:`int`): The quality of the solution
    """

    def __init__(self, pctsp, size=None):
        self._route = []
        
        if size:
            self.size = size
        else:
            self.size = len(pctsp.prize) # Default size value is the total of cities
        
        self.quality = sys.maxsize
        self.pctsp = pctsp
        self.prize = 0

    """
    Computes the quality of the solution.
    """
    def compute(self):
        self.prize = 0
        self.quality = 0

        for i,city in enumerate(self._route):
            if i < self.size:
                self.prize += self.pctsp.prize[city]
                if i > 0:
                    previousCity = self._route[i - 1]
                    self.quality += self.pctsp.cost[previousCity][city]
                if i + 1 == self.size:
                    self.quality += self.pctsp.cost[city][0]
            else:
                self.quality += self.pctsp.penal[city]

    def copy(self):
        cp = copy.copy(self)
        cp._route = list(self._route)

        return cp
    
    def swap(self, i, j):
        city_i = self._route[i]
        city_i_prev = self._route[i-1]
        city_i_next = self._route[(i+1) % self.size]
        
        city_j = self._route[j]

        self.quality = (self.quality
                - self.pctsp.cost[city_i_prev][city_i] - self.pctsp.cost[city_i][city_i_next]
                + self.pctsp.cost[city_i_prev][city_j] + self.pctsp.cost[city_j][city_i_next]
                - self.pctsp.penal[city_j] + self.pctsp.penal[city_i])
        self.prize = self.prize - self.pctsp.prize[city_i] + self.pctsp.prize[city_j]

        self._route[j], self._route[i] = self._route[i], self._route[j]

    def is_valid(self):
        return self.prize >= self.pctsp.prize_min

    def add_city(self):
        city_l = self._route[self.size - 1]
        city_add = self._route[self.size]
        
        self.quality = (self.quality
            - self.pctsp.cost[city_l][0]
            - self.pctsp.penal[city_add]
            + self.pctsp.cost[city_l][city_add]
            + self.pctsp.cost[city_add][0])
        
        self.size += 1
        self.prize += self.pctsp.prize[city_add]

    def remove_city(self, index):
        city_rem = self._route[index]
        city_rem_prev = self._route[index-1]
        city_rem_next = self._route[(index+1)%self.size]

        self.quality = (self.quality
            - self.pctsp.cost[city_rem_prev][city_rem] - self.pctsp.cost[city_rem][city_rem_next]
            + self.pctsp.penal[city_rem]
            + self.pctsp.cost[city_rem_prev][city_rem_next])
        self.prize -= self.pctsp.prize[city_rem]

        del self._route[index]        
        self._route.append(city_rem)

        self.size -= 1

    def remove_cities(self, quant):
        for i in range(self.size-quant,self.size):
            city_rem = self._route[i]
            city_rem_prev = self._route[i-1]

            self.quality = (self.quality 
                - self.pctsp.cost[city_rem_prev][city_rem]
                + self.pctsp.penal[city_rem])
            self.prize -= self.pctsp.prize[city_rem]

        city_rem = self._route[self.size-1]
        city_l = self._route[self.size-quant-1]
        self.quality = (self.quality - self.pctsp.cost[city_rem][0]
            + self.pctsp.cost[city_l][0])

        self.size -= quant

    def print_route(self):
        print(self._route)

    @property
    def route(self):
        return self._route

    @route.setter
    def route(self, r):
        self._route = r
        self.compute()

--boundary_.oOo._oOwi39EtysyA/VWuRlI5hBNKpMmHUblQ
Content-Length: 1668
Content-Type: application/octet-stream
X-File-MD5: e28ff7c55820f001b341a5c815553150
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/model/tests/test_solution.py

# python -m pctsp.model.tests.test_solution
import unittest

from pctsp.model import solution
from pctsp.model import pctsp
import numpy as np

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.p = pctsp.Pctsp()
        self.p.prize = np.array([0, 4, 8, 3])
        self.p.penal = np.array([1000, 7, 11, 17])
        self.p.cost = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    def test_quality(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 4)

    def test_quality_2(self):
        s = solution.Solution(self.p, size=2)
        s.route = [0, 1, 2, 3]
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 30)

    def test_swap(self):
        s = solution.Solution(self.p, size=3)
        s.route = [0, 1, 2, 3]
        
        s.swap(1,3)
        print("Quality: ", s.quality)
        print("route:", s.route)
        self.assertEqual(s.quality, 10)

    def test_add_city(self):
        s = solution.Solution(self.p, size=3)
        s.route = [0, 1, 2, 3]
        
        s.add_city()
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 4)

    def test_remove_city(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]

        s.remove_city(3)
        print("Quality: ", s.quality)
        self.assertEqual(s.quality, 20)

    def test_remove_cities(self):
        s = solution.Solution(self.p)
        s.route = [0, 1, 2, 3]

        s.remove_cities(quant=3)
        self.assertEqual(s.quality, 35)

if __name__ == '__main__':
    unittest.main()

--boundary_.oOo._oOwi39EtysyA/VWuRlI5hBNKpMmHUblQ
Content-Length: 7412
Content-Type: application/octet-stream
X-File-MD5: 6e4b2c360af284e99625406a85eeae52
X-File-Mtime: 1642773254
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/state_pctsp.py

import torch
from typing import NamedTuple
from ...utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F


class StatePCTSP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    expected_prize: torch.Tensor
    real_prize: torch.Tensor
    penalty: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tens