
--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 56
Content-Type: application/octet-stream
X-File-MD5: 961ae9c1ff97c63b893ccddb21e00205
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/README.md

# salesman
Prize Collecting Travelling Salesman Problem

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 257
Content-Type: application/octet-stream
X-File-MD5: bc472fc37ce386d0ced8bdfaa2b99308
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/__init__.py

# package qextractor
#
# Copyright (c) 2015 Rafael Reis
#
"""
Package qextractor - Packages for building and evaluating a machine learning
model to tackle the Quotation Extractor Task

"""
__version__="1.0"
__author__ = "Rafael Reis <rafael2reis@gmail.com>"
--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 50
Content-Type: application/octet-stream
X-File-MD5: d93f3135bc62119af8c2b1cd5c8c7cf5
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/__main__.py

# from qextractor.application import main
# main()
--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 218
Content-Type: application/octet-stream
X-File-MD5: a5bb502b2f1cf3c5a68781ca026954f8
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/algo/__init__.py

# package algo
#
# Copyright (c) 2018 Rafael Reis
#
"""
Package algo - Algorithms for solving the  Prize Collecting Travelling Salesman Problem

"""
__version__="1.0"
__author__ = "Rafael Reis <rafael2reis@gmail.com>"

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 641
Content-Type: application/octet-stream
X-File-MD5: fd8925b93876b7d46b8199a20df1de1e
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/algo/geni.py

# module geni.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
geni module - Auxiliary functions to the GENI method.

"""
__version__="1.0"

import numpy as np
import sys

def geni(v, s, max_i):
    quality_1 = 0
    quality_2 = 0

    s_star = Solution()
    s_start.quality = sys.maxint

    for i in range(1, max_i):
        quality_1 = quality_after_insertion_1(v, i, )
        quality_2 = quality_after_insertion_2()

        if quality_1 < quality_2 and quality_1 < s_star.quality:
            s_star = insertion_1(s)
        elif quality_2 < quality_1 and quality_2 < s_star.quality:
            s_star = insertion_2(s)

    return s_star

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 426
Content-Type: application/octet-stream
X-File-MD5: 4a174889c01300635e95435c13ce5a59
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/algo/genius.py

# module genius.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
genius module - Implements GENIUS, an algorithm for generation of a solution.

"""
__version__="1.0"

from pctsp.model.pctsp import *
from pctsp.model import solution

import numpy as np

def genius(pctsp):
    s = solution.random(pctsp, size=3)
    s = geni(pstsp, s)
    s = us(pctsp, s)

    return s

def geni(pctsp, s):
    return

def us(pctsp, s):
    return

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 2127
Content-Type: application/octet-stream
X-File-MD5: c35a3c96a6c079113cea5420945923fc
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/algo/ilocal_search.py

# module ilocal_search.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
ilocal_search module - Implements Iterate Local Search algorithm.

"""
__version__="1.0"

import numpy as np
import random

def ilocal_search(s, n_runs=10):
    h = s.copy()
    best = s.copy()
    times = [1000] * n_runs  # random.sample(range(1000, 2000), n_runs)

    while len(times) > 0:
        time = times.pop()
        t = 0
        s_tabu = s.copy()
        while t < time:
            r = tweak(s_tabu.copy())
            if r.quality < s_tabu.quality:
                s_tabu = r

                if s_tabu.is_valid():
                    s = s_tabu
            t += 1

        if s.quality < best.quality and s.is_valid():
            best = s
        
        h = newHomeBase(h, s)
        s = perturb(h)
    
    return best

def tweak(solution):
    s = solution

    s_1 = m1(solution.copy())
    s_2 = m2(solution.copy())
    
    if (s_1 and s_1.quality < solution.quality 
        and (not s_2 or s_1.quality < s_2.quality)
        ):#and s_1.is_valid()):
        s = s_1
    elif (s_2 and s_2.quality < solution.quality
        and (not s_1 or s_2.quality < s_1.quality)
        ):#and s_2.is_valid()):
        s = s_2
    else:
        s_3 = m3(solution.copy())
        if (s_3 and s_3.quality < solution.quality
            ):#and s_3.is_valid()):
            s = s_3

    return s

def newHomeBase(h, s):
    if s.quality <= h.quality:
        return s
    else:
        return h

def perturb(solution):
    s = solution.copy()
    if s.size > 5:
        quant = int(s.size/5)
        s.remove_cities(quant=quant)

    return s

def m1(solution):
    size = solution.size
    length = len(solution.route)

    if size > 1 and size < length:
        i = random.randrange(1, size)
        j = random.randrange(size, length)
        solution.swap(i, j)
   
    return solution

def m2(solution):
    if solution.size > 1:
        i = random.randrange(1, solution.size)
        solution.remove_city(index=i)

    return solution

def m3(solution):
    if solution.size < len(solution.route):
        solution.add_city()

    return solution

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 1467
Content-Type: application/octet-stream
X-File-MD5: 50eb182530990c9b6755ba066afd020d
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/application.py

# module application.py
#
# Copyright (c) 2015 Rafael Reis
#
"""
application module - Main module that solves the Prize Collecting Travelling Salesman Problem

"""

from pctsp.model.pctsp import *
from pctsp.model import solution
from pctsp.algo.genius import genius
from pctsp.algo import ilocal_search as ils
from pkg_resources import resource_filename
import random

INPUT_INSTANCE_FILE = resource_filename('pctsp', 'data/problem_20_100_100_1000.pctsp')

def solve_instance(filename, min_prize, runs=10, seed=1234):
    random.seed(seed)
    pctsp = Pctsp()
    pctsp.load(filename, min_prize)
    s = solution.random(pctsp, size=int(len(pctsp.prize) * 0.7))
    s = ils.ilocal_search(s, n_runs=runs)

    return (s.route[1:], s.quality)

def main():
    """Main function, that solves the PCTSP.

    """
    pctsp = Pctsp()
    pctsp.load(INPUT_INSTANCE_FILE, 386)
    #pctsp.prize = np.array([0, 4, 8, 3])
    #pctsp.penal = np.array([1000, 7, 11, 17])
    #pctsp.cost = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    # print(pctsp.type)

    size = int(len(pctsp.prize)*0.7)

    s = solution.random(pctsp, size=size)
    print(s.route)
    print(s.size)
    print(s.quality)
    print(s.is_valid())

    print("\n")

    # s = genius(pctsp)
    # print(s.route)
    # print(s.quality)

    s = ils.ilocal_search(s)
    print(s.route)
    print(s.size)
    print(s.quality)
    print(s.is_valid())

if __name__ == '__main__':
    main()

--boundary_.oOo._l3OJ0Ey3u+kAc+MFwC2TcTWAV7VQtqS4
Content-Length: 1051
Content-Type: application/octet-stream
X-File-MD5: 9fa27bda8494cf2c57f3c56b77c4712b
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/pctsp/salesman/pctsp/model/pctsp.py

# module pctsp.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
pctsp module - Implements Pctsp, a class that describes an instance of the problem..

"""
__version__="1.0"

import numpy as np
import re

class Pctsp(object):
    """
    Attributes:
       c (:obj:`list` of :obj:`list`): Costs from i to j
       p (:obj:`list` of :obj:`int`): Pr