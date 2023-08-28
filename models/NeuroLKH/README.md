# NeuroLKH & LKH-3 

This directory contains the baseline implementations for:
- LKH-3, based on the work in _Helsgaun, Keld. "An extension of the 
Lin-Kernighan-Helsgaun TSP solver 
for constrained traveling salesman and vehicle routing problems." 
Roskilde: Roskilde University 12 (2017)._
- NeuroLKH, based on the work in _Xin, Liang, et al. "NeuroLKH: 
Combining deep learning model with Lin-Kernighan-Helsgaun heuristic for 
solving the traveling salesman problem." 
Advances in Neural Information Processing Systems 34 (2021): 7472-7483._

The code for NeuroLKH is taken from 
https://github.com/liangxinedu/NeuroLKH. The original LKH-3 code 
(version 3.0.6) is taken from 
http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz. 
Some parts of the wrapper files for LKH-3 and NeuroLKH is based on the implementation 
in https://github.com/jokofa/NeuroLS.

Short explanation on how to get started with running the baselines in this
directory:

1) create OBJ dir in SRC
2) run make from models/NeuroLKH/NeuroLKH
3) neuro_lkh.py calls subprocess module with LKH executable. 
The path to the executable is given in the policy config file.
4) in order to run LKH-3, specify `-policy lkh` as an argument 
when running neuro_lkh.py
