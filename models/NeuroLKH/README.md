## Readme

Short explanation on how the files in this directory 
work with the NeuroLKH code

#### Setup
1) create OBJ dir in SRC
2) run make from baselines/NeuroLKH/NeuroLKH
3) neuro_lkh.py calls subprocess module with LKH executable. 
The path to the executable is given in the policy config file.

#### Configuration
Training and inference runs are configured via the hydra cfg yaml files 
located in the directory ./config. 
The runs are facilitated via the Runner model wrapper object.
