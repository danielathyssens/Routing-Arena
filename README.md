# RA
### **Routing Arena**: A Benchmark Suite for Neural Routing Solvers
This repository corresponds to the paper ""_Routing Arena: A Benchmark Suite for Neural Routing Solvers_"".
The Benchamark Suite aims at providing a seamless integration of **consistent evaluation** and **provision of baselines 
and benchmarks** for routing solvers prevalent in the Machine Learning (ML)- and Operations Research (OR) field, by unifying the evaluation 
protocol and extending the pool of easily accessible baselines and datasets.

**The implementation** currently focuses on the _Capacitated Vehicle Routing Problem_ (CVRP) as one of the most common 
problems in Neural Combinatorial Optimization (NCO) and among the most well studied problems in OR. Further 
interfacing, datasets and baselines for the Traveling Salesmen Problem (TSP) and the Vehicle Routing Problem with Time 
Windows (VRPTW) in particular are currently being adopted.

Below are some quick-start instructions to test the functionality of the RA.
A comprehensive **Documentation** on how the Suite is designed and structured can be found in the appendix of the paper.
A copy of the Documentation PDF as well as a **Quick Baseline Setup** Guide will be available here soon. 


---
### Setup
1. Install the requirements as conda environment
```sh
conda env create -f environment.yml
```
2. Update the CPU and GPU PassMark specifications in the respective `config/meta/run.yaml` file of the model you want to 
evaluate to retrieve the correct runtime normalizations that correspond to your machine.
To run the example below, update this [run config file](models/SGBS/config/meta/run.yaml).
The PassMarks for CPUs and GPUs can be found [here](https://www.cpubenchmark.net/high_end_cpus.html) 
and [here]( https://www.videocardbenchmark.net/high_end_gpus.html) respectively. 
The files `formats/CPU_MACHINES.pkl` and `formats/GPU_MACHINES.pkl` comprise a selection of common machine PassMark 
specifications, you can also add your machine's specs there to avoid manually updating the PassMark specifications 
in the config files, or enter the specs as arguments to the command line.
---
### Evaluation Run - Quick Start
After activating the environment and updating the CPU/GPU PassMark specifications, executing an evaluation run for a 
GPU-based ML baseline models, works out of the box. 

**CVRP**: Evaluating the SGBS model with efficient active search [1] on one of the uchoa-type
_XE_ benchmark sets [2] (_XE_1_) on three consecutive runs can be done with the following command:
```
python run_SGBS.py policy=sgbs_EAS env=cvrp_XE_uch XE_type=XE_1 test_cfg.time_limit=implicit number_runs=3 test_cfg.save_for_analysis=True
```

Note the "save for analysis" option in the `test_cfg` argument in the command, that allows users to get a quick 
summary of the results. To access the stored results, run:
```
from models.analyse import average_run_results
avg_res = average_run_results(path_to_results="outputs/saved_results/XE/XE_1/TL_implicit", model_name="SGBS-EAS", number_runs=3) # gives dict with summary stats
```

**TSP**: Preliminary experiment to test the functionality for the TSP on uniformly sampled instances:

```
python run_SGBS.py policy=sgbs_EAS env=tsp100_unf test_cfg.dataset_size=2 test_cfg.time_limit=10 test_cfg.eval_type=simple
```

---
#### References
[1] J. Choo, Y. Kwon, J. Kim, J. Jae, A. Hottung, K. Tierney, and Y. Gwon. 
Simulation-guided beam search for neural combinatorial optimization. In 
NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c7d5afc4-Abstract-Conference.html

[2] A. Hottung and K. Tierney. Neural large neighborhood search for the capacitated vehicle routing
problem. In G. D. Giacomo, A. Catalá, B. Dilkina, M. Milano, S. Barro, A. Bugarín, and
J. Lang, editors, ECAI 2020 - 24th European Conference on Artificial Intelligence, Santiago de
Compostela, Spain, volume 325 of Frontiers in Artificial Intelligence and Applications, pages
443–450. IOS Press, 2020. doi: 10.3233/FAIA200124. URL https://doi.org/10.3233/FAIA200124.

---
#### Acknowledgments

The baseline code implementations in the models directory are based on the 
following original source-code publications of the respective baselines:

- AM: https://github.com/wouterkool/attention-learn-to-route
- DACT: https://github.com/yining043/VRP-DACT
- DPDP: https://github.com/wouterkool/dpdp
- FILO: https://github.com/acco93/filo
- HGS: https://github.com/vidalt/HGS-CVRP
- MDAM: https://github.com/liangxinedu/MDAM
- NeuroLKH: https://github.com/liangxinedu/
- NeuroLS: https://github.com/jokofa/NeuroLS
- NLNS: https://github.com/ahottung/NLNS
- ortools: https://pypi.org/project/ortools/
- POMO: https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver
- Savings: https://github.com/yorak/VeRyPy
- SGBS: https://github.com/yd-kwon/SGBS
- NeuOpt: https://github.com/yining043/NeuOpt

For further information
concerning implementation and paper references, see the README.md in the respective 
baseline directories.