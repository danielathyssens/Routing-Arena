


# CVRPDataset utils

EPS = 0.01  # 0.002 changed in cluster b/c of NLNS # np.finfo(np.float32).eps

CVRPLIB_LINKS = {
    "D": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-D.zip", "D"],
    "X": ["vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-X.zip", "X"],
    "Li": ["vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Li.zip", "Li"],
    "Golden": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Golden.zip", "Golden"],
    "XML100": ["http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-XML100.zip", "XML100"]
}

SCALE_FACTORS_CVRP = {
    "uchoa": 1000,
    "XE": 1000,
    "XML100": 1000,
    "subsampled": 1000,
    "dimacs": 1000,
    "Li": 1000,
    "Golden": 1000,
    "VRPLib": 1000
}

CVRP_DEFAULTS = {  # num vehicles and integer capacity per problem size
    20: [8, 30],
    50: [16, 40],
    100: [32, 50],
    200: [48, 50],
    500: [64, 50],
}

XE_UCHOA_TYPES = {  # depot type and customer distribution type
    'XE_1': ['R', 'RC', "1-100"],
    'XE_2': ['R', 'C', "Q"],
    'XE_3': ['E', 'RC', "1-10"],
    'XE_4': ['C', 'RC', '50-100'],
    'XE_5': ['R', 'C', 'U'],
    'XE_6': ['R', 'R', '50-100'],
    'XE_7': ['R', 'C', 'Q'],
    'XE_8': ['C', 'RC', '50-100'],
    'XE_9': ['C', 'C', '1-100'],
    'XE_10': ['E', 'R', 'U'],
    'XE_11': ['E', 'R', 'U'],
    'XE_12': ['E', 'R', '1-10'],
    'XE_13': ['C', 'RC', '50-100'],
    'XE_14': ['R', 'C', 'U'],
    'XE_15': ['E', 'R', 'SL'],
    'XE_16': ['C', 'R', '1-100'],
    'XE_17': ['R', 'R', '1-100'],
}


DATA_KEYWORDS = {
    'uniform': 'uniform',
    'uniform_fu': 'uniform_fu',
    'nazari': 'uniform',
    'rej': 'rejection_sampled',
    'uchoa': 'uchoa_distributed',
    'explosion': 'explosion',
    'rotation': 'rotation',
    'tsplib': 'tsplib_format',
    'homberger': 'homberger_200',
    'XE': 'XE',
    'S': 'S'
}

TEST_SETS_BKS = ['tsp100_fu.pt',
                 'tsp200_fu.pt',
                 'tsp500_fu.pt',
                 'tsp1000_fu.pt',
                 'tsp10000_fu.pt',
                 'cvrp20_test_seed1234.pkl',
                 'cvrp50_test_seed1234.pkl',
                 'cvrp100_test_seed1234.pkl',
                 'val_seed123_size512.pt',
                 'val_seed123_size512.pkl',
                 'val_seed4321_size512.pkl',
                 'val_seed4321_size128.pkl',
                 'val_seed1234_size128.pkl',
                 'val_seed22222_size128.pkl',
                 'val_seed2451_size128.pkl',
                 'val_seed3471_size128.pkl',
                 'val_seed1234_size128.pt',
                 'val_seed2345_size128.pt',
                 'val_seed44445_size128.pt',
                 'val_seed2025_size128.pt',
                 'val_seed55558_size128.pt',
                 'val_seed5588_size128.pt',
                 'val_seed9876_size128.pt',
                 'val_seed6712_size128.pt',
                 'val_seed7777_size128.pt',
                 'E_R_6_seed123_size512.pt',
                 'val_R_R_1_seed1234_size128.pt',
                 'val_seed4321_size128.pt',
                 'val_E_size2000.pkl',
                 'val_R_size2000.pkl',
                 'val_cvrptw_40.pkl',
                 'val_cvrptw_200.pkl',
                 'XE',
                 'X',
                 'XML100',
                 'subsampled',
                 'Golden']

# XE 10    218 E R     U       3
# XE 11    236 E R     U       18
# XE 12    241 E R     1-10    28
# XE 13    269 C RC(5) 50-100  585
# XE 14    274 R C(3)  U       10
# XE 15    279 E R     SL      192
# XE 16    293 C R     1-100   285
# XE 17    297 R R     1-100   55