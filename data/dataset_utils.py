


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


# XE 10    218 E R     U       3
# XE 11    236 E R     U       18
# XE 12    241 E R     1-10    28
# XE 13    269 C RC(5) 50-100  585
# XE 14    274 R C(3)  U       10
# XE 15    279 E R     SL      192
# XE 16    293 C R     1-100   285
# XE 17    297 R R     1-100   55