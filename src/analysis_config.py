import os


# The script reads benchmark_results.csv from this folder and writes all outputs here.
ANALYSIS_DIR = "/home/anton/code/astd_project_3/results_v2"


CLASSIFIERS_TO_ANALYZE = [
    "ROCKET",
    "MiniROCKET",
    "MultiROCKET",
    "MultiROCKETHydra",
    "1NN-ED",
    "1NN-DTW",
    "ST",
    "BOSS",
    "TSF",
    "InceptionTime",
    "catch22",
]


# Set to None to include all datasets found in the results CSV.
# DATASETS_TO_ANALYZE = [
#     "Computers",
#     "MiddlePhalanxOutlineCorrect",
#     "Worms",
#     #"EMOPain",
#     "EthanolLevel",
#     "Chinatown",
#     "ECG200",
#     "SwedishLeaf",
#     "SyntheticControl",
#     "Beef",
#     "Trace",
#     "TwoPatterns"
# ]

DATASETS_TO_ANALYZE = None