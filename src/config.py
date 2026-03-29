import os

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.convolution_based import MultiRocketClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.feature_based import Catch22Classifier

# --- Datasets ---
DATASETS = [
    "Computers",
    "MiddlePhalanxOutlineCorrect",
    "WormsTwoClass",
    "Worms",
    "Earthquakes",
    "EMOPain",
    "EthanolLevel",
    "Chinatown",
    "ECG200",
    "CBF",
    "SwedishLeaf",
    "SyntheticControl",
    "Beef",
    "Trace",
    "Wafer",
    "TwoPatterns"
]

# --- Classifiers ---
CLASSIFIERS = {
    "ROCKET":       lambda: RocketClassifier(random_state=RANDOM_SEED),
    "MiniROCKET":   lambda: MiniRocketClassifier(random_state=RANDOM_SEED),
    "MultiROCKET":   lambda: MultiRocketClassifier(random_state=RANDOM_SEED),
    "MultiROCKETHydra": lambda: MultiRocketHydraClassifier(random_state=RANDOM_SEED),
    "1NN-ED":       lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean"),
    "1NN-DTW":      lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"),
    "ST":           lambda: ShapeletTransformClassifier(random_state=RANDOM_SEED),
    "BOSS":         lambda: BOSSEnsemble(random_state=RANDOM_SEED),
    "TSF":          lambda: TimeSeriesForestClassifier(random_state=RANDOM_SEED),
    "InceptionTime": lambda: InceptionTimeClassifier(random_state=RANDOM_SEED, n_epochs=100),    #default n_epochs is 1500
    "catch22":      lambda: Catch22Classifier(random_state=RANDOM_SEED),
}

# --- k fold CV settings ---
N_FOLDS = 5
RANDOM_SEED = 42

# --- Output paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
