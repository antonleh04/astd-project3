import os

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.feature_based import Catch22Classifier

# --- Datasets ---
DATASETS = [
    "Computers",
    "ECG200",
    "DistalPhalanxTW",
    "Worms",
    "Earthquakes"
]

# --- Classifiers ---
CLASSIFIERS = {
    "ROCKET":       lambda: RocketClassifier(random_state=RANDOM_SEED),
    "1NN-ED":       lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean"),
    "1NN-DTW":      lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"),
    "ST":           lambda: ShapeletTransformClassifier(random_state=RANDOM_SEED),
    "BOSS":         lambda: BOSSEnsemble(random_state=RANDOM_SEED),
    "TSF":          lambda: TimeSeriesForestClassifier(random_state=RANDOM_SEED),
    #"InceptionTime": lambda: InceptionTimeClassifier(random_state=RANDOM_SEED),
    "catch22":      lambda: Catch22Classifier(random_state=RANDOM_SEED),
}

# --- k fold CV settings ---
N_FOLDS = 5
RANDOM_SEED = 42

# --- Output paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
