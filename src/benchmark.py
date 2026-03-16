import time
import warnings

import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

from config import CLASSIFIERS, DATASETS, N_FOLDS, RANDOM_SEED, RESULTS_CSV, RESULTS_DIR

import os

warnings.filterwarnings("ignore")


def load_dataset(name):
    """Load train+test splits and merge into a single pool."""
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return X, y


def run_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        X, y = load_dataset(ds_name)
        print(f"  Samples: {X.shape[0]}, Length: {X.shape[2]}, Classes: {len(np.unique(y))}")

        # apply k-fold CV
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for clf_name, clf_factory in CLASSIFIERS.items():
                try:
                    clf = clf_factory()

                    t0 = time.perf_counter()
                    clf.fit(X_train, y_train)
                    fit_time = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    y_pred = clf.predict(X_test)
                    predict_time = time.perf_counter() - t0

                    acc = accuracy_score(y_test, y_pred)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    kappa = cohen_kappa_score(y_test, y_pred)

                    print(f"  [{clf_name}] fold {fold_idx+1}/{N_FOLDS}  "
                          f"acc={acc:.4f}  fit={fit_time:.1f}s  pred={predict_time:.1f}s")

                    results.append({
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "fold": fold_idx,
                        "accuracy": acc,
                        "balanced_accuracy": bal_acc,
                        "f1_weighted": f1,
                        "cohen_kappa": kappa,
                        "fit_time_s": fit_time,
                        "predict_time_s": predict_time,
                    })
                except Exception as e:
                    print(f"  [{clf_name}] fold {fold_idx+1}/{N_FOLDS}  FAILED: {e}")
                    results.append({
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "fold": fold_idx,
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "f1_weighted": np.nan,
                        "cohen_kappa": np.nan,
                        "fit_time_s": np.nan,
                        "predict_time_s": np.nan,
                    })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")
    print(f"Total rows: {len(df)}, Failures: {df['accuracy'].isna().sum()}")
    return df


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    run_experiments()
