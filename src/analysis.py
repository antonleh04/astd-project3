import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from analysis_config import CLASSIFIERS_TO_ANALYZE, DATASETS_TO_ANALYZE, ANALYSIS_DIR

import os


def load_results():
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "benchmark_results.csv"))
    if CLASSIFIERS_TO_ANALYZE is not None:
        df = df[df["classifier"].isin(CLASSIFIERS_TO_ANALYZE)]
    if DATASETS_TO_ANALYZE is not None:
        df = df[df["dataset"].isin(DATASETS_TO_ANALYZE)]
    return df


def mean_accuracy_table(df):
    """Pivot table: datasets x classifiers, cells = mean acc ± std."""
    mean = df.groupby(["dataset", "classifier"])["accuracy"].mean().unstack()
    std = df.groupby(["dataset", "classifier"])["accuracy"].std().unstack()
    # Combined string table for display
    combined = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
    print("\n=== Mean Accuracy ± Std ===")
    print(combined.to_string())
    mean.to_csv(os.path.join(ANALYSIS_DIR, "summary_table.csv"))
    return mean, std


# ---- Plots ----

def plot_heatmap(mean_acc):
    """Heatmap of mean accuracy per dataset x classifier."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(mean_acc, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
                linewidths=0.5, vmin=0.5, vmax=1.0)
    ax.set_title("Mean Accuracy across Folds")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "accuracy_heatmap.pdf"))
    plt.close(fig)
    print("Saved accuracy_heatmap.pdf")


def plot_boxplots(df):
    """Per-dataset boxplots of accuracy across folds for each classifier."""
    datasets = df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        sns.boxplot(data=sub, x="classifier", y="accuracy", ax=ax)
        ax.set_title(ds)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "accuracy_boxplots.pdf"))
    plt.close(fig)
    print("Saved accuracy_boxplots.pdf")


def plot_timing(df):
    """Bar chart of mean fit time per classifier (log scale)."""
    timing = df.groupby("classifier")["fit_time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    timing.plot.barh(ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Mean Fit Time (s, log scale)")
    ax.set_title("Training Time Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "timing_barplot.pdf"))
    plt.close(fig)
    print("Saved timing_barplot.pdf")


def plot_fit_time_linear(df):
    """Bar chart of mean fit time per classifier (linear scale)."""
    timing = df.groupby("classifier")["fit_time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    timing.plot.barh(ax=ax)
    ax.set_xlabel("Mean Fit Time (s)")
    ax.set_title("Fit Time Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fit_time_barplot.pdf"))
    plt.close(fig)
    print("Saved fit_time_barplot.pdf")


def plot_predict_time_linear(df):
    """Bar chart of mean predict time per classifier (linear scale)."""
    timing = df.groupby("classifier")["predict_time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    timing.plot.barh(ax=ax)
    ax.set_xlabel("Mean Predict Time (s)")
    ax.set_title("Predict Time Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "predict_time_barplot.pdf"))
    plt.close(fig)
    print("Saved predict_time_barplot.pdf")


# ---- Statistical Tests ----

def compute_ranks(mean_acc):
    """Rank classifiers per dataset (1 = best). Returns ranks DataFrame and average ranks."""
    ranks = mean_acc.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()
    return ranks, avg_ranks


def friedman_test(mean_acc):
    """Run Friedman test on accuracy matrix (datasets x classifiers)."""
    # Each column is a classifier, each row a dataset
    cols = [mean_acc[c].values for c in mean_acc.columns]
    stat, p = stats.friedmanchisquare(*cols)
    return stat, p


def nemenyi_cd(k, n, alpha=0.05):
    """Compute Nemenyi critical difference.

    k: number of classifiers, n: number of datasets.
    q_alpha values from Demsar (2006) Table of critical values for the
    two-tailed Nemenyi test.
    """
    # q_alpha for alpha=0.05, k=2..10 (from standard tables)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 3.031)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return cd


def plot_cd_diagram(avg_ranks, cd, title="Critical Difference Diagram"):
    """Draw a critical difference diagram."""
    k = len(avg_ranks)
    names = avg_ranks.index.tolist()
    ranks = avg_ranks.values

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1.5)
    ax.hlines(1.0, 0.5, k + 0.5, color="black", linewidth=0.5)

    # Tick marks for each rank
    for r in range(1, k + 1):
        ax.vlines(r, 0.95, 1.05, color="black", linewidth=0.5)
        ax.text(r, 0.9, str(r), ha="center", va="top", fontsize=8)

    # Place classifiers
    for i, (name, rank) in enumerate(zip(names, ranks)):
        y_offset = 1.15 + (i % 2) * 0.2  # stagger labels
        ax.plot(rank, 1.0, "o", color="black", markersize=5)
        ax.annotate(f"{name} ({rank:.2f})", xy=(rank, 1.0),
                    xytext=(rank, y_offset),
                    ha="center", va="bottom", fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Draw CD bar
    ax.hlines(0.7, 1, 1 + cd, color="red", linewidth=2)
    ax.text(1 + cd / 2, 0.6, f"CD = {cd:.2f}", ha="center", va="top",
            fontsize=8, color="red")

    # Draw cliques (groups not significantly different)
    sorted_idx = np.argsort(ranks)
    sorted_ranks = ranks[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                # extend or create clique
                merged = False
                for c in cliques:
                    if i in c or j in c:
                        c.add(i)
                        c.add(j)
                        merged = True
                if not merged:
                    cliques.append({i, j})
    # Merge overlapping cliques
    merged_cliques = []
    for c in cliques:
        added = False
        for mc in merged_cliques:
            if mc & c:
                mc |= c
                added = True
                break
        if not added:
            merged_cliques.append(c)

    for ci, c in enumerate(merged_cliques):
        members = sorted(c)
        r_min = sorted_ranks[members[0]]
        r_max = sorted_ranks[members[-1]]
        y = 0.45 - ci * 0.08
        ax.hlines(y, r_min, r_max, color="blue", linewidth=2.5, alpha=0.6)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "cd_diagram.pdf"))
    plt.close(fig)
    print("Saved cd_diagram.pdf")


def write_summary_report(df, mean_acc, std_acc):
    """Write a human-readable summary of interesting findings to a text file."""
    path = os.path.join(ANALYSIS_DIR, "summary_report.txt")
    with open(path, "w") as f:
        classifiers = mean_acc.columns.tolist()
        datasets = mean_acc.index.tolist()

        # --- Mean accuracy across all datasets per algorithm ---
        f.write("=== Mean Accuracy across All Datasets ===\n")
        global_mean = mean_acc.mean(axis=0).sort_values(ascending=False)
        global_std = mean_acc.std(axis=0)
        for clf in global_mean.index:
            f.write(f"  {clf:25s}  {global_mean[clf]:.4f} ± {global_std[clf]:.4f}\n")

        # --- Best algorithm overall ---
        best_clf = global_mean.idxmax()
        f.write(f"\nBest algorithm (highest mean accuracy): {best_clf} ({global_mean[best_clf]:.4f})\n")
        worst_clf = global_mean.idxmin()
        f.write(f"Worst algorithm (lowest mean accuracy): {worst_clf} ({global_mean[worst_clf]:.4f})\n")

        # --- Per-algorithm: best and worst dataset ---
        f.write("\n=== Best and Worst Dataset per Algorithm ===\n")
        for clf in classifiers:
            best_ds = mean_acc[clf].idxmax()
            worst_ds = mean_acc[clf].idxmin()
            f.write(f"  {clf:25s}  best: {best_ds} ({mean_acc.loc[best_ds, clf]:.4f})"
                    f"  worst: {worst_ds} ({mean_acc.loc[worst_ds, clf]:.4f})\n")

        # --- Per-dataset winner ---
        f.write("\n=== Best Algorithm per Dataset ===\n")
        for ds in datasets:
            winner = mean_acc.loc[ds].idxmax()
            f.write(f"  {ds:25s}  {winner} ({mean_acc.loc[ds, winner]:.4f})\n")

        # --- Number of dataset wins per algorithm ---
        f.write("\n=== Number of Dataset Wins per Algorithm ===\n")
        wins = mean_acc.idxmax(axis=1).value_counts()
        for clf in classifiers:
            f.write(f"  {clf:25s}  {wins.get(clf, 0)} wins\n")

        # --- Mean fit and predict times ---
        f.write("\n=== Mean Fit Time (seconds) ===\n")
        fit_times = df.groupby("classifier")["fit_time_s"].mean().sort_values()
        for clf, t in fit_times.items():
            f.write(f"  {clf:25s}  {t:.4f}s\n")

        f.write("\n=== Mean Predict Time (seconds) ===\n")
        predict_times = df.groupby("classifier")["predict_time_s"].mean().sort_values()
        for clf, t in predict_times.items():
            f.write(f"  {clf:25s}  {t:.4f}s\n")

        # --- Hardest and easiest datasets ---
        f.write("\n=== Dataset Difficulty (mean accuracy across all algorithms) ===\n")
        ds_mean = mean_acc.mean(axis=1).sort_values(ascending=False)
        for ds in ds_mean.index:
            f.write(f"  {ds:25s}  {ds_mean[ds]:.4f}\n")
        f.write(f"\nEasiest dataset: {ds_mean.idxmax()} ({ds_mean.max():.4f})\n")
        f.write(f"Hardest dataset: {ds_mean.idxmin()} ({ds_mean.min():.4f})\n")

    print(f"Saved {path}")


def write_statistical_report(mean_acc, ranks, avg_ranks, fstat, fpval, cd):
    """Write statistical test results to text file."""
    path = os.path.join(ANALYSIS_DIR, "statistical_tests.txt")
    with open(path, "w") as f:
        f.write("=== Friedman Test ===\n")
        f.write(f"Statistic: {fstat:.4f}\n")
        f.write(f"p-value:   {fpval:.6f}\n")
        f.write(f"Significant (p < 0.05): {fpval < 0.05}\n\n")

        f.write("=== Average Ranks (lower is better) ===\n")
        for name, rank in avg_ranks.items():
            f.write(f"  {name:20s} {rank:.3f}\n")

        k = len(avg_ranks)
        n = len(mean_acc)
        f.write(f"\n=== Nemenyi Post-hoc Test ===\n")
        f.write(f"Critical Difference (CD): {cd:.4f}\n")
        f.write(f"k={k} classifiers, N={n} datasets, alpha=0.05\n\n")

        f.write("Pairwise rank differences (significant if > CD):\n")
        names = avg_ranks.index.tolist()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j])
                sig = "*" if diff > cd else ""
                f.write(f"  {names[i]:20s} vs {names[j]:20s}: "
                        f"diff={diff:.3f} {sig}\n")

    print(f"Saved {path}")


# ---- Main ----

def main():
    df = load_results()
    mean_acc, std_acc = mean_accuracy_table(df)

    write_summary_report(df, mean_acc, std_acc)
    plot_heatmap(mean_acc)
    plot_boxplots(df)
    plot_timing(df)
    plot_fit_time_linear(df)
    plot_predict_time_linear(df)

    ranks, avg_ranks = compute_ranks(mean_acc)
    fstat, fpval = friedman_test(mean_acc)
    k = len(mean_acc.columns)
    n = len(mean_acc)
    cd = nemenyi_cd(k, n)

    print(f"\nFriedman test: stat={fstat:.4f}, p={fpval:.6f}")
    print(f"Nemenyi CD = {cd:.4f}")
    print(f"\nAverage ranks:\n{avg_ranks.to_string()}")

    plot_cd_diagram(avg_ranks, cd)
    write_statistical_report(mean_acc, ranks, avg_ranks, fstat, fpval, cd)


if __name__ == "__main__":
    main()
