"""
analysis_v2.py 

Reads benchmark_results.csv via analysis_config and produces all outputs
into benchmark_analysis/ (tables, plots, statistical reports).
"""

import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy import stats

from analysis_config import ANALYSIS_DIR, CLASSIFIERS_TO_ANALYZE, DATASETS_TO_ANALYZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_analysis")

SINGLE_COL = 3.5   # inches – single-column figure width
DOUBLE_COL = 7.0   # inches – double-column / full-width figure

ALL_METRICS = ["accuracy", "balanced_accuracy", "f1_weighted", "cohen_kappa"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Bal. Accuracy",
    "f1_weighted": "F1 (weighted)",
    "cohen_kappa": "Cohen's κ",
}

# Abbreviated names for tight x-tick labels (boxplots, etc.)
SHORT_NAMES = {
    "MultiROCKETHydra": "MRHydra",
    "MultiROCKET": "MultiR",
    "MiniROCKET": "MiniR",
    "InceptionTime": "IncTime",
}

RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short(name):
    """Return abbreviated classifier name for compact labels."""
    return SHORT_NAMES.get(name, name)


def _savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results():
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "benchmark_results.csv"))
    if CLASSIFIERS_TO_ANALYZE is not None:
        df = df[df["classifier"].isin(CLASSIFIERS_TO_ANALYZE)]
    if DATASETS_TO_ANALYZE is not None:
        df = df[df["dataset"].isin(DATASETS_TO_ANALYZE)]
    return df


# ---------------------------------------------------------------------------
# Accuracy table + LaTeX
# ---------------------------------------------------------------------------


def mean_accuracy_table(df):
    mean = df.groupby(["dataset", "classifier"])["accuracy"].mean().unstack()
    std = df.groupby(["dataset", "classifier"])["accuracy"].std().unstack()

    # CSV
    mean.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"))
    print("  Saved summary_table.csv")

    # LaTeX table with bold best-per-row
    _write_latex_table(mean, std)
    return mean, std


def _write_latex_table(mean, std):
    """Write a booktabs LaTeX table with bold best value per row and an average rank row."""
    classifiers = mean.columns.tolist()
    datasets = mean.index.tolist()

    ranks, avg_ranks = compute_ranks(mean)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean classification accuracy ($\pm$ std) across 5 folds. "
                 r"\textbf{Bold} marks the best result per dataset.}")
    lines.append(r"\label{tab:accuracy}")
    col_fmt = "l" + "c" * len(classifiers)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header
    header = "Dataset & " + " & ".join(
        r"\rotatebox{60}{" + c + "}" for c in classifiers
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for ds in datasets:
        best_clf = mean.loc[ds].idxmax()
        cells = []
        for clf in classifiers:
            m = mean.loc[ds, clf]
            s = std.loc[ds, clf]
            if pd.isna(m):
                cells.append("--")
            else:
                txt = f"{m:.3f}"
                if clf == best_clf:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(ds + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # Average accuracy row
    global_mean = mean.mean(axis=0)
    best_overall = global_mean.idxmax()
    cells = []
    for clf in classifiers:
        txt = f"{global_mean[clf]:.3f}"
        if clf == best_overall:
            txt = r"\textbf{" + txt + "}"
        cells.append(txt)
    lines.append(r"\textit{Avg. Accuracy} & " + " & ".join(cells) + r" \\")

    # Average rank row
    cells = []
    best_rank_clf = avg_ranks.idxmin()
    for clf in classifiers:
        txt = f"{avg_ranks[clf]:.2f}"
        if clf == best_rank_clf:
            txt = r"\textbf{" + txt + "}"
        cells.append(txt)
    lines.append(r"\textit{Avg. Rank} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path = os.path.join(OUTPUT_DIR, "accuracy_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved accuracy_table.tex")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_heatmap(mean_acc):
    n_clf = len(mean_acc.columns)
    n_ds = len(mean_acc.index)
    fig_w = max(DOUBLE_COL, n_clf * 0.72)
    fig_h = max(4.0, n_ds * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw heatmap without annotations first
    sns.heatmap(
        mean_acc, annot=False, cmap="YlGn", ax=ax,
        linewidths=0.5, vmin=0.5, vmax=1.0,
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
    )

    # Add annotations manually – bold the best per row
    for i, ds in enumerate(mean_acc.index):
        best_clf = mean_acc.loc[ds].idxmax()
        for j, clf in enumerate(mean_acc.columns):
            val = mean_acc.iloc[i, j]
            if pd.isna(val):
                continue
            weight = "bold" if clf == best_clf else "normal"
            ax.text(j + 0.5, i + 0.5, f"{val:.3f}",
                    ha="center", va="center", fontsize=7, fontweight=weight)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Mean Classification Accuracy")
    fig.tight_layout()
    _savefig(fig, "accuracy_heatmap.pdf")


def plot_boxplots(df):
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, nrows * 2.4))
    axes = axes.flatten()

    classifiers = (CLASSIFIERS_TO_ANALYZE
                   if CLASSIFIERS_TO_ANALYZE is not None
                   else sorted(df["classifier"].unique()))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["dataset"] == ds]
        sns.boxplot(
            data=sub, x="classifier", y="accuracy", ax=ax,
            order=classifiers, width=0.6,
            fliersize=2, linewidth=0.7,
        )
        ax.set_title(ds, fontsize=9, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy" if idx % ncols == 0 else "")
        ax.set_xticklabels([_short(c) for c in classifiers],
                           rotation=90, fontsize=6)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.subplots_adjust(hspace=0.75, wspace=0.35)
    _savefig(fig, "accuracy_boxplots.pdf")


def plot_timing(df):
    """Generate fit-time, predict-time bar plots and scatter plots."""
    _plot_time_bars(df, "fit_time_s", "Mean Fit Time (s, log scale)", "fit_time_barplot.pdf")
    _plot_time_bars(df, "predict_time_s", "Mean Predict Time (s, log scale)", "predict_time_barplot.pdf")
    _plot_timing_scatter(df)
    _plot_accuracy_vs_time(df)


def _plot_time_bars(df, col, xlabel, filename):
    grouped = df.groupby("classifier")[col]
    mean_t = grouped.mean().sort_values()
    std_t = grouped.std().reindex(mean_t.index)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, 3.5))
    y_pos = np.arange(len(mean_t))

    # Clip lower error bar so it doesn't go below zero on log scale
    xerr_lo = np.minimum(std_t.values, mean_t.values * 0.99)
    xerr_hi = std_t.values
    ax.barh(y_pos, mean_t.values, xerr=[xerr_lo, xerr_hi],
            capsize=2, color="steelblue", edgecolor="white", height=0.65,
            error_kw={"linewidth": 0.7})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mean_t.index, fontsize=8)
    ax.set_xscale("log")

    # Format ticks as plain numbers (e.g. 0.001, 0.01, 1, 10, 100, 1000)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:g}"
    ))
    ax.set_xlabel(xlabel)

    # Value labels – place past the end of the error bar so they never overlap
    for i, (v, name) in enumerate(zip(mean_t.values, mean_t.index)):
        label = f"{v:.1f}s" if v >= 1 else f"{v:.3f}s"
        err_end = v + std_t[name] if pd.notna(std_t[name]) else v
        ax.text(err_end * 1.25, i, label, va="center", fontsize=6)

    fig.tight_layout()
    _savefig(fig, filename)


def _plot_timing_scatter(df):
    fit = df.groupby("classifier")["fit_time_s"].mean()
    pred = df.groupby("classifier")["predict_time_s"].mean()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

    ax.scatter(fit, pred, s=40, zorder=3, color="steelblue", edgecolors="white",
               linewidths=0.5)

    for clf in fit.index:
        ax.annotate(
            _short(clf), (fit[clf], pred[clf]),
            textcoords="offset points", xytext=(5, 4), fontsize=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Fit Time (s, log scale)")
    ax.set_ylabel("Mean Predict Time (s, log scale)")
    ax.set_title("Training vs Inference Time")
    fig.tight_layout()
    _savefig(fig, "timing_scatter.pdf")


def _plot_accuracy_vs_time(df):
    """Scatter plot comparing mean accuracy against total time (fit + predict)."""
    acc = df.groupby("classifier")["accuracy"].mean()
    total_time = (df.groupby("classifier")["fit_time_s"].mean()
                  + df.groupby("classifier")["predict_time_s"].mean())

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

    ax.scatter(total_time, acc, s=40, zorder=3, color="steelblue",
               edgecolors="white", linewidths=0.5)

    texts = []
    for clf in acc.index:
        texts.append(ax.text(total_time[clf], acc[clf], _short(clf), fontsize=6))

    ax.set_xscale("log")
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                lw=0.5))
    ax.set_xlabel("Mean Total Time (s, log scale)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Total Time")
    fig.tight_layout()
    _savefig(fig, "accuracy_vs_time_scatter.pdf")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def compute_ranks(mean_acc):
    """Rank classifiers per dataset (1 = best). NaN receives worst rank k."""
    k = len(mean_acc.columns)
    ranks = mean_acc.rank(axis=1, ascending=False)
    ranks = ranks.fillna(k)
    avg_ranks = ranks.mean(axis=0).sort_values()
    return ranks, avg_ranks


def friedman_test(ranks):
    """Friedman test on pre-computed ranks (respects NaN penalty ranks)."""
    N, k = ranks.shape
    rank_sums = ranks.sum(axis=0)
    stat = (12 / (N * k * (k + 1))) * (rank_sums ** 2).sum() - 3 * N * (k + 1)
    p = stats.chi2.sf(stat, df=k - 1)
    return stat, p


def nemenyi_cd(k, n, alpha=0.05):
    """Nemenyi critical difference (Demsar 2006)."""
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
    }
    if k not in q_table:
        raise ValueError(f"q-value not available for k={k} classifiers (supported: 2–12)")
    q = q_table[k]
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return cd


# ---------------------------------------------------------------------------
# CD diagram
# ---------------------------------------------------------------------------


def plot_cd_diagram(avg_ranks, cd, title="Critical Difference Diagram"):
    """Demsar-style CD diagram: labels on left/right sides, no overlap."""
    k = len(avg_ranks)
    names = avg_ranks.index.tolist()
    ranks = avg_ranks.values

    sorted_idx = np.argsort(ranks)
    sorted_ranks = ranks[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    # -- Layout constants (in axis coordinates) --
    # Better-ranked classifiers get labels on the LEFT (top-down),
    # worse-ranked classifiers get labels on the RIGHT (top-down).
    mid = (k + 1) // 2  # left side gets one more if odd
    left_items = list(range(mid))           # indices into sorted arrays
    right_items = list(range(mid, k))

    label_spacing = 0.12        # vertical gap between stacked labels
    axis_y = 0.50               # y-position of the rank axis
    left_x = 0.6                # x where left labels are placed
    right_x = k + 0.4           # x where right labels are placed
    top_start = axis_y + 0.15 + (max(len(left_items), len(right_items)) - 1) * label_spacing

    fig_h = max(3.5, 1.5 + k * 0.28)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, fig_h))
    ax.set_xlim(0.0, k + 1.0)
    y_top = top_start + 0.30
    y_bot = axis_y - 0.15 - 0.10  # room for cliques below

    # Find cliques first so we know how much bottom space to reserve
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                merged = False
                for c in cliques:
                    if i in c or j in c:
                        c.add(i)
                        c.add(j)
                        merged = True
                if not merged:
                    cliques.append({i, j})
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
    n_cliques = len(merged_cliques)
    y_bot = axis_y - 0.20 - n_cliques * 0.10

    ax.set_ylim(y_bot - 0.10, y_top)

    # -- Rank axis --
    ax.hlines(axis_y, 1, k, color="black", linewidth=1.0)
    for r in range(1, k + 1):
        ax.vlines(r, axis_y - 0.03, axis_y + 0.03, color="black", linewidth=0.8)
        ax.text(r, axis_y - 0.07, str(r), ha="center", va="top", fontsize=7)

    # -- CD bar at top --
    cd_y = top_start + 0.12
    ax.hlines(cd_y, 1, 1 + cd, color="red", linewidth=2.5)
    ax.text(1 + cd / 2, cd_y + 0.04, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=8, color="red", fontweight="bold")

    # -- Place labels on LEFT side (better classifiers, top-down) --
    for pos, idx in enumerate(left_items):
        name = sorted_names[idx]
        rank_val = sorted_ranks[idx]
        label_y = top_start - pos * label_spacing
        label = f"{name} ({rank_val:.2f})"

        # Horizontal line from label to rank, then vertical drop to axis
        ax.hlines(label_y, left_x, rank_val, color="gray", linewidth=0.5)
        ax.vlines(rank_val, axis_y, label_y, color="gray", linewidth=0.5)
        ax.plot(rank_val, axis_y, "o", color="black", markersize=4, zorder=5)
        ax.text(left_x - 0.05, label_y, label,
                ha="right", va="center", fontsize=7.5)

    # -- Place labels on RIGHT side (worse classifiers, top-down) --
    for pos, idx in enumerate(right_items):
        name = sorted_names[idx]
        rank_val = sorted_ranks[idx]
        label_y = top_start - pos * label_spacing
        label = f"{name} ({rank_val:.2f})"

        ax.hlines(label_y, rank_val, right_x, color="gray", linewidth=0.5)
        ax.vlines(rank_val, axis_y, label_y, color="gray", linewidth=0.5)
        ax.plot(rank_val, axis_y, "o", color="black", markersize=4, zorder=5)
        ax.text(right_x + 0.05, label_y, label,
                ha="left", va="center", fontsize=7.5)

    # -- Clique bars below axis --
    for ci, c in enumerate(merged_cliques):
        members = sorted(c)
        r_min = sorted_ranks[members[0]]
        r_max = sorted_ranks[members[-1]]
        y = axis_y - 0.18 - ci * 0.10
        ax.hlines(y, r_min, r_max, color="steelblue", linewidth=3.5, alpha=0.7)

    ax.set_title(title, pad=12)
    ax.axis("off")
    fig.tight_layout()
    _savefig(fig, "cd_diagram.pdf")


# ---------------------------------------------------------------------------
# Pairwise wins heatmap
# ---------------------------------------------------------------------------


def plot_pairwise_wins(mean_acc):
    classifiers = mean_acc.columns.tolist()
    n_clf = len(classifiers)
    n_ds = len(mean_acc)
    wins = pd.DataFrame(0, index=classifiers, columns=classifiers)

    for ds in mean_acc.index:
        for i, ci in enumerate(classifiers):
            for j, cj in enumerate(classifiers):
                if i == j:
                    continue
                vi = mean_acc.loc[ds, ci]
                vj = mean_acc.loc[ds, cj]
                if pd.notna(vi) and pd.notna(vj) and vi > vj:
                    wins.loc[ci, cj] += 1

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    # Mask diagonal
    mask = np.eye(n_clf, dtype=bool)
    sns.heatmap(
        wins, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
        mask=mask, linewidths=0.5, center=n_ds / 2,
        annot_kws={"size": 7},
        cbar_kws={"label": "Dataset wins", "shrink": 0.7},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title("Pairwise Dataset Wins (row beats column)")
    fig.tight_layout()
    _savefig(fig, "pairwise_wins_heatmap.pdf")


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-metric analysis
# ---------------------------------------------------------------------------


def multi_metric_summary_table(df):
    """CSV + LaTeX table: mean ± std for every metric, per classifier."""
    classifiers = (CLASSIFIERS_TO_ANALYZE
                   if CLASSIFIERS_TO_ANALYZE is not None
                   else sorted(df["classifier"].unique()))
    rows = []
    for clf in classifiers:
        sub = df[df["classifier"] == clf]
        row = {"Classifier": clf}
        for m in ALL_METRICS:
            if m in sub.columns:
                mean_vals = sub.groupby("dataset")[m].mean()
                row[f"{METRIC_LABELS[m]} mean"] = mean_vals.mean()
                row[f"{METRIC_LABELS[m]} std"] = mean_vals.std()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("Classifier")
    summary.to_csv(os.path.join(OUTPUT_DIR, "multi_metric_summary.csv"))
    print("  Saved multi_metric_summary.csv")

    _write_multi_metric_latex(summary, classifiers)
    return summary


def _write_multi_metric_latex(summary, classifiers):
    """LaTeX booktabs table with all metrics, bold best per column."""
    metric_cols = [c for c in summary.columns if c.endswith(" mean")]
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean performance across datasets for all evaluation metrics. "
                 r"\textbf{Bold} marks the best result per metric.}")
    lines.append(r"\label{tab:multi_metric}")
    col_fmt = "l" + "c" * len(metric_cols)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header
    header_names = [c.replace(" mean", "") for c in metric_cols]
    header = "Classifier & " + " & ".join(header_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per column
    best = {c: summary[c].idxmax() for c in metric_cols}

    for clf in classifiers:
        cells = []
        for c in metric_cols:
            std_col = c.replace(" mean", " std")
            m = summary.loc[clf, c]
            s = summary.loc[clf, std_col] if std_col in summary.columns else 0
            txt = f"{m:.3f}"
            if clf == best[c]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(f"{clf} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path = os.path.join(OUTPUT_DIR, "multi_metric_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved multi_metric_table.tex")


def plot_metric_rank_stability(df):
    """Bump chart: classifier ranks under each metric. Reveals if rankings are
    metric-sensitive or stable across evaluation criteria."""
    classifiers = (CLASSIFIERS_TO_ANALYZE
                   if CLASSIFIERS_TO_ANALYZE is not None
                   else sorted(df["classifier"].unique()))
    metrics_present = [m for m in ALL_METRICS if m in df.columns]

    rank_data = {}
    for m in metrics_present:
        mean_per_clf = df.groupby("classifier")[m].mean()
        rank_data[m] = mean_per_clf.rank(ascending=False)

    rank_df = pd.DataFrame(rank_data).reindex(classifiers)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, max(3.5, len(classifiers) * 0.35)))

    cmap = plt.cm.tab10
    x_positions = np.arange(len(metrics_present))

    for i, clf in enumerate(classifiers):
        ranks = rank_df.loc[clf].values
        color = cmap(i % 10)
        ax.plot(x_positions, ranks, "o-", color=color, linewidth=1.2,
                markersize=5, label=_short(clf), zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_present], fontsize=8)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Classifier Rank Stability Across Metrics")
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(classifiers) + 1))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              borderaxespad=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    _savefig(fig, "metric_rank_stability.pdf")


def plot_accuracy_vs_balanced(df):
    """Scatter: accuracy vs balanced accuracy per (classifier, dataset).
    Points far from the diagonal indicate class-imbalance sensitivity."""
    metrics_present = [m for m in ["accuracy", "balanced_accuracy"] if m in df.columns]
    if len(metrics_present) < 2:
        return

    grouped = df.groupby(["classifier", "dataset"])[["accuracy", "balanced_accuracy"]].mean()
    grouped = grouped.reset_index()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))

    classifiers = (CLASSIFIERS_TO_ANALYZE
                   if CLASSIFIERS_TO_ANALYZE is not None
                   else sorted(grouped["classifier"].unique()))
    cmap = plt.cm.tab10

    for i, clf in enumerate(classifiers):
        sub = grouped[grouped["classifier"] == clf]
        ax.scatter(sub["accuracy"], sub["balanced_accuracy"],
                   s=20, alpha=0.7, color=cmap(i % 10), label=_short(clf),
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Diagonal reference
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.6, zorder=1)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Accuracy vs Balanced Accuracy")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.tight_layout()
    _savefig(fig, "accuracy_vs_balanced.pdf")


def plot_metric_correlation(df):
    """Heatmap of Spearman rank correlations between metrics, computed over
    (classifier, dataset) pairs. Shows whether metrics provide redundant or
    complementary information."""
    metrics_present = [m for m in ALL_METRICS if m in df.columns]
    if len(metrics_present) < 2:
        return

    grouped = df.groupby(["classifier", "dataset"])[metrics_present].mean().reset_index()
    corr = grouped[metrics_present].corr(method="spearman")
    corr.index = [METRIC_LABELS[m] for m in corr.index]
    corr.columns = [METRIC_LABELS[m] for m in corr.columns]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.1))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", ax=ax,
                vmin=0.8, vmax=1.0, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})
    ax.set_title("Metric Rank Correlation")
    fig.tight_layout()
    _savefig(fig, "metric_correlation.pdf")


def plot_dataset_metric_profiles(df):
    """Heatmap showing per-dataset, per-metric mean across classifiers.
    Normalised per metric (z-score) to reveal which datasets are relatively
    harder or easier depending on the evaluation criterion."""
    metrics_present = [m for m in ALL_METRICS if m in df.columns]
    if len(metrics_present) < 2:
        return

    ds_means = df.groupby("dataset")[metrics_present].mean()
    # Z-score normalise per metric for comparability
    ds_z = (ds_means - ds_means.mean()) / ds_means.std()
    ds_z.columns = [METRIC_LABELS[m] for m in ds_z.columns]
    ds_z = ds_z.sort_index()

    n_ds = len(ds_z)
    fig_h = max(3.5, n_ds * 0.35)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, fig_h))
    sns.heatmap(ds_z, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                center=0, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "z-score"})
    ax.set_title("Dataset Difficulty by Metric (z-score)")
    ax.set_ylabel("")
    fig.tight_layout()
    _savefig(fig, "dataset_metric_profiles.pdf")


def plot_kappa_vs_accuracy(df):
    """Scatter: Cohen's kappa vs accuracy per classifier (averaged across datasets).
    Kappa corrects for chance agreement, so divergence from the diagonal reveals
    classifiers that benefit from class priors."""
    if "cohen_kappa" not in df.columns:
        return

    grouped = df.groupby("classifier")[["accuracy", "cohen_kappa"]].mean()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.2))
    ax.scatter(grouped["accuracy"], grouped["cohen_kappa"],
               s=40, color="steelblue", edgecolors="white",
               linewidths=0.5, zorder=3)

    texts = []
    for clf in grouped.index:
        texts.append(ax.text(grouped.loc[clf, "accuracy"],
                             grouped.loc[clf, "cohen_kappa"],
                             _short(clf), fontsize=6))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Mean Cohen's κ")
    ax.set_title("Accuracy vs Cohen's Kappa")
    fig.tight_layout()
    _savefig(fig, "kappa_vs_accuracy.pdf")


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------


def write_summary_report(df, mean_acc, std_acc):
    path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    classifiers = mean_acc.columns.tolist()
    datasets = mean_acc.index.tolist()

    with open(path, "w") as f:
        f.write(f"Benchmark Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Classifiers: {len(classifiers)}  |  Datasets: {len(datasets)}\n")
        f.write("=" * 70 + "\n\n")

        # Mean accuracy across all datasets
        f.write("MEAN ACCURACY ACROSS ALL DATASETS\n")
        f.write("-" * 50 + "\n")
        global_mean = mean_acc.mean(axis=0).sort_values(ascending=False)
        global_std = mean_acc.std(axis=0)
        for clf in global_mean.index:
            f.write(f"  {clf:25s}  {global_mean[clf]:.4f} +/- {global_std[clf]:.4f}\n")

        best_clf = global_mean.idxmax()
        worst_clf = global_mean.idxmin()
        f.write(f"\n  Best:  {best_clf} ({global_mean[best_clf]:.4f})\n")
        f.write(f"  Worst: {worst_clf} ({global_mean[worst_clf]:.4f})\n")

        # Best / worst dataset per algorithm
        f.write(f"\n\nBEST AND WORST DATASET PER ALGORITHM\n")
        f.write("-" * 50 + "\n")
        for clf in classifiers:
            col = mean_acc[clf].dropna()
            if col.empty:
                continue
            best_ds = col.idxmax()
            worst_ds = col.idxmin()
            f.write(f"  {clf:25s}  best: {best_ds} ({col[best_ds]:.4f})"
                    f"  worst: {worst_ds} ({col[worst_ds]:.4f})\n")

        # Best algorithm per dataset
        f.write(f"\n\nBEST ALGORITHM PER DATASET\n")
        f.write("-" * 50 + "\n")
        for ds in datasets:
            winner = mean_acc.loc[ds].idxmax()
            f.write(f"  {ds:35s}  {winner} ({mean_acc.loc[ds, winner]:.4f})\n")

        # Wins
        f.write(f"\n\nDATASET WINS PER ALGORITHM\n")
        f.write("-" * 50 + "\n")
        wins = mean_acc.idxmax(axis=1).value_counts()
        for clf in classifiers:
            f.write(f"  {clf:25s}  {wins.get(clf, 0)} wins\n")

        # Timing
        f.write(f"\n\nMEAN FIT TIME (seconds)\n")
        f.write("-" * 50 + "\n")
        fit_times = df.groupby("classifier")["fit_time_s"].mean().sort_values()
        for clf, t in fit_times.items():
            f.write(f"  {clf:25s}  {t:.4f}s\n")

        f.write(f"\n\nMEAN PREDICT TIME (seconds)\n")
        f.write("-" * 50 + "\n")
        predict_times = df.groupby("classifier")["predict_time_s"].mean().sort_values()
        for clf, t in predict_times.items():
            f.write(f"  {clf:25s}  {t:.4f}s\n")

        # Dataset difficulty
        f.write(f"\n\nDATASET DIFFICULTY (mean accuracy across algorithms)\n")
        f.write("-" * 50 + "\n")
        ds_mean = mean_acc.mean(axis=1).sort_values(ascending=False)
        for ds in ds_mean.index:
            f.write(f"  {ds:35s}  {ds_mean[ds]:.4f}\n")
        f.write(f"\n  Easiest: {ds_mean.idxmax()} ({ds_mean.max():.4f})\n")
        f.write(f"  Hardest: {ds_mean.idxmin()} ({ds_mean.min():.4f})\n")

    print(f"  Saved summary_report.txt")


def write_statistical_report(mean_acc, ranks, avg_ranks, fstat, fpval, cd):
    path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
    k = len(avg_ranks)
    n = len(mean_acc)

    with open(path, "w") as f:
        f.write(f"Statistical Tests Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        n_nan = mean_acc.isna().sum().sum()
        if n_nan > 0:
            f.write(f"Note: {n_nan} missing value(s) assigned worst rank "
                    f"({k}) as penalty.\n\n")

        f.write("FRIEDMAN TEST\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Statistic : {fstat:.4f}\n")
        f.write(f"  p-value   : {fpval:.6f}\n")
        f.write(f"  Significant (p < 0.05): {fpval < 0.05}\n\n")

        f.write("AVERAGE RANKS (lower is better)\n")
        f.write("-" * 50 + "\n")
        for name, rank in avg_ranks.items():
            f.write(f"  {name:25s}  {rank:.3f}\n")

        f.write(f"\n\nNEMENYI POST-HOC TEST\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Critical Difference (CD): {cd:.4f}\n")
        f.write(f"  k = {k} classifiers, N = {n} datasets, alpha = 0.05\n\n")

        f.write("PAIRWISE RANK DIFFERENCES\n")
        f.write("-" * 50 + "\n")
        names = avg_ranks.index.tolist()
        sig_pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j])
                sig = "*" if diff > cd else ""
                f.write(f"  {names[i]:20s} vs {names[j]:20s}  "
                        f"diff = {diff:.3f} {sig}\n")
                if diff > cd:
                    sig_pairs.append((names[i], names[j], diff))

        f.write(f"\n\nSIGNIFICANT DIFFERENCES (diff > CD = {cd:.4f})\n")
        f.write("-" * 50 + "\n")
        if sig_pairs:
            for a, b, d in sig_pairs:
                f.write(f"  {a:20s} vs {b:20s}  diff = {d:.3f}\n")
        else:
            f.write("  No significant pairwise differences found.\n")

    print(f"  Saved statistical_tests.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    plt.rcParams.update(RCPARAMS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data …")
    df = load_results()
    print(f"  {len(df)} rows, {df['classifier'].nunique()} classifiers, "
          f"{df['dataset'].nunique()} datasets\n")

    print("Tables:")
    mean_acc, std_acc = mean_accuracy_table(df)
    # NEW: Call the multi-metric table
    multi_metric_summary_table(df)

    print("\nPlots:")
    plot_heatmap(mean_acc)
    plot_boxplots(df)
    plot_timing(df)
    plot_pairwise_wins(mean_acc)

    # NEW: Call all the new multi-metric plots
    print("\nMulti-Metric Plots:")
    plot_metric_rank_stability(df)
    plot_accuracy_vs_balanced(df)
    plot_metric_correlation(df)
    plot_dataset_metric_profiles(df)
    plot_kappa_vs_accuracy(df)

    print("\nStatistical tests:")
    ranks, avg_ranks = compute_ranks(mean_acc)
    k = len(mean_acc.columns)
    n = len(mean_acc)
    fstat, fpval = friedman_test(ranks)
    cd = nemenyi_cd(k, n)
    print(f"  Friedman: stat={fstat:.4f}, p={fpval:.6f}")
    print(f"  Nemenyi CD = {cd:.4f}")

    plot_cd_diagram(avg_ranks, cd)

    print("\nReports:")
    write_summary_report(df, mean_acc, std_acc)
    write_statistical_report(mean_acc, ranks, avg_ranks, fstat, fpval, cd)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
