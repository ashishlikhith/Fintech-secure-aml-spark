"""Create AML distribution and model evaluation plots."""

from pathlib import Path
from typing import Any

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame


PLOT_FILENAMES = [
    "aml_class_distribution.png",
    "correlation_heatmap.png",
    "precision_recall_curves.png",
    "roc_curves.png",
    "top10_feature_importance.png",
    "model_comparison.png",
    "precision_recall_f1_comparison.png",
    "confusion_matrices.png",
]

PLOT_TITLES = [
    "AML class distribution",
    "Correlation heatmap",
    "Precision-recall curves",
    "ROC curves",
    "Top-10 feature importance",
    "Model comparison",
    "Precision/recall/F1 comparison",
    "Confusion matrices",
]


def require_interactive_backend() -> None:
    """Fail clearly unless Matplotlib has a desktop-window backend available."""
    backend = matplotlib.get_backend().lower()
    window_backends = {
        "gtk3agg",
        "gtk3cairo",
        "gtk4agg",
        "gtk4cairo",
        "macosx",
        "qtagg",
        "qtcairo",
        "qt5agg",
        "qt5cairo",
        "tkagg",
        "tkcairo",
        "wx",
        "wxagg",
        "wxcairo",
    }
    if backend not in window_backends:
        raise RuntimeError(
            f"Interactive chart windows require a GUI Matplotlib backend; "
            f"the selected backend is {matplotlib.get_backend()!r}. On Windows, "
            "configure TkAgg (built into this Python installation) or QtAgg "
            "with a compatible Qt package; do not use Agg."
        )


def display_saved_visualizations(output_dir: str) -> list[Path]:
    """Display the eight existing PNGs in order without training models again."""
    require_interactive_backend()
    output_path = Path(output_dir)
    plot_paths = [output_path / filename for filename in PLOT_FILENAMES]
    missing = [path for path in plot_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot display saved charts; missing files: "
            + ", ".join(str(path) for path in missing)
        )

    for plot_path, title in zip(plot_paths, PLOT_TITLES):
        figure, axis = plt.subplots()
        try:
            axis.imshow(plt.imread(plot_path))
            axis.set_title(title)
            axis.axis("off")
            figure.tight_layout()
            print(f"Displaying saved chart: {plot_path}")
            plt.show(block=True)
        finally:
            plt.close(figure)
    return plot_paths


def _curve_points(
    labels: list[int], scores: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build tied-score-aware precision-recall and ROC curve coordinates."""
    label_values = np.asarray(labels, dtype=np.int8)
    score_values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_labels = label_values[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(sorted_labels == 1)[group_ends]
    false_positives = group_ends + 1 - true_positives
    positive_count = int(np.count_nonzero(label_values == 1))
    negative_count = len(label_values) - positive_count
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    true_positive_rate = true_positives / positive_count
    false_positive_rate = false_positives / negative_count
    return (
        np.r_[0.0, recall],
        np.r_[1.0, precision],
        np.r_[0.0, false_positive_rate, 1.0],
        np.r_[0.0, true_positive_rate, 1.0],
    )


def create_visualizations(
    transactions: DataFrame,
    model_metrics: list[dict[str, Any]],
    output_dir: str,
    show_interactive: bool = True,
) -> list[Path]:
    """Save eight distinct class balance, feature, and model evaluation plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if show_interactive:
        require_interactive_backend()
    sns.set_theme(style="whitegrid")
    saved_paths: list[Path] = []

    def save_plot(figure: plt.Figure, filename: str) -> None:
        plot_path = output_path / filename
        try:
            figure.savefig(plot_path, dpi=150)
            if show_interactive:
                print(f"Displaying chart: {plot_path}")
                plt.show(block=True)
        finally:
            plt.close(figure)
        saved_paths.append(plot_path)
        print(f"Saved plot: {plot_path}")

    counts = transactions.groupBy("is_laundering").count().toPandas()
    counts["class"] = counts["is_laundering"].map(
        {0: "Not laundering", 1: "Laundering"}
    )
    total_rows = int(counts["count"].sum())
    counts["percentage"] = counts["count"] / total_rows * 100
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = sns.barplot(
        data=counts, x="class", y="count", hue="class", legend=False, ax=axis
    )
    for bar, (_, row) in zip(bars.patches, counts.iterrows()):
        bar.axes.annotate(
            f"{int(row['count']):,}\n({row['percentage']:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    axis.set(title="IBM AML transaction class distribution", xlabel="", ylabel="Rows")
    figure.tight_layout()
    save_plot(figure, "aml_class_distribution.png")

    correlation_sample = (
        transactions.select("amount", "is_near_threshold")
        .sample(False, 0.1, seed=42)
        .limit(20_000)
        .toPandas()
    )
    if correlation_sample.empty:
        correlation_sample = transactions.select(
            "amount", "is_near_threshold"
        ).limit(20_000).toPandas()
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        correlation_sample.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        center=0,
        ax=axis,
    )
    axis.set_title("Correlation of simple numeric transaction features")
    figure.tight_layout()
    save_plot(figure, "correlation_heatmap.png")

    for curve_name, filename, x_label, y_label, title in (
        (
            "precision_recall",
            "precision_recall_curves.png",
            "Recall",
            "Precision",
            "Precision–recall curves on the chronological test split",
        ),
        (
            "roc",
            "roc_curves.png",
            "False positive rate",
            "True positive rate",
            "ROC curves on the chronological test split",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 5))
        for result in model_metrics:
            recall, precision, false_positive_rate, true_positive_rate = (
                _curve_points(
                    result["_test_labels"], result["_test_probabilities"]
                )
            )
            if curve_name == "precision_recall":
                axis.plot(recall, precision, label=str(result["model"]))
            else:
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    label=str(result["model"]),
                )
        if curve_name == "roc":
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        else:
            prevalence = float(np.mean(model_metrics[0]["_test_labels"]))
            axis.axhline(
                prevalence,
                linestyle="--",
                color="gray",
                label=f"Test prevalence ({prevalence:.3%})",
            )
        axis.set(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(loc="best")
        figure.tight_layout()
        save_plot(figure, filename)

    forest_result = next(
        (
            result
            for result in model_metrics
            if "_feature_importances" in result
        ),
        None,
    )
    if forest_result is None:
        raise ValueError("Random Forest feature importances are required for plotting.")
    importance = pd.DataFrame(
        [
            {"feature": name, "importance": value}
            for name, value in forest_result["_feature_importances"].items()
        ]
    ).sort_values("importance", ascending=False).head(10)
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=importance,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        ax=axis,
    )
    axis.set(
        title="Top feature importances (RandomForestClassifier)",
        xlabel="Importance",
        ylabel="Feature",
    )
    figure.tight_layout()
    save_plot(figure, "top10_feature_importance.png")

    for metric_columns, filename, title in (
        (
            ["accuracy", "pr_auc", "roc_auc"],
            "model_comparison.png",
            "Accuracy and ranking metric comparison",
        ),
        (
            ["precision", "recall", "f1"],
            "precision_recall_f1_comparison.png",
            "Precision, recall, and F1 comparison",
        ),
    ):
        comparison = pd.DataFrame(model_metrics).melt(
            id_vars="model",
            value_vars=metric_columns,
            var_name="metric",
            value_name="score",
        )
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(data=comparison, x="metric", y="score", hue="model", ax=axis)
        axis.set(title=title, xlabel="Metric", ylabel="Score", ylim=(0, 1))
        axis.legend(title="Model")
        figure.tight_layout()
        save_plot(figure, filename)

    figure, axes = plt.subplots(
        1, len(model_metrics), figsize=(5 * len(model_metrics), 4)
    )
    if len(model_metrics) == 1:
        axes = [axes]
    for axis, metrics in zip(axes, model_metrics):
        matrix = pd.DataFrame(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            index=["Actual negative", "Actual positive"],
            columns=["Predicted negative", "Predicted positive"],
        )
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(str(metrics["model"]))
        axis.set_xlabel("Prediction")
        axis.set_ylabel("Actual")
    figure.suptitle("Confusion matrices on the chronological test split")
    figure.tight_layout()
    save_plot(figure, "confusion_matrices.png")
    return saved_paths
