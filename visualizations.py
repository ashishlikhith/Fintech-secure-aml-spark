"""Create compact charts from Spark summaries and held-out predictions."""

from pathlib import Path
import math
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Visualization saved: {output_path}")


def _plot_transaction_overview(base_df: DataFrame, output_dir: Path) -> None:
    label_counts = (
        base_df.groupBy("is_laundering")
        .count()
        .orderBy("is_laundering")
        .toPandas()
    )
    daily = (
        base_df.groupBy(F.to_date("timestamp").alias("date"))
        .agg(
            F.count("*").alias("transactions"),
            F.count("is_laundering").alias("labeled_transactions"),
            F.sum("is_laundering").alias("flagged"),
        )
        .withColumn(
            "flag_rate", F.col("flagged") / F.col("labeled_transactions")
        )
        .orderBy("date")
        .toPandas()
    )
    payment_types = (
        base_df.groupBy("payment_type")
        .agg(
            F.count("*").alias("transactions"),
            F.avg("is_laundering").alias("flag_rate"),
        )
        .orderBy(F.desc("transactions"))
        .limit(12)
        .toPandas()
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels = label_counts["is_laundering"].fillna(-1).astype(str).replace(
        {"0": "Not flagged", "1": "Flagged", "-1": "Unlabeled"}
    )
    sns.barplot(
        x=labels,
        y=label_counts["count"],
        hue=labels,
        legend=False,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set(title="Label balance", xlabel="", ylabel="Transactions")

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        sns.lineplot(data=daily, x="date", y="transactions", ax=axes[1], color="#3478a8")
        axes[1].set(title="Transaction volume over time", xlabel="Date")
        rate_axis = axes[1].twinx()
        sns.lineplot(
            data=daily,
            x="date",
            y="flag_rate",
            ax=rate_axis,
            color="#d1495b",
            label="Flagged share",
        )
        rate_axis.set_ylabel("Flagged share")
        rate_axis.set_ylim(0, 1)

    if not payment_types.empty:
        payment_types = payment_types.sort_values("flag_rate", ascending=False)
        sns.barplot(
            data=payment_types,
            x="flag_rate",
            y="payment_type",
            hue="payment_type",
            legend=False,
            ax=axes[2],
            palette="mako",
        )
        axes[2].set(
            title="Flagged share by payment type",
            xlabel="Flagged share",
            ylabel="Payment type",
            xlim=(0, 1),
        )

    _save_figure(figure, output_dir / "transaction_overview.png")


def _plot_model_metrics(
    model_results: Mapping[str, object], output_dir: Path
) -> None:
    comparison = model_results["comparison"]
    if comparison is None:
        print("Skipping model metric plots: no classifiers were evaluated.")
        return
    metrics = comparison.toPandas()
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, 5))
    metric_columns = ["precision", "recall", "f1", "area_under_pr"]
    melted = metrics.melt(
        id_vars="model", value_vars=metric_columns, var_name="metric", value_name="score"
    )
    sns.barplot(
        data=melted,
        x="metric",
        y="score",
        hue="model",
        ax=axis,
        palette="Set2",
    )
    axis.set(
        title="Future-period model comparison (positive-class metrics)",
        xlabel="Metric",
        ylabel="Score",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=20)
    _save_figure(figure, output_dir / "model_comparison.png")


def _plot_confusion_matrices(
    model_results: Mapping[str, object], output_dir: Path
) -> None:
    classification_models = [
        name
        for name in (
            "Decision Tree",
            "Random Forest",
            "Gradient-Boosted Trees",
        )
        if name in model_results["predictions"]
    ]
    if not classification_models:
        return

    figure, axes = plt.subplots(1, len(classification_models), figsize=(6 * len(classification_models), 5))
    if len(classification_models) == 1:
        axes = [axes]
    for axis, name in zip(axes, classification_models):
        predictions = model_results["predictions"][name]
        matrix = (
            predictions.groupBy("label", "prediction")
            .count()
            .toPandas()
            .pivot(index="label", columns="prediction", values="count")
            .fillna(0)
        )
        matrix = matrix.reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["Not flagged", "Flagged"],
            yticklabels=["Not flagged", "Flagged"],
        )
        axis.set(
            title=f"{name} confusion matrix (future holdout)",
            xlabel="Predicted",
            ylabel="Actual",
        )
    _save_figure(figure, output_dir / "confusion_matrices.png")


def _plot_feature_importance(
    model_results: Mapping[str, object], output_dir: Path
) -> None:
    model_names = [
        name
        for name in ("Decision Tree", "Random Forest", "Gradient-Boosted Trees")
        if name in model_results["models"]
    ]
    feature_names = model_results["feature_names"]
    figure, axes = plt.subplots(
        1, len(model_names), figsize=(6 * len(model_names), 7), squeeze=False
    )
    for axis, name in zip(axes[0], model_names):
        model = model_results["models"][name]
        importances = model.featureImportances.toArray()
        if len(importances) != len(feature_names):
            raise ValueError(
                f"{name} importance count does not match the assembled features."
            )
        ranked = sorted(
            zip(feature_names, importances), key=lambda item: item[1], reverse=True
        )[:12]
        labels = [item[0] for item in reversed(ranked)]
        values = [float(item[1]) for item in reversed(ranked)]
        axis.barh(labels, values, color="#3478a8")
        axis.set(title=name, xlabel="Feature importance", ylabel="")
    figure.suptitle("Top model feature importances")
    _save_figure(figure, output_dir / "feature_importance.png")


def _plot_transaction_flow_map(base_df: DataFrame, output_dir: Path) -> None:
    edges = (
        base_df.groupBy("sender_account", "receiver_account")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_laundering").alias("flagged"),
        )
        .orderBy(F.desc("transactions"))
        .limit(30)
        .toPandas()
    )
    if edges.empty:
        return

    accounts = sorted(set(edges["sender_account"]) | set(edges["receiver_account"]))
    angle_step = 2 * 3.141592653589793 / len(accounts)
    positions = {
        account: (
            1.4 * math.cos(index * angle_step),
            1.4 * math.sin(index * angle_step),
        )
        for index, account in enumerate(accounts)
    }
    figure, axis = plt.subplots(figsize=(11, 9))
    maximum = max(float(edges["transactions"].max()), 1.0)
    for row in edges.itertuples(index=False):
        x1, y1 = positions[row.sender_account]
        x2, y2 = positions[row.receiver_account]
        flagged_share = float(row.flagged or 0) / max(int(row.transactions), 1)
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": "#c84c4c" if flagged_share > 0 else "#7695a5",
                "alpha": 0.25 + 0.65 * flagged_share,
                "linewidth": 0.5 + 3 * int(row.transactions) / maximum,
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )
    xs, ys = zip(*(positions[account] for account in accounts))
    axis.scatter(xs, ys, s=700, color="#356d8a", edgecolor="white", zorder=3)
    for account, (x, y) in positions.items():
        axis.text(x, y, account, ha="center", va="center", color="white", fontsize=7, zorder=4)
    axis.set(
        title="Transaction-flow map (top 30 account pairs; red = labeled activity)",
        aspect="equal",
    )
    axis.axis("off")
    _save_figure(figure, output_dir / "transaction_flow_map.png")


def create_pipeline_visualizations(
    base_df: DataFrame,
    model_results: Mapping[str, object],
    output_dir: str,
) -> None:
    """Write bounded-size overview, model-evaluation, and flow-map plots."""
    sns.set_theme(style="whitegrid")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _plot_transaction_overview(base_df, output_path)
    _plot_model_metrics(model_results, output_path)
    _plot_confusion_matrices(model_results, output_path)
    _plot_feature_importance(model_results, output_path)
    _plot_transaction_flow_map(base_df, output_path)
