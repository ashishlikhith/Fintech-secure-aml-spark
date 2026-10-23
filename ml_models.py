"""Chronological data splits and three simple Spark ML classifiers."""

import csv
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


MODEL_FEATURES = ["amount", "is_near_threshold", "payment_type_index"]
MODEL_NAMES = ["DecisionTreeClassifier", "RandomForestClassifier", "GBTClassifier"]
METRIC_COLUMNS = [
    ("Model", "model"),
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1"),
    ("PR-AUC", "pr_auc"),
    ("ROC-AUC", "roc_auc"),
    ("TP", "tp"),
    ("TN", "tn"),
    ("FP", "fp"),
    ("FN", "fn"),
    ("Threshold", "threshold"),
]


def _class_counts(data: DataFrame) -> dict[float, int]:
    return {
        float(row["label"]): int(row["count"])
        for row in data.groupBy("label").count().collect()
    }


def _chronological_splits(
    data: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Use timestamp quantiles so later transactions never train the models."""
    epoch = F.unix_timestamp("timestamp")
    cutoffs = data.select(epoch.alias("epoch")).approxQuantile(
        "epoch", [0.6, 0.8], 0.001
    )
    if len(cutoffs) != 2:
        raise ValueError("Could not calculate chronological split dates.")
    first_cutoff, second_cutoff = cutoffs
    return (
        data.filter(epoch < first_cutoff),
        data.filter((epoch >= first_cutoff) & (epoch < second_cutoff)),
        data.filter(epoch >= second_cutoff),
    )


def _split_stats(counts: dict[float, int]) -> dict[str, int]:
    positive = counts.get(1.0, 0)
    negative = counts.get(0.0, 0)
    return {
        "rows": positive + negative,
        "positive": positive,
        "negative": negative,
    }


def _validate_split(name: str, counts: dict[float, int]) -> None:
    if not counts.get(0.0) or not counts.get(1.0):
        raise ValueError(
            f"{name} split needs both AML labels; observed class counts: {counts}"
        )


def print_data_usage_report(
    source_stats: dict[str, int] | None,
    sample_size: int,
    sample_seed: int,
    sample_counts: dict[float, int],
    split_stats: dict[str, dict[str, int]],
    model_metrics: list[dict[str, Any]] | None = None,
    total_runtime_seconds: float | None = None,
) -> None:
    """Print the measured source, sample, split, feature, and model data usage."""
    phase = "final" if model_metrics is not None else "before model training"
    print(f"\nDATA USAGE REPORT ({phase})")
    if source_stats is None:
        print("Source row count and source class counts: unavailable.")
    else:
        source_rows = source_stats["source_rows"]
        source_positive = source_stats["positive"]
        source_negative = source_stats["negative"]
        labelled_rows = source_positive + source_negative
        print(f"Source row count: {source_rows:,}")
        print(
            f"Source labeled AML-positive: {source_positive:,} "
            f"({source_positive / labelled_rows:.4%}); AML-negative: "
            f"{source_negative:,} ({source_negative / labelled_rows:.4%})."
        )

    sample_positive = int(sample_counts.get(1.0, 0))
    sample_negative = int(sample_counts.get(0.0, 0))
    sample_rows = sample_positive + sample_negative
    print(f"Sample row count: {sample_rows:,}; requested size: {sample_size:,}; seed: {sample_seed}.")
    print(
        f"Sample AML-positive: {sample_positive:,} "
        f"({sample_positive / sample_rows:.4%}); AML-negative: "
        f"{sample_negative:,} ({sample_negative / sample_rows:.4%})."
    )
    for split_name in ("train", "validation", "test"):
        stats = split_stats[split_name]
        prevalence = stats["positive"] / stats["rows"] if stats["rows"] else 0
        print(
            f"{split_name.title()}: {stats['rows']:,} rows; "
            f"AML-positive={stats['positive']:,} ({prevalence:.4%}); "
            f"AML-negative={stats['negative']:,}."
        )
    print(
        f"Model features ({len(MODEL_FEATURES)}): "
        + ", ".join(MODEL_FEATURES)
    )
    test_rows = split_stats["test"]["rows"]
    if model_metrics is None:
        for name in MODEL_NAMES:
            print(f"{name}: {test_rows:,} test rows to evaluate; training time pending.")
        print("Total runtime: in progress.")
    else:
        for result in model_metrics:
            print(
                f"{result['model']}: {result['rows_evaluated']:,} rows evaluated; "
                f"training time={result['training_seconds']:.2f} seconds."
            )
        if total_runtime_seconds is None:
            print("Total runtime: not yet available.")
        else:
            print(f"Total runtime: {total_runtime_seconds:.2f} seconds.")


def _collect_scores(predictions: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = predictions.select("label", "positive_probability").collect()
    labels = np.fromiter((int(row["label"]) for row in rows), dtype=np.int8)
    scores = np.fromiter(
        (float(row["positive_probability"]) for row in rows), dtype=np.float64
    )
    return labels, scores


def _best_validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose the validation probability cutoff with the highest exact F1."""
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Validation labels and probability scores must be non-empty.")
    positive_count = int(np.count_nonzero(labels == 1))
    if not positive_count:
        raise ValueError("Validation data must contain positive AML labels.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_true_positives = np.cumsum(sorted_labels == 1)
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    true_positives = cumulative_true_positives[group_ends]
    false_positives = group_ends + 1 - true_positives
    false_negatives = positive_count - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    f1_scores = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=denominator != 0,
    )
    best_index = int(np.argmax(f1_scores))
    return float(sorted_scores[group_ends[best_index]])


def _classification_metrics(predictions: DataFrame) -> dict[str, float | int]:
    """Calculate confusion counts and threshold-based scores in one Spark action."""
    row = predictions.agg(
        F.sum(
            F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)
        ).alias("fn"),
    ).first()

    tp, tn, fp, fn = (int(row[key] or 0) for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def _print_model_table(results: list[dict[str, Any]]) -> None:
    headers = [header for header, _ in METRIC_COLUMNS]
    rows = [
        [
            result[key]
            if key == "model" or key in ("tp", "tn", "fp", "fn")
            else f"{result[key]:.4f}"
            for _, key in METRIC_COLUMNS
        ]
        for result in results
    ]
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("\nMODEL TEST SCORES")
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def train_and_evaluate_models(
    transactions: DataFrame,
    output_dir: str,
    source_stats: dict[str, int] | None = None,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    """Fit on older rows and score each model at its own validation F1 threshold."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labelled = (
        transactions.filter(F.col("is_laundering").isin(0, 1))
        .withColumn("label", F.col("is_laundering").cast("double"))
    )
    train, validation, test = _chronological_splits(labelled)
    split_data = {"train": train, "validation": validation, "test": test}
    split_stats: dict[str, dict[str, int]] = {}
    split_counts: dict[str, dict[float, int]] = {}
    for name, data in split_data.items():
        counts = _class_counts(data)
        _validate_split(name.title(), counts)
        split_counts[name] = counts
        split_stats[name] = _split_stats(counts)

    sample_counts = _class_counts(labelled)
    actual_sample_size = sum(sample_counts.values())
    print_data_usage_report(
        source_stats,
        sample_size if sample_size is not None else actual_sample_size,
        sample_seed,
        sample_counts,
        split_stats,
    )
    print(
        "Split policy: chronological by timestamp (60% train, 20% validation, "
        "20% test quantiles). Positive-rate differences are measured temporal "
        "distribution shift, not stratified away; timestamps and labels are "
        "excluded from model features."
    )
    print(
        "Leakage controls: account IDs and transaction IDs are excluded; the "
        "payment-type index is fit on training rows only; class weights use "
        "training labels only."
    )

    train_counts = split_counts["train"]
    positive_weight = train_counts[0.0] / train_counts[1.0]
    weighted_train = train.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, F.lit(positive_weight)).otherwise(F.lit(1.0)),
    )

    payment_indexer = StringIndexer(
        inputCol="payment_type",
        outputCol="payment_type_index",
        handleInvalid="keep",
    ).fit(weighted_train)
    assembler = VectorAssembler(
        inputCols=MODEL_FEATURES,
        outputCol="features",
        handleInvalid="error",
    )

    def make_features(data: DataFrame, include_weight: bool) -> DataFrame:
        indexed = payment_indexer.transform(data)
        assembled = assembler.transform(indexed)
        columns = ["label", "features"]
        if include_weight:
            columns.append("class_weight")
        return assembled.select(*columns)

    train_features = make_features(weighted_train, include_weight=True)
    validation_features = make_features(validation, include_weight=False)
    test_features = make_features(test, include_weight=False)

    classifiers = [
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="class_weight",
                maxDepth=4,
                seed=42,
            ),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="class_weight",
                numTrees=8,
                maxDepth=4,
                featureSubsetStrategy="sqrt",
                seed=42,
            ),
        ),
        (
            "GBTClassifier",
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="class_weight",
                maxIter=5,
                maxDepth=3,
                stepSize=0.1,
                seed=42,
            ),
        ),
    ]
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR"
    )
    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
    )

    results: list[dict[str, Any]] = []
    for name, classifier in classifiers:
        print(f"Training {name}...")
        started = perf_counter()
        model = classifier.fit(train_features)
        training_seconds = perf_counter() - started

        validation_predictions = model.transform(validation_features).withColumn(
            "positive_probability",
            vector_to_array(F.col("probability")).getItem(1),
        ).persist(StorageLevel.DISK_ONLY)
        try:
            validation_labels, validation_scores = _collect_scores(
                validation_predictions
            )
            threshold = _best_validation_threshold(
                validation_labels, validation_scores
            )
        finally:
            validation_predictions.unpersist()

        test_predictions = (
            model.transform(test_features)
            .withColumn(
                "positive_probability",
                vector_to_array(F.col("probability")).getItem(1),
            )
            .withColumn(
                "prediction",
                (F.col("positive_probability") >= F.lit(threshold)).cast("double"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        try:
            metrics = _classification_metrics(test_predictions)
            metrics["pr_auc"] = pr_evaluator.evaluate(test_predictions)
            metrics["roc_auc"] = roc_evaluator.evaluate(test_predictions)
            test_labels, test_scores = _collect_scores(test_predictions)
        finally:
            test_predictions.unpersist()

        result: dict[str, Any] = {
            "model": name,
            **metrics,
            "threshold": threshold,
            "training_seconds": training_seconds,
            "rows_evaluated": split_stats["test"]["rows"],
            "_test_labels": test_labels.tolist(),
            "_test_probabilities": test_scores.tolist(),
            "_train_rows": split_stats["train"]["rows"],
            "_validation_rows": split_stats["validation"]["rows"],
            "_test_rows": split_stats["test"]["rows"],
            "_split_stats": split_stats,
        }
        if name == "RandomForestClassifier":
            result["_feature_importances"] = {
                feature: float(importance)
                for feature, importance in zip(
                    MODEL_FEATURES, model.featureImportances.toArray()
                )
            }
        results.append(result)
        print(
            f"  Validation-selected threshold={threshold:.6f}; "
            f"test precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}."
        )
        print(
            f"  Test confusion counts: TP={metrics['tp']}, TN={metrics['tn']}, "
            f"FP={metrics['fp']}, FN={metrics['fn']}."
        )

    _print_model_table(results)
    metrics_file = output_path / "model_metrics.csv"
    with metrics_file.open("w", encoding="utf-8", newline="") as output:
        headers = [header for header, _ in METRIC_COLUMNS]
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(
            {
                header: result[key]
                for header, key in METRIC_COLUMNS
            }
            for result in results
        )
    print(f"Saved metrics CSV: {metrics_file}")
    print(
        "Each model's confusion counts and threshold-based metrics use its own "
        "test predictions at its own validation-selected threshold. Equal "
        "counts can still occur naturally; PR-AUC and ROC-AUC use each model's "
        "own continuous probability scores."
    )
    return results
