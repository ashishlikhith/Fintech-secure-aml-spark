import csv
import json
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    MinMaxScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.feature_engineering import CLASSIFICATION_FEATURES


class ModelEvaluationResult(TypedDict):
    comparison: DataFrame | None
    models: dict[str, object]
    predictions: dict[str, DataFrame]
    feature_names: list[str]


def _chronological_splits(
    data: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Use disjoint earliest 60%, middle 20%, and latest 20% time periods."""
    cutoffs = data.select(
        F.unix_timestamp("timestamp").alias("split_epoch")
    ).approxQuantile("split_epoch", [0.6, 0.8], 0.001)
    if len(cutoffs) != 2:
        empty = data.limit(0)
        return empty, empty, empty
    first_cutoff, second_cutoff = cutoffs
    event_epoch = F.unix_timestamp("timestamp")
    return (
        data.filter(event_epoch < first_cutoff),
        data.filter((event_epoch >= first_cutoff) & (event_epoch < second_cutoff)),
        data.filter(event_epoch >= second_cutoff),
    )


def _class_counts(data: DataFrame) -> dict[float, int]:
    return {
        float(row["label"]): int(row["count"])
        for row in data.groupBy("label").count().collect()
    }


def _classification_metrics(
    predictions: DataFrame, prediction_col: str = "prediction"
) -> dict[str, float | int]:
    counts = predictions.agg(
        F.sum(
            F.when(
                (F.col("label") == 1) & (F.col(prediction_col) == 1), 1
            ).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when(
                (F.col("label") == 0) & (F.col(prediction_col) == 0), 1
            ).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when(
                (F.col("label") == 0) & (F.col(prediction_col) == 1), 1
            ).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when(
                (F.col("label") == 1) & (F.col(prediction_col) == 0), 1
            ).otherwise(0)
        ).alias("fn"),
    ).first()
    true_positive = int(counts["tp"] or 0)
    true_negative = int(counts["tn"] or 0)
    false_positive = int(counts["fp"] or 0)
    false_negative = int(counts["fn"] or 0)
    total = true_positive + true_negative + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = (
        true_positive / recall_denominator if recall_denominator else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    positive_rate = (
        (true_positive + false_negative) / total if total else 0.0
    )
    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "positive_rate": positive_rate,
    }


def _class_weighted(
    data: DataFrame, counts: dict[float, int]
) -> DataFrame:
    positive = counts.get(1.0, 0)
    negative = counts.get(0.0, 0)
    if not positive or not negative:
        return data.withColumn("class_weight", F.lit(1.0))
    positive_weight = negative / positive
    return data.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(positive_weight)).otherwise(F.lit(1.0)),
    )


def _select_validation_threshold(validation_predictions: DataFrame) -> float:
    scores = vector_to_array("probability")[1]
    bucketed = (
        validation_predictions.select(
            "label",
            F.floor(F.least(F.greatest(scores, F.lit(0.0)), F.lit(1.0)) * 1000)
            .cast("int")
            .alias("score_bucket"),
        )
        .groupBy("score_bucket")
        .agg(
            F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("positive"),
            F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).alias("negative"),
        )
        .orderBy(F.desc("score_bucket"))
        .collect()
    )
    total_positive = sum(int(row["positive"]) for row in bucketed)
    if total_positive == 0:
        raise ValueError("Validation period contains no positive labels.")

    true_positive = false_positive = 0
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for row in bucketed:
        true_positive += int(row["positive"])
        false_positive += int(row["negative"])
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / total_positive
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        threshold = int(row["score_bucket"]) / 1000
        candidate_key = (f1, precision, threshold)
        if candidate_key > best_key:
            best_key = candidate_key
            best_threshold = threshold
    return best_threshold


def _classifier_pipeline() -> list[tuple[str, object]]:
    return [
        (
            "Decision Tree",
            DecisionTreeClassifier(
                labelCol="label",
                featuresCol="features",
                maxDepth=5,
                weightCol="class_weight",
                seed=42,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="class_weight",
                numTrees=20,
                maxDepth=5,
                seed=42,
            ),
        ),
        (
            "Gradient-Boosted Trees",
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="class_weight",
                maxIter=8,
                maxDepth=3,
                seed=42,
            ),
        ),
    ]


def _write_supervised_artifacts(
    output_dir: Path,
    rows: list[dict[str, object]],
    split_counts: dict[str, dict[float, int]],
) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0])
        with (metrics_dir / "model_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "positive_label": 1,
        "threshold_selection": "maximum validation F1 over probability bins of width 0.001",
        "split_class_counts": {
            name: {str(label): count for label, count in counts.items()}
            for name, counts in split_counts.items()
        },
        "models": rows,
    }
    (metrics_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def train_and_evaluate_models(
    ml_df: DataFrame,
    output_dir: str = "artifacts",
) -> ModelEvaluationResult:
    """Train and evaluate exactly three classifiers on chronological splits."""
    started = perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labelled = ml_df.filter(F.col("label").isin(0.0, 1.0))
    train, validation, test = _chronological_splits(labelled)
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    counts: dict[str, dict[float, int]] = {}
    split_counts: dict[str, dict[float, int]] = {}
    trained_models: dict[str, object] = {}
    model_predictions: dict[str, DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    try:
        for name, frame in splits.items():
            counts[name] = _class_counts(frame)
            split_total = sum(counts[name].values())
            positive = counts[name].get(1.0, 0)
            split_counts[name] = counts[name]
            print(
                f"{name.title()} class distribution: "
                f"rows={split_total}, positives={positive}, "
                f"positive_prevalence={positive / split_total if split_total else 0:.6f} "
                f"(positive count / all labeled rows)."
            )
        if any(
            not counts[name].get(0.0) or not counts[name].get(1.0)
            for name in ("train", "validation", "test")
        ):
            raise ValueError(
                "Each chronological train, validation, and test split must contain "
                "both labels (0 and 1); see the split counts above."
            )
        train_data = _class_weighted(splits["train"], counts["train"])
        payment_type_indexer = StringIndexer(
            inputCol="payment_type",
            outputCol="payment_type_index",
            handleInvalid="keep",
        )
        payment_currency_indexer = StringIndexer(
            inputCol="payment_currency",
            outputCol="payment_currency_index",
            handleInvalid="keep",
        )
        payment_type_encoder = OneHotEncoder(
            inputCols=["payment_type_index", "payment_currency_index"],
            outputCols=["payment_type_vector", "payment_currency_vector"],
            handleInvalid="keep",
        )
        assembler = VectorAssembler(
            inputCols=CLASSIFICATION_FEATURES
            + ["payment_type_vector", "payment_currency_vector"],
            outputCol="raw_features",
            handleInvalid="error",
        )
        preprocessing = Pipeline(
            stages=[
                payment_type_indexer,
                payment_currency_indexer,
                payment_type_encoder,
                assembler,
                MinMaxScaler(
                    inputCol="raw_features",
                    outputCol="features",
                    min=0.0,
                    max=1.0,
                ),
            ]
        )
        preprocessing_model = preprocessing.fit(train_data)
        train_features = preprocessing_model.transform(train_data).persist(
            StorageLevel.DISK_ONLY
        )
        validation_features = preprocessing_model.transform(
            splits["validation"]
        ).persist(StorageLevel.DISK_ONLY)
        test_features = preprocessing_model.transform(splits["test"]).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            train_features.count()
            validation_features.count()
            test_features.count()
            feature_metadata = train_features.schema["raw_features"].metadata[
                "ml_attr"
            ]
            feature_names = [""] * int(feature_metadata["num_attrs"])
            for attributes in feature_metadata.get("attrs", {}).values():
                for attribute in attributes:
                    feature_names[int(attribute["idx"])] = attribute["name"]
            if not feature_names or any(not name for name in feature_names):
                raise ValueError("Spark did not provide complete feature-name metadata.")

            for name, estimator in _classifier_pipeline():
                model_started = perf_counter()
                model = estimator.fit(train_features)
                validation_output = model.transform(validation_features)
                threshold = _select_validation_threshold(validation_output)
                predictions = (
                    model.transform(test_features)
                    .withColumn(
                        "default_prediction", F.col("prediction")
                    )
                    .withColumn(
                        "prediction",
                        F.when(
                            vector_to_array("probability")[1] >= threshold,
                            F.lit(1.0),
                        ).otherwise(F.lit(0.0)),
                    )
                    .select(
                        "transaction_id",
                        "label",
                        "prediction",
                        "default_prediction",
                        "probability",
                    )
                    .persist(StorageLevel.DISK_ONLY)
                )
                model_predictions[name] = predictions
                predictions.count()
                default_metrics = _classification_metrics(
                    predictions, prediction_col="default_prediction"
                )
                metrics = _classification_metrics(predictions)
                metrics["default_precision"] = float(default_metrics["precision"])
                metrics["default_recall"] = float(default_metrics["recall"])
                metrics["default_f1"] = float(default_metrics["f1"])
                metrics["default_false_positives"] = int(
                    default_metrics["false_positives"]
                )
                metrics["default_false_negatives"] = int(
                    default_metrics["false_negatives"]
                )
                metrics["threshold"] = threshold
                metrics["area_under_pr"] = BinaryClassificationEvaluator(
                    labelCol="label",
                    rawPredictionCol="probability",
                    metricName="areaUnderPR",
                ).evaluate(predictions)
                metrics["area_under_roc"] = BinaryClassificationEvaluator(
                    labelCol="label",
                    rawPredictionCol="probability",
                    metricName="areaUnderROC",
                ).evaluate(predictions)
                elapsed = perf_counter() - model_started
                print(
                    f"\n{name} (threshold selected on validation={threshold:.3f}): "
                    f"precision={metrics['precision']:.4f}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"false_positives={metrics['false_positives']}, "
                    f"false_negatives={metrics['false_negatives']}, "
                    f"PR-AUC={metrics['area_under_pr']:.4f}, "
                    f"ROC-AUC={metrics['area_under_roc']:.4f}, "
                    f"default-threshold precision="
                    f"{metrics['default_precision']:.4f}, "
                    f"time={elapsed:.2f}s"
                )
                metric_rows.append(
                    {
                        "model": name,
                        "task": "binary classification",
                        **metrics,
                        "training_seconds": elapsed,
                    }
                )
                trained_models[name] = model
        finally:
            train_features.unpersist()
            validation_features.unpersist()
            test_features.unpersist()
    except Exception:
        for predictions in model_predictions.values():
            predictions.unpersist()
        raise

    if metric_rows:
        schema = (
            "model string, task string, accuracy double, precision double, recall double, "
            "f1 double, false_positives long, false_negatives long, "
            "default_precision double, default_recall double, default_f1 double, "
            "default_false_positives long, default_false_negatives long, "
            "threshold double, area_under_pr double, area_under_roc double, "
            "positive_rate double, training_seconds double"
        )
        comparison = ml_df.sparkSession.createDataFrame(
            [
                tuple(
                    row.get(column)
                    for column in (
                        "model",
                        "task",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "false_positives",
                        "false_negatives",
                        "default_precision",
                        "default_recall",
                        "default_f1",
                        "default_false_positives",
                        "default_false_negatives",
                        "threshold",
                        "area_under_pr",
                        "area_under_roc",
                        "positive_rate",
                        "training_seconds",
                    )
                )
                for row in metric_rows
            ],
            schema=schema,
        )
        print("\nHeld-out classification model comparison:")
        comparison.orderBy("model").show(20, truncate=False)
    else:
        raise RuntimeError("No supervised classifiers completed.")

    if len(metric_rows) != 3:
        raise RuntimeError(
            f"Expected three completed classifiers, got {len(metric_rows)}."
        )
    _write_supervised_artifacts(output_path, metric_rows, split_counts)

    print(f"Model evaluation completed in {perf_counter() - started:.2f}s.")
    return {
        "comparison": comparison,
        "models": trained_models,
        "predictions": model_predictions,
        "feature_names": feature_names,
    }
