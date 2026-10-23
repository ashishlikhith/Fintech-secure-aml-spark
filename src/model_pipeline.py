from time import perf_counter
from typing import TypedDict

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    MinMaxScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.feature_engineering import CLASSIFICATION_FEATURES


class ModelEvaluationResult(TypedDict):
    comparison: DataFrame | None
    models: dict[str, object]
    predictions: dict[str, DataFrame]


def _chronological_train_test_split(
    data: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Hold out the latest 20% of timestamps to test future-period generalization."""
    cutoff_values = data.select(
        F.unix_timestamp("timestamp").alias("split_epoch")
    ).approxQuantile("split_epoch", [0.8], 0.001)
    if not cutoff_values:
        return data.limit(0), data.limit(0)
    cutoff = cutoff_values[0]
    event_epoch = F.unix_timestamp("timestamp")
    return data.filter(event_epoch < cutoff), data.filter(
        event_epoch >= cutoff
    )


def _has_both_classes(data: DataFrame) -> bool:
    return data.select("label").distinct().count() == 2


def _classification_metrics(predictions: DataFrame) -> dict[str, float]:
    counts = predictions.agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
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
    recall = true_positive / recall_denominator if recall_denominator else 0.0
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
        "positive_rate": positive_rate,
    }


def _class_weighted(data: DataFrame) -> DataFrame:
    counts = {
        row["label"]: row["count"]
        for row in data.groupBy("label").count().collect()
    }
    positive = counts.get(1.0, 0)
    negative = counts.get(0.0, 0)
    if not positive or not negative:
        return data.withColumn("class_weight", F.lit(1.0))
    positive_weight = negative / positive
    return data.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(positive_weight)).otherwise(F.lit(1.0)),
    )


def train_and_evaluate_models(
    ml_df: DataFrame,
) -> ModelEvaluationResult:
    """Compare Decision Tree and Random Forest on a future-period holdout."""
    spark = ml_df.sparkSession
    started = perf_counter()
    classification_rows: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}
    model_predictions: dict[str, DataFrame] = {}

    labelled = ml_df.filter(F.col("label").isin(0.0, 1.0))
    class_counts = labelled.groupBy("label").count()
    has_two_classes = class_counts.count() == 2
    classification_train, classification_test = _chronological_train_test_split(
        labelled
    )
    classification_train = classification_train.cache()
    classification_test = classification_test.cache()
    try:
        train_count = classification_train.count()
        test_count = classification_test.count()
        if (
            has_two_classes
            and train_count
            and test_count
            and _has_both_classes(classification_train)
            and _has_both_classes(classification_test)
        ):
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
            scaler = MinMaxScaler(
                inputCol="raw_features",
                outputCol="features",
                min=0.0,
                max=1.0,
            )
            preprocessing = Pipeline(
                stages=[
                    payment_type_indexer,
                    payment_currency_indexer,
                    payment_type_encoder,
                    assembler,
                    scaler,
                ]
            )
            preprocessing_model = preprocessing.fit(classification_train)
            train_features = _class_weighted(
                preprocessing_model.transform(classification_train)
            ).cache()
            test_features = preprocessing_model.transform(classification_test).cache()
            try:
                train_features.count()
                test_features.count()
                classifiers = [
                    (
                        "Decision Tree",
                        DecisionTreeClassifier(
                            labelCol="label",
                            featuresCol="features",
                            maxDepth=8,
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
                            numTrees=50,
                            maxDepth=8,
                            seed=42,
                        ),
                    ),
                ]
                for name, estimator in classifiers:
                    model_started = perf_counter()
                    model = estimator.fit(train_features)
                    predictions = model.transform(test_features).cache()
                    predictions.count()
                    metrics = _classification_metrics(predictions)
                    metrics["area_under_pr"] = BinaryClassificationEvaluator(
                        labelCol="label",
                        rawPredictionCol="rawPrediction",
                        metricName="areaUnderPR",
                    ).evaluate(predictions)
                    metrics["area_under_roc"] = BinaryClassificationEvaluator(
                        labelCol="label",
                        rawPredictionCol="rawPrediction",
                        metricName="areaUnderROC",
                    ).evaluate(predictions)
                    elapsed = perf_counter() - model_started
                    print(f"\n{name} confusion matrix (actual, predicted):")
                    predictions.groupBy("label", "prediction").count().orderBy(
                        "label", "prediction"
                    ).show(4, truncate=False)
                    print(
                        f"{name}: accuracy={metrics['accuracy']:.4f}, "
                        f"precision={metrics['precision']:.4f}, "
                        f"recall={metrics['recall']:.4f}, "
                        f"F1={metrics['f1']:.4f}, "
                        f"PR-AUC={metrics['area_under_pr']:.4f}, "
                        f"ROC-AUC={metrics['area_under_roc']:.4f}, "
                        f"positive-rate={metrics['positive_rate']:.4f}, "
                        f"time={elapsed:.2f}s"
                    )
                    classification_rows.append(
                        {
                            "model": name,
                            "task": "binary classification",
                            **metrics,
                            "training_seconds": elapsed,
                        }
                    )
                    trained_models[name] = model
                    model_predictions[name] = predictions

            finally:
                train_features.unpersist()
                test_features.unpersist()
        else:
            print(
                "Supervised classification skipped: at least two examples of each "
                "binary label class are required."
            )
    finally:
        classification_train.unpersist()
        classification_test.unpersist()

    clustering_train, clustering_test = _chronological_train_test_split(ml_df)
    if clustering_train.limit(1).count() and clustering_test.limit(1).count():
        clustering_train = clustering_train.cache()
        clustering_test = clustering_test.cache()
        try:
            clustering_preprocessing = Pipeline(
                stages=[
                    StringIndexer(
                        inputCol="payment_type",
                        outputCol="payment_type_index",
                        handleInvalid="keep",
                    ),
                    StringIndexer(
                        inputCol="payment_currency",
                        outputCol="payment_currency_index",
                        handleInvalid="keep",
                    ),
                    OneHotEncoder(
                        inputCols=["payment_type_index", "payment_currency_index"],
                        outputCols=["payment_type_vector", "payment_currency_vector"],
                        handleInvalid="keep",
                    ),
                    VectorAssembler(
                        inputCols=CLASSIFICATION_FEATURES
                        + ["payment_type_vector", "payment_currency_vector"],
                        outputCol="raw_features",
                        handleInvalid="error",
                    ),
                    MinMaxScaler(
                        inputCol="raw_features",
                        outputCol="features",
                        min=0.0,
                        max=1.0,
                    ),
                ]
            ).fit(clustering_train)
            clustering_train_features = clustering_preprocessing.transform(
                clustering_train
            ).cache()
            clustering_test_features = clustering_preprocessing.transform(
                clustering_test
            ).cache()
            try:
                train_clustering_count = clustering_train_features.count()
                clustering_test_features.count()
                if train_clustering_count >= 2:
                    model_started = perf_counter()
                    kmeans = KMeans(
                        k=2,
                        seed=42,
                        featuresCol="features",
                        predictionCol="cluster",
                    ).fit(clustering_train_features)
                    cluster_predictions = kmeans.transform(
                        clustering_test_features
                    ).cache()
                    cluster_predictions.count()
                    elapsed = perf_counter() - model_started
                    print("\nK-means test cluster sizes (exploratory, not labels):")
                    cluster_predictions.groupBy("cluster").count().orderBy(
                        "cluster"
                    ).show(10, truncate=False)
                    print(f"K-means completed in {elapsed:.2f}s.")
                    trained_models["K-means"] = kmeans
                    model_predictions["K-means"] = cluster_predictions
                else:
                    print("K-means skipped: at least two training rows are required.")
            finally:
                clustering_train_features.unpersist()
                clustering_test_features.unpersist()
        finally:
            clustering_train.unpersist()
            clustering_test.unpersist()
    else:
        print("K-means skipped: non-empty train and test splits are required.")

    comparison_frames = [
        spark.range(1).select(
            F.lit(row["model"]).alias("model"),
            F.lit(row["task"]).alias("task"),
            F.lit(row["accuracy"]).cast("double").alias("accuracy"),
            F.lit(row["precision"]).cast("double").alias("precision"),
            F.lit(row["recall"]).cast("double").alias("recall"),
            F.lit(row["f1"]).cast("double").alias("f1"),
            F.lit(row["area_under_pr"]).cast("double").alias("area_under_pr"),
            F.lit(row["area_under_roc"]).cast("double").alias("area_under_roc"),
            F.lit(row["positive_rate"]).cast("double").alias("positive_rate"),
            F.lit(row["training_seconds"]).cast("double").alias("training_seconds"),
        )
        for row in classification_rows
    ]
    comparison = None
    if comparison_frames:
        comparison = comparison_frames[0]
        for frame in comparison_frames[1:]:
            comparison = comparison.unionByName(frame)
    if comparison is not None:
        print("\nHeld-out classification model comparison:")
        comparison.orderBy("task", "model").show(20, truncate=False)
    else:
        print("No models were evaluated; check that the inputs contain usable records.")

    print(f"Model evaluation completed in {perf_counter() - started:.2f}s.")
    return {
        "comparison": comparison,
        "models": trained_models,
        "predictions": model_predictions,
    }
