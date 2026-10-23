"""
ml_models.py
------------
Trains and evaluates exactly three classifiers for AML detection:
  1. DecisionTreeClassifier
  2. RandomForestClassifier
  3. GBTClassifier

Uses chronological train/validation/test splits (60/20/20 by time).
Reports precision, recall, F1, PR-AUC, ROC-AUC, accuracy, and confusion matrix.

Viva summary:
  - prepare_features()          → assembles numeric features into a vector
  - train_and_evaluate_models() → trains 3 models, returns metrics & predictions
"""

import json
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# These are the numeric columns we use as model inputs.
# IMPORTANT: is_laundering (the label) is NOT included → no target leakage.
# risk_score is excluded too because it is derived from the same features.
FEATURE_COLUMNS = [
    "amount",                    # transaction amount
    "velocity_24h",              # number of sender's txns in last 24 h
    "volume_24h",                # total $ sender sent in last 24 h
    "near_threshold_count_24h",  # near-threshold txns in last 24 h
    "prior_transaction_count",   # sender's lifetime transaction count
    "prior_average_amount",      # sender's lifetime average amount
]


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------

def _chronological_split(df: DataFrame):
    """
    Split by time: earliest 60% → train, next 20% → validation, last 20% → test.
    This prevents future data leaking into training (temporal leakage).
    """
    epoch = F.unix_timestamp("timestamp")
    cutoffs = df.select(epoch.alias("ep")).approxQuantile("ep", [0.6, 0.8], 0.001)

    if len(cutoffs) != 2:
        raise ValueError("Could not compute time-based split cutoffs.")

    t1, t2 = cutoffs
    train = df.filter(epoch < t1)
    val   = df.filter((epoch >= t1) & (epoch < t2))
    test  = df.filter(epoch >= t2)
    return train, val, test


def _print_split_info(name: str, split: DataFrame):
    """Print class distribution for a split (for the viva: proves balanced splits)."""
    counts = {
        float(r["label"]): int(r["count"])
        for r in split.groupBy("label").count().collect()
    }
    total = sum(counts.values())
    pos = counts.get(1.0, 0)
    rate = pos / total if total else 0
    print(f"  {name:12s}: {total:>8,} rows | positive={pos:>6,} ({rate:.4%})")
    return counts


# ---------------------------------------------------------------------------
# Confusion-matrix based metrics
# ---------------------------------------------------------------------------

def _compute_metrics(predictions: DataFrame) -> dict:
    """
    Compute TP, TN, FP, FN from label vs prediction columns,
    then derive precision, recall, F1, and accuracy.
    """
    row = predictions.agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).first()

    tp, tn, fp, fn = int(row["tp"]), int(row["tn"]), int(row["fp"]), int(row["fn"])
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / total if total else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Feature preparation pipeline
# ---------------------------------------------------------------------------

def _build_pipeline():
    """
    Build a simple ML Pipeline:
      1. StringIndexer  → encode payment_type as a number
      2. VectorAssembler → combine all features into one vector column
    No MinMaxScaler to keep it simple and explainable.
    """
    # Encode the categorical 'payment_type' as a numeric index
    indexer = StringIndexer(
        inputCol="payment_type",
        outputCol="payment_type_index",
        handleInvalid="keep",
    )

    # Combine all features into a single vector
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS + ["payment_type_index"],
        outputCol="features",
        handleInvalid="skip",
    )

    return Pipeline(stages=[indexer, assembler])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate_models(
    df: DataFrame,
    output_dir: str = "output",
) -> dict:
    """
    Train Decision Tree, Random Forest, and GBT on the enriched data.

    Steps:
      1. Create label column from is_laundering
      2. Split data chronologically (60/20/20)
      3. Fit preprocessing pipeline on train data
      4. Train each classifier, evaluate on test data
      5. Save metrics JSON

    Returns a dict with keys:
      "metrics"     → list of per-model metric dicts
      "predictions" → dict of {model_name: predictions_DataFrame}
    """
    started = perf_counter()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Prepare the label ----
    labelled = (
        df.filter(F.col("is_laundering").isin(0, 1))
        .withColumn("label", F.col("is_laundering").cast("double"))
        .fillna(0.0, subset=FEATURE_COLUMNS)
    )

    # ---- Chronological split ----
    train_raw, val_raw, test_raw = _chronological_split(labelled)

    print("\nData split class distributions:")
    train_counts = _print_split_info("Train", train_raw)
    val_counts   = _print_split_info("Validation", val_raw)
    test_counts  = _print_split_info("Test", test_raw)

    # Check both classes exist in every split
    for name, counts in [("train", train_counts), ("validation", val_counts), ("test", test_counts)]:
        if not counts.get(0.0) or not counts.get(1.0):
            raise ValueError(f"{name} split is missing one class. Cannot train.")

    # ---- Add class weights to handle imbalance ----
    # Give higher weight to rare positive class so the model doesn't ignore it
    pos_count = train_counts.get(1.0, 0)
    neg_count = train_counts.get(0.0, 0)
    weight_ratio = neg_count / pos_count if pos_count else 1.0
    train_weighted = train_raw.withColumn(
        "weight",
        F.when(F.col("label") == 1.0, F.lit(weight_ratio)).otherwise(F.lit(1.0)),
    )

    # ---- Fit preprocessing on training data only ----
    pipeline = _build_pipeline()
    pipeline_model = pipeline.fit(train_weighted)

    train_feat = pipeline_model.transform(train_weighted).select("label", "weight", "features")
    train_feat = train_feat.persist(StorageLevel.DISK_ONLY)
    train_feat.count()  # materialize

    test_feat = pipeline_model.transform(test_raw).select("transaction_id", "label", "features")
    test_feat = test_feat.persist(StorageLevel.DISK_ONLY)
    test_feat.count()

    # ---- Define the three classifiers ----
    # Conservative parameters suitable for 16 GB RAM laptop
    classifiers = [
        (
            "Decision Tree",
            DecisionTreeClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="weight",
                maxDepth=5,             # shallow tree → easy to explain
                seed=42,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="weight",
                numTrees=10,            # 10 trees (low memory)
                maxDepth=5,
                featureSubsetStrategy="sqrt",
                seed=42,
            ),
        ),
        (
            "Gradient-Boosted Trees",
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                weightCol="weight",
                maxIter=10,             # 10 boosting rounds
                maxDepth=4,
                seed=42,
            ),
        ),
    ]

    # ---- Train and evaluate each model ----
    all_metrics = []
    all_predictions = {}

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR"
    )
    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
    )

    try:
        for name, estimator in classifiers:
            t0 = perf_counter()
            print(f"\nTraining {name}…")

            model = estimator.fit(train_feat)
            preds = model.transform(test_feat).select(
                "transaction_id", "label", "prediction", "probability"
            )
            preds = preds.persist(StorageLevel.DISK_ONLY)
            preds.count()  # materialize

            # Compute metrics
            m = _compute_metrics(preds)
            m["pr_auc"]  = pr_evaluator.evaluate(preds)
            m["roc_auc"] = roc_evaluator.evaluate(preds)
            m["model"]   = name
            m["time_s"]  = round(perf_counter() - t0, 2)

            all_metrics.append(m)
            all_predictions[name] = preds

            # Print results
            print(
                f"  {name}: precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
                f"accuracy={m['accuracy']:.4f}  time={m['time_s']}s"
            )
            print(
                f"  Confusion: TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}"
            )
    finally:
        train_feat.unpersist()
        test_feat.unpersist()

    # ---- Explain why accuracy is misleading ----
    print(
        "\n⚠ Note on accuracy: With ~99.8% legitimate transactions, a model that "
        "predicts 'not laundering' for everything gets ~99.8% accuracy but catches "
        "zero fraud. That's why we focus on precision, recall, F1, and PR-AUC."
    )

    # ---- Save metrics to JSON ----
    metrics_file = out_path / "model_metrics.json"
    metrics_file.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved: {metrics_file}")

    elapsed = perf_counter() - started
    print(f"All models trained and evaluated in {elapsed:.1f}s.")

    return {
        "metrics": all_metrics,
        "predictions": all_predictions,
    }
