from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


CLASSIFICATION_FEATURES = [
    "amount",
    "velocity_24h",
    "volume_24h",
    "near_threshold_count_24h",
    "shared_device_count",
    "shared_contact_count",
    "prior_transaction_count",
    "prior_average_amount",
]
def process_ml_features(
    spark: SparkSession,
    base_df: DataFrame,
    output_dir: str = "artifacts",
) -> DataFrame:
    """Select leakage-conscious numeric features and create a small correlation plot."""
    started = perf_counter()
    required_columns = {
        "transaction_id",
        "timestamp",
        "amount",
        "payment_type",
        "payment_currency",
        "is_laundering",
        "velocity_24h",
        "volume_24h",
        "near_threshold_count_24h",
        "shared_device_count",
        "shared_contact_count",
        "prior_transaction_count",
        "prior_average_amount",
        "risk_score",
        "risk_category",
        "risk_indicators",
    }
    missing_columns = sorted(required_columns.difference(base_df.columns))
    if missing_columns:
        raise ValueError(
            "Enriched data is missing required columns: " + ", ".join(missing_columns)
        )

    numeric_columns = sorted(set(CLASSIFICATION_FEATURES))
    ml_ready_df = (
        base_df.select(
            "transaction_id",
            "timestamp",
            "payment_type",
            "payment_currency",
            "is_laundering",
            "risk_score",
            "risk_category",
            "risk_indicators",
            *numeric_columns,
        )
        .withColumn("label", F.col("is_laundering").cast("double"))
        .fillna(0.0, subset=numeric_columns)
    )
    if not ml_ready_df.limit(1).count():
        raise ValueError("Feature engineering received no transaction records.")
    ml_ready_df.createOrReplaceTempView("ml_features")

    assembler = VectorAssembler(
        inputCols=CLASSIFICATION_FEATURES,
        outputCol="correlation_features",
        handleInvalid="error",
    )
    correlation_input = assembler.transform(ml_ready_df.select(*CLASSIFICATION_FEATURES))
    correlation_matrix = Correlation.corr(
        correlation_input, "correlation_features", "pearson"
    ).first()[0].toArray()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(CLASSIFICATION_FEATURES)))
    axis.set_yticks(range(len(CLASSIFICATION_FEATURES)))
    axis.set_xticklabels(CLASSIFICATION_FEATURES, rotation=45, ha="right")
    axis.set_yticklabels(CLASSIFICATION_FEATURES)
    axis.set_title("Transaction feature correlations (Pearson)")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    figure.savefig(output_path / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    print(
        "Correlation matrix is computed in Spark; only its "
        f"{len(CLASSIFICATION_FEATURES)}x{len(CLASSIFICATION_FEATURES)} result "
        f"is plotted. Heatmap: {output_path / 'correlation_heatmap.png'}"
    )
    print(
        "Correlation indicates linear association, not causation; correlated "
        "features may be redundant."
    )
    print(f"Feature preparation completed in {perf_counter() - started:.2f}s.")
    return ml_ready_df
