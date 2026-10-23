from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
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
    finite_features = F.lit(True)
    for column in CLASSIFICATION_FEATURES:
        value = F.col(column)
        finite_features = finite_features & value.isNotNull() & ~F.isnan(value) & (
            F.abs(value) < F.lit(float("inf"))
        )
    profile_expressions = [
        F.count("*").alias("row_count"),
        F.sum(F.when(~finite_features, 1).otherwise(0)).alias("invalid_rows"),
    ]
    for column in CLASSIFICATION_FEATURES:
        value = F.col(column)
        valid = finite_features
        profile_expressions.extend(
            [
                F.sum(F.when(valid, 1).otherwise(0)).alias(f"{column}_valid"),
                F.min(F.when(valid, value)).alias(f"{column}_min"),
                F.max(F.when(valid, value)).alias(f"{column}_max"),
            ]
        )
    profile = ml_ready_df.agg(*profile_expressions).first()
    if not profile["row_count"]:
        raise ValueError("Feature engineering received no transaction records.")

    invalid_feature_rows = int(profile["invalid_rows"] or 0)
    if invalid_feature_rows:
        print(
            f"Excluded {invalid_feature_rows} row(s) containing NaN or infinite "
            "model features."
        )
    ml_ready_df = ml_ready_df.filter(finite_features)
    if invalid_feature_rows == profile["row_count"]:
        raise ValueError("No rows have finite numerical model features.")
    ml_ready_df.createOrReplaceTempView("ml_features")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    finite_columns: list[str] = []
    excluded_columns = []
    for column in CLASSIFICATION_FEATURES:
        valid_count = int(profile[f"{column}_valid"] or 0)
        minimum = profile[f"{column}_min"]
        maximum = profile[f"{column}_max"]
        if valid_count < 2 or minimum is None or maximum is None or minimum == maximum:
            excluded_columns.append(column)
        else:
            finite_columns.append(column)
    if excluded_columns:
        print(
            "Excluded constant or insufficient numerical columns from correlation "
            "only: " + ", ".join(excluded_columns)
        )

    if finite_columns:
        assembler = VectorAssembler(
            inputCols=finite_columns,
            outputCol="correlation_features",
            handleInvalid="error",
        )
        correlation_input = assembler.transform(
            ml_ready_df.select(*finite_columns)
        )
        correlation_matrix = Correlation.corr(
            correlation_input, "correlation_features", "pearson"
        ).first()[0].toArray()
        image = axis.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(finite_columns)))
        axis.set_yticks(range(len(finite_columns)))
        axis.set_xticklabels(finite_columns, rotation=45, ha="right")
        axis.set_yticklabels(finite_columns)
        axis.set_title("Transaction feature correlations (Pearson)")
        figure.colorbar(image, ax=axis, label="Correlation")
    else:
        axis.text(
            0.5,
            0.5,
            "No varying finite numerical features available",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        print("Warning: no valid varying columns were available for correlation.")
    figure.tight_layout()
    figure.savefig(output_path / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    print(
        "Correlation matrix is computed in Spark; only its "
        f"{len(finite_columns)}x{len(finite_columns)} result "
        f"is plotted. Heatmap: {output_path / 'correlation_heatmap.png'}"
    )
    print(
        "Correlation indicates linear association, not causation; correlated "
        "features may be redundant."
    )
    print(f"Feature preparation completed in {perf_counter() - started:.2f}s.")
    return ml_ready_df
