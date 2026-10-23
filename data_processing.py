"""Read, sample, clean, and prepare the IBM AMLSim HI-Small transactions."""

import csv
import os
import tempfile
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


def _column_indexes(csv_path: str) -> tuple[dict[str, int], list[str]]:
    """Find the IBM columns by header name, including its duplicate Account headers."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
    if not header:
        raise ValueError(f"CSV is empty or has no header: {csv_path}")

    names = [name.strip().casefold() for name in header]
    accounts = [index for index, name in enumerate(names) if name == "account"]
    required = {
        "timestamp": "timestamp",
        "from_bank": "from bank",
        "to_bank": "to bank",
        "amount": "amount paid",
        "payment_currency": "payment currency",
        "payment_type": "payment format",
        "label": "is laundering",
    }
    missing = [name for name in required.values() if name not in names]
    if len(accounts) < 2:
        missing.append("two Account columns")
    if missing:
        raise ValueError("IBM CSV is missing required columns: " + ", ".join(missing))

    indexes = {key: names.index(name) for key, name in required.items()}
    indexes["sender_account"] = accounts[0]
    indexes["receiver_account"] = accounts[1]
    indexes["column_count"] = len(header)
    return indexes, header


def _read_raw(
    spark: SparkSession, csv_path: str, indexes: dict[str, int]
) -> tuple[DataFrame, list[str]]:
    columns = [f"source_{index}" for index in range(indexes["column_count"])]
    schema = StructType([StructField(name, StringType(), True) for name in columns])
    rows = spark.read.option("header", "false").schema(schema).csv(csv_path)
    timestamp = F.trim(F.col(columns[indexes["timestamp"]]))
    rows = rows.filter(
        F.lower(F.regexp_replace(timestamp, "^\ufeff", "")) != "timestamp"
    )
    return rows, columns


def create_stratified_sample(
    spark: SparkSession,
    source_path: str,
    sample_path: str,
    sample_size: int = 500_000,
    seed: int = 42,
    source_stats: dict[str, int] | None = None,
) -> str:
    """Save an exact-size random sample with the source AML class ratio."""
    source_file = Path(source_path).resolve()
    output_file = Path(sample_path).resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"IBM transaction CSV not found: {source_file}")
    if source_file == output_file:
        raise ValueError("The sample path must not overwrite the original dataset.")
    if sample_size < 2:
        raise ValueError("Sample size must be at least two rows.")

    indexes, header = _column_indexes(str(source_file))
    raw, columns = _read_raw(spark, str(source_file), indexes)
    label_column = columns[indexes["label"]]
    labelled = raw.withColumn(
        "_label", F.trim(F.col(label_column)).cast("int")
    )
    source_counts = labelled.agg(
        F.count("*").alias("source_rows"),
        F.sum(F.when(F.col("_label") == 1, 1).otherwise(0)).alias("positive"),
        F.sum(F.when(F.col("_label") == 0, 1).otherwise(0)).alias("negative"),
    ).first()
    counts = {
        1: int(source_counts["positive"] or 0),
        0: int(source_counts["negative"] or 0),
    }
    total = sum(counts.values())
    positive = counts.get(1, 0)
    negative = counts.get(0, 0)
    if source_stats is not None:
        source_stats.update(
            {
                "source_rows": int(source_counts["source_rows"] or 0),
                "positive": positive,
                "negative": negative,
            }
        )
    if not positive or not negative:
        raise ValueError("The IBM dataset must contain both AML labels.")
    if total < sample_size:
        raise ValueError(
            f"Requested {sample_size:,} rows, but the dataset has only {total:,} "
            "validly labeled rows."
        )

    positive_quota = round(sample_size * positive / total)
    positive_quota = min(max(positive_quota, 1), sample_size - 1)
    negative_quota = sample_size - positive_quota
    sampled = None
    for label, quota in ((0, negative_quota), (1, positive_quota)):
        group = (
            labelled.filter(F.col("_label") == label)
            .orderBy(F.rand(seed))
            .limit(quota)
            .select(*columns)
        )
        sampled = group if sampled is None else sampled.unionByName(group)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".aml-sample-", dir=str(output_file.parent)
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_file = temporary_root / output_file.name
        sampled_counts = {0: 0, 1: 0}
        with temporary_file.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            row_count = 0
            for row in sampled.coalesce(1).toLocalIterator():
                writer.writerow(row)
                sampled_counts[int(row[indexes["label"]])] += 1
                row_count += 1
        if row_count != sample_size or sampled_counts != {
            0: negative_quota,
            1: positive_quota,
        }:
            raise RuntimeError(
                f"Sample verification failed: got {row_count:,} rows and "
                f"label counts {sampled_counts}."
            )
        os.replace(temporary_file, output_file)

    print(f"Saved reproducible sample: {output_file}")
    return str(output_file)


def load_and_clean(spark: SparkSession, csv_path: str) -> DataFrame:
    """Parse IBM fields, build sender/receiver account IDs, and remove invalid rows."""
    indexes, _ = _column_indexes(csv_path)
    raw, columns = _read_raw(spark, csv_path, indexes)

    def source_column(name: str) -> Column:
        return F.trim(F.col(columns[indexes[name]]))

    timestamp_text = source_column("timestamp")
    timestamp = F.coalesce(
        F.to_timestamp(timestamp_text, "yyyy/MM/dd HH:mm"),
        F.to_timestamp(timestamp_text, "yyyy/MM/dd HH:mm:ss"),
        F.to_timestamp(timestamp_text, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(timestamp_text),
    )
    transaction_id = F.sha2(
        F.to_json(
            F.struct(
                *[
                    F.col(column).alias(f"field_{index}")
                    for index, column in enumerate(columns)
                ]
            )
        ),
        256,
    )
    transactions = raw.select(
        transaction_id.alias("transaction_id"),
        timestamp.alias("timestamp"),
        F.concat(
            source_column("from_bank"),
            F.lit(":"),
            source_column("sender_account"),
        ).alias("sender_account"),
        F.concat(
            source_column("to_bank"),
            F.lit(":"),
            source_column("receiver_account"),
        ).alias("receiver_account"),
        source_column("amount").cast("double").alias("amount"),
        source_column("payment_type").alias("payment_type"),
        source_column("label").cast("int").alias("is_laundering"),
    )
    valid = (
        F.col("timestamp").isNotNull()
        & F.col("sender_account").isNotNull()
        & F.col("receiver_account").isNotNull()
        & F.col("amount").isNotNull()
        & (F.col("amount") >= 0)
        & F.col("payment_type").isNotNull()
        & (F.trim(F.col("payment_type")) != "")
        & F.col("is_laundering").isin(0, 1)
    )
    return transactions.filter(valid)


def add_features(transactions: DataFrame) -> DataFrame:
    """Add a simple amount-based indicator; no label-derived feature is created."""
    return transactions.withColumn(
        "is_near_threshold",
        (
            (F.col("amount") >= 9_000)
            & (F.col("amount") < 10_000)
        ).cast("int"),
    )
