"""
data_processing.py
-------------------
Reads the IBM AMLSim HI-Small CSV, creates a reproducible 500k stratified
sample, cleans the data, and engineers features for the ML classifiers.

Viva summary:
  - read_and_sample()  → creates/reuses the 500k-row sample CSV
  - load_and_clean()   → parses timestamps, casts types, removes bad rows
  - engineer_features()→ adds rolling-window and risk-signal columns
"""

import csv
import os
import shutil
import tempfile
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


# ---------------------------------------------------------------------------
# 1. Stratified sampling  (preserves original class distribution)
# ---------------------------------------------------------------------------

def _detect_columns(csv_path: str) -> dict:
    """Read the CSV header once to find column positions by name."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), None)
    if not header:
        raise ValueError("CSV file is empty or has no header.")

    normalised = [col.strip().casefold() for col in header]

    # The IBM CSV has two columns both named "Account"
    account_positions = [i for i, c in enumerate(normalised) if c == "account"]
    if len(account_positions) < 2:
        raise ValueError("Expected two 'Account' columns in the IBM CSV.")

    mapping = {
        "timestamp":        normalised.index("timestamp"),
        "from_bank":        normalised.index("from bank"),
        "to_bank":          normalised.index("to bank"),
        "sender_account":   account_positions[0],
        "receiver_account": account_positions[1],
        "amount_paid":      normalised.index("amount paid"),
        "payment_currency": normalised.index("payment currency"),
        "payment_format":   normalised.index("payment format"),
        "is_laundering":    normalised.index("is laundering"),
        "column_count":     len(header),
    }
    return mapping, header


def read_and_sample(
    spark: SparkSession,
    source_path: str,
    sample_path: str,
    sample_size: int = 500_000,
    seed: int = 42,
) -> str:
    """
    Create a reproducible, stratified sample of `sample_size` rows.
    - Uses F.rand(seed) + orderBy to shuffle, NOT first-N rows.
    - Preserves the original positive/negative ratio.
    - Saves the sample as a standalone CSV so reruns reuse it.
    Returns the path to the sample CSV.
    """
    if not Path(source_path).is_file():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    mapping, header = _detect_columns(source_path)

    # Build a generic schema (source_0, source_1, …) so we can read any CSV
    col_names = [f"source_{i}" for i in range(mapping["column_count"])]
    schema = StructType([StructField(c, StringType(), True) for c in col_names])

    # Read with header=false so Spark doesn't eat the header row
    raw = (
        spark.read.option("header", "false")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(source_path)
    )

    # Filter out the header row that Spark read as data
    ts_col = col_names[mapping["timestamp"]]
    raw = raw.filter(F.lower(F.trim(F.regexp_replace(F.col(ts_col), "^\ufeff", ""))) != "timestamp")

    # Extract the label column for stratification
    label_col = col_names[mapping["is_laundering"]]
    raw = raw.withColumn("_label", F.trim(F.col(label_col)).cast("int"))
    raw = raw.filter(F.col("_label").isin(0, 1))

    # Count each class in the full dataset
    class_counts = {
        int(r["_label"]): int(r["count"])
        for r in raw.groupBy("_label").count().collect()
    }
    total = sum(class_counts.values())
    pos = class_counts.get(1, 0)
    neg = class_counts.get(0, 0)
    print(f"Full dataset: {total:,} rows  |  AML-positive={pos:,} ({pos/total:.4%})  |  AML-negative={neg:,}")

    if total < sample_size:
        raise ValueError(f"Need {sample_size:,} rows but only {total:,} available.")

    # Calculate how many of each class to keep (proportional)
    pos_quota = max(1, min(round(sample_size * pos / total), sample_size - 1))
    neg_quota = sample_size - pos_quota
    print(f"Sample quotas: positive={pos_quota:,}, negative={neg_quota:,}")

    # Build the sample CSV if it doesn't already exist
    out = Path(sample_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not out.exists():
        print("Creating new stratified sample (this may take a minute)…")
        neg_sample = (
            raw.filter(F.col("_label") == 0)
            .orderBy(F.rand(seed))    # random shuffle with fixed seed
            .limit(neg_quota)
            .select(*col_names)
        )
        pos_sample = (
            raw.filter(F.col("_label") == 1)
            .orderBy(F.rand(seed))
            .limit(pos_quota)
            .select(*col_names)
        )
        sampled = neg_sample.unionByName(pos_sample)

        # Write through a temp directory for atomicity
        with tempfile.TemporaryDirectory(prefix=".aml-sample-", dir=str(out.parent)) as tmp:
            csv_dir = Path(tmp) / "csv"
            sampled.coalesce(1).write.mode("error").option("header", "false").csv(str(csv_dir))
            part_file = next(csv_dir.glob("part-*.csv"), None)
            if part_file is None:
                raise RuntimeError("Spark failed to write the sample CSV.")
            final = Path(tmp) / out.name
            with final.open("w", encoding="utf-8", newline="") as dest:
                csv.writer(dest).writerow(header)
                with part_file.open("r", encoding="utf-8", newline="") as src:
                    shutil.copyfileobj(src, dest)
            os.replace(final, out)

    print(f"Sample CSV: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# 2. Load and clean transactions
# ---------------------------------------------------------------------------

def load_and_clean(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Read the sample CSV, parse timestamps, cast types, remove invalid rows.
    Returns a clean DataFrame with columns:
      transaction_id, timestamp, sender_account, receiver_account,
      amount, payment_currency, payment_type, is_laundering
    """
    mapping, _ = _detect_columns(csv_path)
    col_names = [f"source_{i}" for i in range(mapping["column_count"])]
    schema = StructType([StructField(c, StringType(), True) for c in col_names])

    raw = (
        spark.read.option("header", "false")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(csv_path)
    )

    # Helper to trim a source column
    def src(name):
        return F.trim(F.col(col_names[mapping[name]]))

    # Skip the header row if Spark included it
    raw = raw.filter(
        F.lower(F.regexp_replace(src("timestamp"), "^\ufeff", "")) != "timestamp"
    )

    # Parse timestamps (the IBM CSV uses yyyy/MM/dd HH:mm format)
    parsed_ts = F.coalesce(
        F.to_timestamp(src("timestamp"), "yyyy/MM/dd HH:mm"),
        F.to_timestamp(src("timestamp"), "yyyy/MM/dd HH:mm:ss"),
        F.to_timestamp(src("timestamp"), "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(src("timestamp")),
    )

    # Build a unique transaction ID from a hash of all fields
    tx_id = F.concat(
        F.lit("IBM-"),
        F.sha2(
            F.to_json(F.struct(*[F.col(c).alias(f"f{i}") for i, c in enumerate(col_names)])),
            256,
        ),
    )

    # Select and rename columns to our standard schema
    df = raw.select(
        tx_id.alias("transaction_id"),
        parsed_ts.alias("timestamp"),
        F.concat(src("from_bank"), F.lit(":"), src("sender_account")).alias("sender_account"),
        F.concat(src("to_bank"), F.lit(":"), src("receiver_account")).alias("receiver_account"),
        src("amount_paid").cast("double").alias("amount"),
        src("payment_currency").alias("payment_currency"),
        src("payment_format").alias("payment_type"),
        src("is_laundering").cast("int").alias("is_laundering"),
    )

    # Keep only valid rows
    valid = (
        F.col("transaction_id").isNotNull()
        & F.col("timestamp").isNotNull()
        & F.col("sender_account").isNotNull()
        & F.col("receiver_account").isNotNull()
        & F.col("amount").isNotNull()
        & (F.col("amount") >= 0)
        & F.col("is_laundering").isin(0, 1)
    )
    before = df.count()
    df = df.filter(valid).dropDuplicates(["transaction_id"])
    after = df.count()
    if before > after:
        print(f"Cleaning: kept {after:,} of {before:,} rows ({before - after} removed).")

    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering  (rolling windows + risk signals)
# ---------------------------------------------------------------------------

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Add account-level features computed from transaction history.
    Every feature is explained in comments for the viva.

    Features added:
      velocity_24h           – how many transactions this sender made in the last 24 h
      volume_24h             – total amount this sender sent in the last 24 h
      near_threshold_count_24h – count of amounts in [9000, 10000) in last 24 h
                                 (structuring indicator: splitting to avoid reporting)
      prior_transaction_count – lifetime count of sender's earlier transactions
      prior_average_amount    – average amount of sender's earlier transactions
      is_near_threshold       – 1 if this transaction amount is in [9000, 10000)
      risk_score              – simple rule-based risk score (0-100)
      risk_category           – low / medium / high based on risk_score
    """
    THRESHOLD = 10_000.0  # illustrative reporting threshold

    # Mark transactions near the threshold (structuring indicator)
    df = df.withColumn(
        "is_near_threshold",
        ((F.col("amount") >= THRESHOLD * 0.9) & (F.col("amount") < THRESHOLD)).cast("int"),
    )

    # Convert timestamp to epoch seconds for window calculations
    df = df.withColumn("_epoch", F.unix_timestamp("timestamp"))

    # Windows: partition by sender, ordered by time
    acct_window = Window.partitionBy("sender_account").orderBy("_epoch")
    rolling_24h = acct_window.rangeBetween(-86400, -1)   # last 24 hours (before current tx)
    prior_all   = acct_window.rangeBetween(Window.unboundedPreceding, -1)  # all prior

    # Rolling 24-hour features
    df = df.withColumn(
        "velocity_24h",                              # tx count in last 24 h
        F.coalesce(F.count("transaction_id").over(rolling_24h), F.lit(0)),
    )
    df = df.withColumn(
        "volume_24h",                                # total $ sent in last 24 h
        F.coalesce(F.sum("amount").over(rolling_24h), F.lit(0.0)),
    )
    df = df.withColumn(
        "near_threshold_count_24h",                  # near-threshold txns in last 24 h
        F.coalesce(F.sum("is_near_threshold").over(rolling_24h), F.lit(0)),
    )

    # Lifetime features
    df = df.withColumn(
        "prior_transaction_count",                   # total prior txns by this sender
        F.coalesce(F.count("transaction_id").over(prior_all), F.lit(0)),
    )
    df = df.withColumn(
        "prior_average_amount",                      # average amount of prior txns
        F.coalesce(F.avg("amount").over(prior_all), F.lit(0.0)),
    )

    # Simple rule-based risk score (each indicator adds points)
    risk = (
        F.when(F.col("is_near_threshold") == 1, 25).otherwise(0)                    # near threshold
        + F.when(F.col("near_threshold_count_24h") >= 2, 35).otherwise(0)            # repeated structuring
        + F.when(F.col("velocity_24h") >= 5, 15).otherwise(0)                        # high velocity
    )
    df = df.withColumn("risk_score", F.least(risk, F.lit(100)).cast("int"))

    df = df.withColumn(
        "risk_category",
        F.when(F.col("risk_score") >= 60, "high")
        .when(F.col("risk_score") >= 30, "medium")
        .otherwise("low"),
    )

    # Drop the helper epoch column
    df = df.drop("_epoch")

    return df
