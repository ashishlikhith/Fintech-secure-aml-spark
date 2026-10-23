import csv
from pathlib import Path
from time import perf_counter

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


CUSTOMER_COLUMNS = (
    "user_id",
    "full_name",
    "device_id",
    "contact_email",
    "signup_country",
    "account_created_at",
)
CUSTOMER_SCHEMA = StructType(
    [
        StructField("user_id", StringType(), False),
        StructField("full_name", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("contact_email", StringType(), True),
        StructField("signup_country", StringType(), True),
        StructField("account_created_at", StringType(), True),
    ]
)
TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("sender_account", StringType(), True),
        StructField("receiver_account", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("payment_type", StringType(), True),
        StructField("is_laundering", IntegerType(), True),
    ]
)


def _require_file(path: str, description: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")


def _clean_transactions(
    raw_transactions: DataFrame, content_addressed_ids: bool = False
) -> DataFrame:
    if "payment_currency" not in raw_transactions.columns:
        raw_transactions = raw_transactions.withColumn(
            "payment_currency", F.lit("UNKNOWN")
        )
    else:
        raw_transactions = raw_transactions.withColumn(
            "payment_currency",
            F.when(
                F.trim(F.col("payment_currency")) == "", "UNKNOWN"
            ).otherwise(F.trim(F.col("payment_currency"))),
        )
    raw_transactions.createOrReplaceTempView("transactions_raw")
    transactions = (
        raw_transactions.withColumn("timestamp", F.to_timestamp("timestamp"))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("is_laundering", F.col("is_laundering").cast("int"))
    )
    valid_transaction = (
        F.col("transaction_id").isNotNull()
        & F.col("timestamp").isNotNull()
        & F.col("sender_account").isNotNull()
        & F.col("receiver_account").isNotNull()
        & F.col("amount").isNotNull()
        & (F.col("amount") >= 0)
        & F.col("payment_type").isNotNull()
        & (
            F.col("is_laundering").isNull()
            | F.col("is_laundering").isin(0, 1)
        )
    )
    quality = transactions.agg(
        F.count("*").alias("input_rows"),
        F.sum(F.when(valid_transaction, 1).otherwise(0)).alias("valid_rows"),
    ).first()
    invalid_transaction_count = int(quality["input_rows"] - (quality["valid_rows"] or 0))
    if invalid_transaction_count:
        print(f"Skipped {invalid_transaction_count} invalid transaction row(s).")
    if not quality["valid_rows"]:
        raise ValueError("Transaction input contains no valid transaction records.")
    transactions = transactions.filter(valid_transaction)
    if content_addressed_ids:
        transactions = transactions.dropDuplicates(["transaction_id"])
    else:
        transactions = transactions.dropDuplicates()
        transaction_id_counts = transactions.groupBy("transaction_id").count()
        conflicting_transaction_ids = transaction_id_counts.filter(
            F.col("count") > 1
        ).select("transaction_id")
        conflicting_count = conflicting_transaction_ids.count()
        if conflicting_count:
            print(
                f"Excluded {conflicting_count} transaction id(s) with conflicting rows."
            )
            transactions = transactions.join(
                conflicting_transaction_ids, "transaction_id", "left_anti"
            )
        if not transactions.limit(1).count():
            raise ValueError("Transaction input contains no valid transaction records.")
    return transactions


def _ibm_column_indexes(path: str) -> dict[str, int]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source_file:
        header = next(csv.reader(source_file), None)
    if not header:
        raise ValueError("IBM transaction input is empty or has no CSV header.")

    normalized = [column.strip().casefold() for column in header]
    account_indexes = [
        index for index, column in enumerate(normalized) if column == "account"
    ]
    required_columns = {
        "timestamp": "timestamp",
        "from_bank": "from bank",
        "to_bank": "to bank",
        "amount_paid": "amount paid",
        "payment_currency": "payment currency",
        "payment_format": "payment format",
        "is_laundering": "is laundering",
    }
    indexes = {
        key: normalized.index(column)
        for key, column in required_columns.items()
        if column in normalized
    }
    if len(account_indexes) >= 2:
        indexes["sender_account"] = account_indexes[0]
        indexes["receiver_account"] = account_indexes[1]
    missing_columns = set(required_columns).difference(indexes)
    if "sender_account" not in indexes or "receiver_account" not in indexes:
        missing_columns.add("two Account columns")
    if missing_columns:
        raise ValueError(
            "IBM transaction CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    indexes["column_count"] = len(header)
    return indexes


def run_ibm_data_engineering(
    spark: SparkSession,
    tx_file_path: str,
    near_threshold: float = 10000.0,
    near_threshold_ratio: float = 0.9,
    lookback_seconds: int = 86400,
    repeated_near_threshold_count: int = 2,
    high_velocity_count: int = 5,
    max_rows: int | None = None,
) -> DataFrame:
    """Load IBM AMLSim transactions and enrich them without synthetic profiles."""
    if near_threshold <= 0:
        raise ValueError("near_threshold must be greater than zero")
    if not 0 < near_threshold_ratio < 1:
        raise ValueError("near_threshold_ratio must be between zero and one")
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be greater than zero")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")
    _require_file(tx_file_path, "IBM transaction")
    started = perf_counter()
    indexes = _ibm_column_indexes(tx_file_path)
    source_columns = [
        f"source_{index}" for index in range(indexes["column_count"])
    ]
    source_schema = StructType(
        [StructField(column, StringType(), True) for column in source_columns]
    )
    source = (
        spark.read.option("header", "false")
        .option("mode", "PERMISSIVE")
        .schema(source_schema)
        .csv(tx_file_path)
    )

    def source_column(name: str) -> Column:
        return F.trim(F.col(source_columns[indexes[name]]))

    source = source.filter(
        F.lower(
            F.regexp_replace(source_column("timestamp"), "^\ufeff", "")
        )
        != "timestamp"
    )
    if max_rows is not None:
        source = source.limit(max_rows)

    parsed_timestamp = F.coalesce(
        F.to_timestamp(source_column("timestamp"), "yyyy/MM/dd HH:mm:ss"),
        F.to_timestamp(source_column("timestamp"), "yyyy/MM/dd HH:mm"),
        F.to_timestamp(source_column("timestamp"), "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(source_column("timestamp"), "yyyy-MM-dd HH:mm"),
        F.to_timestamp(source_column("timestamp")),
    )
    transaction_id = F.concat(
        F.lit("IBM-"),
        F.sha2(
            F.to_json(
                F.struct(
                    *[
                        F.col(column).alias(f"field_{index}")
                        for index, column in enumerate(source_columns)
                    ]
                )
            ),
            256,
        ),
    )
    raw_transactions = source.select(
        transaction_id.alias("transaction_id"),
        F.date_format(parsed_timestamp, "yyyy-MM-dd HH:mm:ss").alias("timestamp"),
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
        source_column("amount_paid").alias("amount"),
        source_column("payment_currency").alias("payment_currency"),
        source_column("payment_format").alias("payment_type"),
        source_column("is_laundering").alias("is_laundering"),
    )
    transactions = _clean_transactions(
        raw_transactions, content_addressed_ids=True
    )
    customers = spark.createDataFrame([], schema=CUSTOMER_SCHEMA)
    return _enrich_transactions(
        customers,
        transactions,
        near_threshold,
        near_threshold_ratio,
        lookback_seconds,
        repeated_near_threshold_count,
        high_velocity_count,
        started,
    )


def _parse_customer_line(line: str) -> tuple[str, ...] | None:
    values = [value.strip() for value in line.split("|")]
    if len(values) != len(CUSTOMER_COLUMNS) or not values[0]:
        return None
    return tuple(value or None for value in values)


def run_data_engineering(
    spark: SparkSession,
    user_file_path: str,
    tx_file_path: str,
    near_threshold: float = 10000.0,
    near_threshold_ratio: float = 0.9,
    lookback_seconds: int = 86400,
    repeated_near_threshold_count: int = 2,
    high_velocity_count: int = 5,
) -> DataFrame:
    """Load, clean, join, and lazily enrich the customer and transaction data."""
    if near_threshold <= 0:
        raise ValueError("near_threshold must be greater than zero")
    if not 0 < near_threshold_ratio < 1:
        raise ValueError("near_threshold_ratio must be between zero and one")
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be greater than zero")

    _require_file(user_file_path, "Customer")
    _require_file(tx_file_path, "Transaction")
    started = perf_counter()

    customer_lines = spark.sparkContext.textFile(user_file_path).filter(
        lambda line: bool(line.strip()) and not line.startswith("user_id|")
    )
    parsed_customers = customer_lines.map(_parse_customer_line)
    valid_customers = parsed_customers.filter(lambda row: row is not None)
    malformed_customer_count = parsed_customers.count() - valid_customers.count()
    if malformed_customer_count:
        print(f"Skipped {malformed_customer_count} malformed customer row(s).")

    customers = spark.createDataFrame(valid_customers, schema=CUSTOMER_SCHEMA)
    customers = customers.dropDuplicates()
    customers = customers.withColumn(
        "account_created_at", F.to_date("account_created_at")
    )
    if not customers.filter(F.col("user_id").isNotNull()).limit(1).count():
        raise ValueError("Customer input contains no valid customer records.")

    customer_id_counts = customers.groupBy("user_id").count()
    ambiguous_customer_ids = customer_id_counts.filter(F.col("count") > 1).select(
        "user_id"
    )
    ambiguous_count = ambiguous_customer_ids.count()
    if ambiguous_count:
        print(
            f"Excluded {ambiguous_count} customer id(s) with conflicting profile rows."
        )
        customers = customers.join(ambiguous_customer_ids, "user_id", "left_anti")

    for column in ("device_id", "contact_email", "signup_country"):
        customers = customers.withColumn(
            column,
            F.when(F.trim(F.col(column)) == "", None).otherwise(F.trim(F.col(column))),
        )

    raw_transactions = (
        spark.read.option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(TRANSACTION_SCHEMA)
        .csv(tx_file_path)
    )
    transactions = _clean_transactions(raw_transactions)

    return _enrich_transactions(
        customers,
        transactions,
        near_threshold,
        near_threshold_ratio,
        lookback_seconds,
        repeated_near_threshold_count,
        high_velocity_count,
        started,
    )


def _enrich_transactions(
    customers: DataFrame,
    transactions: DataFrame,
    near_threshold: float,
    near_threshold_ratio: float,
    lookback_seconds: int,
    repeated_near_threshold_count: int,
    high_velocity_count: int,
    started: float,
) -> DataFrame:
    if near_threshold <= 0:
        raise ValueError("near_threshold must be greater than zero")
    if not 0 < near_threshold_ratio < 1:
        raise ValueError("near_threshold_ratio must be between zero and one")
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be greater than zero")

    customers.createOrReplaceTempView("customers")
    transactions.createOrReplaceTempView("transactions")
    device_counts = (
        customers.filter(F.col("device_id").isNotNull())
        .groupBy("device_id")
        .agg(F.countDistinct("user_id").alias("shared_device_count"))
    )
    contact_counts = (
        customers.filter(F.col("contact_email").isNotNull())
        .groupBy("contact_email")
        .agg(F.countDistinct("user_id").alias("shared_contact_count"))
    )
    joined = (
        transactions.alias("t")
        .join(
            customers.select(
                "user_id",
                "full_name",
                "device_id",
                "contact_email",
                "signup_country",
                "account_created_at",
            ).alias("c"),
            F.col("t.sender_account") == F.col("c.user_id"),
            "left",
        )
        .drop("user_id")
        .join(device_counts, "device_id", "left")
        .join(contact_counts, "contact_email", "left")
        .fillna(
            1,
            subset=["shared_device_count", "shared_contact_count"],
        )
        .withColumn("event_epoch", F.unix_timestamp("timestamp"))
        .withColumn(
            "is_near_threshold",
            (
                (F.col("amount") >= near_threshold * near_threshold_ratio)
                & (F.col("amount") < near_threshold)
            ).cast("int"),
        )
    )

    account_window = Window.partitionBy("sender_account").orderBy("event_epoch")
    rolling_window = account_window.rangeBetween(-lookback_seconds, -1)
    prior_window = account_window.rangeBetween(Window.unboundedPreceding, -1)
    enriched = (
        joined.withColumn(
            "velocity_24h",
            F.coalesce(F.count("transaction_id").over(rolling_window), F.lit(0)),
        )
        .withColumn(
            "volume_24h",
            F.coalesce(F.sum("amount").over(rolling_window), F.lit(0.0)),
        )
        .withColumn(
            "near_threshold_count_24h",
            F.coalesce(F.sum("is_near_threshold").over(rolling_window), F.lit(0)),
        )
        .withColumn(
            "prior_transaction_count",
            F.coalesce(F.count("transaction_id").over(prior_window), F.lit(0)),
        )
        .withColumn(
            "prior_average_amount",
            F.coalesce(F.avg("amount").over(prior_window), F.lit(0.0)),
        )
    )

    shared_identity = (F.col("shared_device_count") > 1) | (
        F.col("shared_contact_count") > 1
    )
    profile_timing_conflict = (
        F.col("account_created_at").isNotNull()
        & (F.to_date("timestamp") < F.col("account_created_at"))
    )
    repeated_near_threshold = (
        F.col("near_threshold_count_24h") + F.col("is_near_threshold")
        >= repeated_near_threshold_count
    )
    high_velocity = F.col("velocity_24h") >= high_velocity_count
    risk_score = (
        F.when(F.col("is_near_threshold") == 1, 25).otherwise(0)
        + F.when(repeated_near_threshold, 35).otherwise(0)
        + F.when(shared_identity, 25).otherwise(0)
        + F.when(high_velocity, 15).otherwise(0)
        + F.when(profile_timing_conflict, 20).otherwise(0)
    )
    enriched = (
        enriched.withColumn("risk_score", F.least(risk_score, F.lit(100)).cast("int"))
        .withColumn(
            "risk_indicators",
            F.concat_ws(
                ", ",
                F.when(F.col("is_near_threshold") == 1, "near_threshold"),
                F.when(repeated_near_threshold, "repeated_near_threshold"),
                F.when(shared_identity, "shared_device_or_contact"),
                F.when(high_velocity, "high_transaction_velocity"),
                F.when(profile_timing_conflict, "transaction_before_account_creation"),
            ),
        )
        .withColumn(
            "risk_category",
            F.when(F.col("risk_score") >= 60, "high")
            .when(F.col("risk_score") >= 30, "medium")
            .otherwise("low"),
        )
        .drop("event_epoch")
    )
    enriched.createOrReplaceTempView("aml_analysis")
    print(f"Input validation and enrichment setup completed in {perf_counter() - started:.2f}s.")
    return enriched
