from pyspark.sql import DataFrame, SparkSession


def _register_rdd_example_views(
    spark: SparkSession, transactions: DataFrame
) -> None:
    """Convert a bounded DataFrame sample to an RDD, then back to a DataFrame."""
    example_columns = [
        "transaction_id",
        "sender_account",
        "receiver_account",
        "amount",
        "payment_type",
        "is_laundering",
    ]
    small_dataframe = transactions.select(*example_columns).limit(100)

    # This deliberately small conversion demonstrates the RDD-to-DataFrame API.
    transaction_rdd = small_dataframe.rdd
    rdd_dataframe = transaction_rdd.toDF(schema=small_dataframe.schema)
    # RDD.toDF uses PySpark's active session; rebind this tiny example to the
    # session whose SQL queries will read its temporary views.
    rdd_dataframe = spark.createDataFrame(
        rdd_dataframe.rdd, schema=small_dataframe.schema
    )
    rdd_dataframe.createOrReplaceTempView("rdd_transactions")

    # DataFrame aggregation creates the second view used by the SQL join lesson.
    sender_summary = rdd_dataframe.groupBy("sender_account").count()
    sender_summary.createOrReplaceTempView("rdd_sender_summary")


def build_sql_queries(near_threshold: float = 10000.0) -> dict[str, str]:
    """Return AML summaries plus short SQL examples for learning."""
    if near_threshold <= 0:
        raise ValueError("near_threshold must be greater than zero")
    lower_bound = near_threshold * 0.9
    return {
        "transaction_totals": """
            SELECT COUNT(*) AS transaction_count,
                   SUM(amount) AS total_amount,
                   AVG(amount) AS average_amount,
                   MIN(amount) AS minimum_amount,
                   MAX(amount) AS maximum_amount
            FROM transactions
        """,
        "suspicious_transactions_by_risk": """
            SELECT risk_category,
                   COUNT(*) AS transaction_count,
                   AVG(risk_score) AS average_risk_score,
                   SUM(amount) AS total_amount
            FROM aml_analysis
            GROUP BY risk_category
            ORDER BY average_risk_score DESC
        """,
        "highest_risk_transactions": """
            SELECT transaction_id, timestamp, sender_account, receiver_account,
                   amount, risk_score, risk_category, risk_indicators
            FROM aml_analysis
            WHERE risk_score > 0
            ORDER BY risk_score DESC, amount DESC
            LIMIT 20
        """,
        "payment_type_summary": """
            SELECT payment_type,
                   COUNT(*) AS transaction_count,
                   SUM(is_laundering) AS labeled_positive_count,
                   AVG(is_laundering) AS labeled_positive_rate
            FROM transactions
            GROUP BY payment_type
            ORDER BY transaction_count DESC
            LIMIT 20
        """,
        "near_threshold_summary": f"""
            SELECT sender_account,
                   COUNT(*) AS transaction_count,
                   SUM(amount) AS total_amount
            FROM aml_analysis
            WHERE amount >= {lower_bound} AND amount < {near_threshold}
            GROUP BY sender_account
            ORDER BY transaction_count DESC, total_amount DESC
            LIMIT 20
        """,
        "sender_receiver_patterns": """
            SELECT sender_account, receiver_account,
                   COUNT(*) AS transaction_count,
                   SUM(amount) AS total_amount,
                   SUM(is_laundering) AS labeled_positive_count
            FROM transactions
            GROUP BY sender_account, receiver_account
            ORDER BY transaction_count DESC, total_amount DESC
            LIMIT 20
        """,
        "like_wildcard_example": """
            SELECT transaction_id, payment_type, amount
            FROM rdd_transactions
            WHERE LOWER(payment_type) LIKE '%wire%'
            ORDER BY amount DESC
            LIMIT 20
        """,
        "dataframe_join_sql_example": """
            SELECT summary.sender_account,
                   summary.count AS sender_transaction_count,
                   tx.payment_type,
                   tx.is_laundering
            FROM rdd_sender_summary AS summary
            INNER JOIN rdd_transactions AS tx
                ON summary.sender_account = tx.sender_account
            WHERE LOWER(tx.payment_type) LIKE '%wire%'
            ORDER BY summary.count DESC
            LIMIT 20
        """,
    }


def run_sql_analytics(
    spark: SparkSession,
    near_threshold: float = 10000.0,
    transactions: DataFrame | None = None,
) -> dict[str, DataFrame]:
    """Register temporary views and execute the AML and teaching queries."""
    if transactions is None:
        transactions = spark.table("transactions")
    transactions.createOrReplaceTempView("transactions")
    transactions.createOrReplaceTempView("aml_analysis")
    _register_rdd_example_views(spark, transactions)

    return {
        name: spark.sql(query)
        for name, query in build_sql_queries(near_threshold).items()
    }
