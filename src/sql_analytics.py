from pyspark.sql import DataFrame, SparkSession


def build_sql_queries(near_threshold: float = 10000.0) -> dict[str, str]:
    """Return bounded Spark SQL queries for transaction and risk analysis."""
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
    }


def run_sql_analytics(
    spark: SparkSession,
    near_threshold: float = 10000.0,
) -> dict[str, DataFrame]:
    """Execute the AML queries against the registered pipeline views."""
    return {
        name: spark.sql(query)
        for name, query in build_sql_queries(near_threshold).items()
    }
