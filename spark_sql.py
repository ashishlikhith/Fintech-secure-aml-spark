"""Short Spark SQL examples using real IBM AMLSim transaction columns."""

from pyspark.sql import DataFrame, SparkSession


def build_queries() -> dict[str, str]:
    """Return seven small queries demonstrating the required SQL topics."""
    return {
        # SELECT and WHERE
        "high_value_transactions": """
            SELECT transaction_id, sender_account, receiver_account,
                   amount, payment_type
            FROM transactions
            WHERE amount > 50000
            ORDER BY amount DESC
            LIMIT 20
        """,
        # GROUP BY and HAVING, with COUNT and SUM
        "active_senders": """
            SELECT sender_account, COUNT(*) AS transaction_count,
                   SUM(amount) AS total_sent
            FROM transactions
            GROUP BY sender_account
            HAVING COUNT(*) >= 3
            ORDER BY transaction_count DESC
            LIMIT 20
        """,
        # DISTINCT and ORDER BY
        "payment_types": """
            SELECT DISTINCT payment_type
            FROM transactions
            ORDER BY payment_type
        """,
        # COUNT, SUM, AVG, MIN, and MAX
        "transaction_statistics": """
            SELECT COUNT(*) AS transaction_count, SUM(amount) AS total_amount,
                   AVG(amount) AS average_amount, MIN(amount) AS minimum_amount,
                   MAX(amount) AS maximum_amount
            FROM transactions
        """,
        # Join each transaction to a summary of its sender account.
        "sender_account_join": """
            SELECT t.transaction_id, t.sender_account, t.amount,
                   a.transaction_count, a.average_amount
            FROM transactions AS t
            INNER JOIN (
                SELECT sender_account, COUNT(*) AS transaction_count,
                       AVG(amount) AS average_amount
                FROM transactions
                GROUP BY sender_account
            ) AS a ON t.sender_account = a.sender_account
            WHERE a.transaction_count >= 3
            ORDER BY a.transaction_count DESC, t.amount DESC
            LIMIT 20
        """,
        # LIKE: % matches any number of characters; _ matches one character.
        "wildcard_payment_search": """
            SELECT transaction_id, payment_type, amount
            FROM transactions
            WHERE payment_type LIKE '%Wire%'
               OR payment_type LIKE 'C_____'
            ORDER BY amount DESC
            LIMIT 20
        """,
        # Compare the AML label rate across actual payment formats.
        "aml_rate_by_payment_type": """
            SELECT payment_type, COUNT(*) AS transaction_count,
                   SUM(is_laundering) AS aml_positive,
                   AVG(is_laundering) AS aml_rate
            FROM transactions
            GROUP BY payment_type
            ORDER BY aml_rate DESC
        """,
    }


def run_sql_queries(spark: SparkSession, transactions: DataFrame) -> dict[str, DataFrame]:
    """Register the transaction table and execute each syllabus query."""
    transactions.createOrReplaceTempView("transactions")
    return {
        name: spark.sql(query)
        for name, query in build_queries().items()
    }
