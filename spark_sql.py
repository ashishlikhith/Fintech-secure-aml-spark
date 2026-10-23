"""
spark_sql.py
------------
Contains 7 Spark SQL queries that demonstrate syllabus concepts:
  1. SELECT + WHERE           – filter high-value transactions
  2. GROUP BY + HAVING        – senders with many transactions
  3. ORDER BY + DISTINCT      – distinct payment types ranked
  4. COUNT, SUM, AVG, MIN, MAX – aggregate statistics
  5. JOIN                     – join transactions with a sender summary
  6. LIKE with % and _        – wildcard pattern matching
  7. Combined query           – GROUP BY + ORDER BY + aggregates

Each query uses actual columns from the IBM AML dataset.

Viva summary:
  - build_queries()      → returns a dict of {name: SQL string}
  - run_sql_queries()    → registers temp views and executes all queries
"""

from pyspark.sql import DataFrame, SparkSession


def build_queries() -> dict[str, str]:
    """
    Return a dict of named SQL queries.
    All queries run against the 'transactions' temp view which has columns:
      transaction_id, timestamp, sender_account, receiver_account,
      amount, payment_type, payment_currency, is_laundering,
      velocity_24h, volume_24h, risk_score, risk_category, etc.
    """
    return {
        # ------------------------------------------------------------------
        # Q1: SELECT + WHERE – find transactions above $50,000
        # Demonstrates: basic column selection and row filtering
        # ------------------------------------------------------------------
        "high_value_transactions": """
            SELECT transaction_id, sender_account, receiver_account,
                   amount, payment_type, is_laundering
            FROM transactions
            WHERE amount > 50000
            ORDER BY amount DESC
            LIMIT 20
        """,

        # ------------------------------------------------------------------
        # Q2: GROUP BY + HAVING – senders with 3+ transactions
        # Demonstrates: grouping, aggregate filter (HAVING), COUNT
        # ------------------------------------------------------------------
        "active_senders": """
            SELECT sender_account,
                   COUNT(*)    AS tx_count,
                   SUM(amount) AS total_sent
            FROM transactions
            GROUP BY sender_account
            HAVING COUNT(*) >= 3
            ORDER BY tx_count DESC
            LIMIT 20
        """,

        # ------------------------------------------------------------------
        # Q3: ORDER BY + DISTINCT – list all unique payment types
        # Demonstrates: DISTINCT keyword and alphabetical ordering
        # ------------------------------------------------------------------
        "distinct_payment_types": """
            SELECT DISTINCT payment_type
            FROM transactions
            ORDER BY payment_type
        """,

        # ------------------------------------------------------------------
        # Q4: COUNT, SUM, AVG, MIN, MAX – dataset-wide statistics
        # Demonstrates: all five aggregate functions in one query
        # ------------------------------------------------------------------
        "transaction_statistics": """
            SELECT COUNT(*)    AS total_transactions,
                   SUM(amount) AS total_amount,
                   AVG(amount) AS avg_amount,
                   MIN(amount) AS min_amount,
                   MAX(amount) AS max_amount
            FROM transactions
        """,

        # ------------------------------------------------------------------
        # Q5: JOIN – join transactions with per-sender summary
        # Demonstrates: INNER JOIN on a derived table (subquery)
        # ------------------------------------------------------------------
        "join_sender_summary": """
            SELECT t.transaction_id,
                   t.sender_account,
                   t.amount,
                   s.tx_count,
                   s.avg_amount AS sender_avg
            FROM transactions t
            INNER JOIN (
                SELECT sender_account,
                       COUNT(*)    AS tx_count,
                       AVG(amount) AS avg_amount
                FROM transactions
                GROUP BY sender_account
            ) s ON t.sender_account = s.sender_account
            WHERE s.tx_count >= 3
            ORDER BY s.tx_count DESC, t.amount DESC
            LIMIT 20
        """,

        # ------------------------------------------------------------------
        # Q6: LIKE with % and _ – wildcard pattern searches
        # Demonstrates: % matches any characters, _ matches exactly one
        # ------------------------------------------------------------------
        "wildcard_search": """
            SELECT transaction_id, payment_type, amount
            FROM transactions
            WHERE payment_type LIKE '%Wire%'
               OR payment_type LIKE 'C___k'
            ORDER BY amount DESC
            LIMIT 20
        """,

        # ------------------------------------------------------------------
        # Q7: Combined – AML flag rate by payment type
        # Demonstrates: GROUP BY, aggregates, ORDER BY together
        # ------------------------------------------------------------------
        "aml_rate_by_payment_type": """
            SELECT payment_type,
                   COUNT(*)              AS total,
                   SUM(is_laundering)    AS flagged,
                   AVG(is_laundering)    AS flag_rate
            FROM transactions
            GROUP BY payment_type
            ORDER BY flag_rate DESC
        """,
    }


def run_sql_queries(spark: SparkSession, df: DataFrame) -> dict[str, DataFrame]:
    """
    Register the enriched DataFrame as a temp view and execute all queries.
    Returns a dict of {query_name: result_DataFrame}.
    """
    # Register so SQL can reference it as 'transactions'
    df.createOrReplaceTempView("transactions")

    results = {}
    for name, sql in build_queries().items():
        results[name] = spark.sql(sql)

    return results
