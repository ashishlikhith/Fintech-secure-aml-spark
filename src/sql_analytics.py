from pyspark.sql import DataFrame, SparkSession


def build_sql_queries(near_threshold: float = 10000.0) -> dict[str, str]:
    """Return reusable Spark SQL statements using an illustrative threshold."""
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
        "highest_value_transactions": """
            SELECT transaction_id, timestamp, sender_account, receiver_account,
                   amount, payment_type
            FROM transactions
            ORDER BY amount DESC, timestamp DESC
            LIMIT 20
        """,
        "customer_transaction_summary": """
            SELECT c.user_id, c.full_name, c.signup_country,
                   COUNT(t.transaction_id) AS transaction_count,
                   SUM(t.amount) AS sent_amount,
                   AVG(t.amount) AS average_transaction_amount
            FROM customers c
            LEFT JOIN transactions t ON c.user_id = t.sender_account
            GROUP BY c.user_id, c.full_name, c.signup_country
            HAVING COUNT(t.transaction_id) > 0
            ORDER BY sent_amount DESC
        """,
        "account_transaction_frequency": """
            SELECT sender_account AS account_id,
                   COUNT(*) AS transaction_count,
                   COUNT(DISTINCT receiver_account) AS distinct_receivers
            FROM transactions
            GROUP BY sender_account
            ORDER BY transaction_count DESC
        """,
        "suspicious_transactions_by_risk": """
            SELECT risk_category, COUNT(*) AS transaction_count,
                   AVG(risk_score) AS average_risk_score,
                   SUM(amount) AS total_amount
            FROM aml_analysis
            GROUP BY risk_category
            ORDER BY average_risk_score DESC
        """,
        "highest_risk_transactions": """
            SELECT transaction_id, sender_account, receiver_account, amount,
                   risk_score, risk_category, risk_indicators
            FROM aml_analysis
            WHERE risk_score > 0
            ORDER BY risk_score DESC, amount DESC
            LIMIT 20
        """,
        "account_risk_ranking": """
            SELECT sender_account AS account_id,
                   MAX(risk_score) AS highest_transaction_risk,
                   AVG(risk_score) AS average_transaction_risk,
                   COUNT(*) AS transaction_count,
                   SUM(CASE WHEN risk_category = 'high' THEN 1 ELSE 0 END)
                       AS high_risk_transaction_count
            FROM aml_analysis
            GROUP BY sender_account
            ORDER BY highest_transaction_risk DESC,
                     high_risk_transaction_count DESC
        """,
        "repeated_near_threshold_accounts": f"""
            SELECT sender_account AS account_id,
                   MAX(near_threshold_count_24h)
                       AS peak_near_threshold_count_24h,
                   COUNT(*) AS near_threshold_transaction_count,
                   SUM(amount) AS near_threshold_total
            FROM aml_analysis
            WHERE amount >= {lower_bound} AND amount < {near_threshold}
            GROUP BY sender_account
            HAVING MAX(near_threshold_count_24h) >= 2
            ORDER BY near_threshold_transaction_count DESC,
                     near_threshold_total DESC
        """,
        "shared_contact_information": """
            SELECT contact_email, COUNT(DISTINCT user_id) AS profile_count
            FROM customers
            WHERE contact_email IS NOT NULL
            GROUP BY contact_email
            HAVING COUNT(DISTINCT user_id) > 1
            ORDER BY profile_count DESC
        """,
        "customer_account_join": """
            SELECT t.transaction_id, t.timestamp, c.user_id, c.full_name,
                   c.signup_country, t.sender_account, t.receiver_account,
                   t.amount, t.payment_type
            FROM transactions t
            LEFT JOIN customers c ON t.sender_account = c.user_id
            ORDER BY t.timestamp, t.transaction_id
        """,
        "inbound_outbound_totals": """
            SELECT account_id,
                   SUM(outbound_amount) AS outbound_amount,
                   SUM(inbound_amount) AS inbound_amount,
                   SUM(transaction_count) AS transaction_count
            FROM (
                SELECT sender_account AS account_id, SUM(amount) AS outbound_amount,
                       CAST(0 AS DOUBLE) AS inbound_amount, COUNT(*) AS transaction_count
                FROM transactions
                GROUP BY sender_account
                UNION ALL
                SELECT receiver_account AS account_id, CAST(0 AS DOUBLE) AS outbound_amount,
                       SUM(amount) AS inbound_amount, COUNT(*) AS transaction_count
                FROM transactions
                GROUP BY receiver_account
            ) flows
            GROUP BY account_id
            ORDER BY outbound_amount + inbound_amount DESC
        """,
        "daily_transaction_summary": """
            SELECT TO_DATE(timestamp) AS transaction_date,
                   COUNT(*) AS transaction_count,
                   SUM(amount) AS total_amount,
                   AVG(amount) AS average_amount
            FROM transactions
            GROUP BY TO_DATE(timestamp)
            ORDER BY transaction_date
        """,
        "duplicate_transaction_ids": """
            SELECT transaction_id, COUNT(*) AS row_count
            FROM transactions_raw
            WHERE transaction_id IS NOT NULL
            GROUP BY transaction_id
            HAVING COUNT(*) > 1
            ORDER BY row_count DESC
        """,
        "missing_data_summary": """
            SELECT
                SUM(CASE WHEN transaction_id IS NULL THEN 1 ELSE 0 END)
                    AS missing_transaction_ids,
                SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END)
                    AS missing_timestamps,
                SUM(CASE WHEN sender_account IS NULL THEN 1 ELSE 0 END)
                    AS missing_senders,
                SUM(CASE WHEN receiver_account IS NULL THEN 1 ELSE 0 END)
                    AS missing_receivers,
                SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END)
                    AS missing_amounts,
                SUM(CASE WHEN is_laundering IS NULL THEN 1 ELSE 0 END)
                    AS unlabeled_transactions
            FROM transactions_raw
        """,
        "missing_customer_profile_data": """
            SELECT
                SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END)
                    AS missing_user_ids,
                SUM(CASE WHEN contact_email IS NULL THEN 1 ELSE 0 END)
                    AS missing_contact_emails,
                SUM(CASE WHEN signup_country IS NULL THEN 1 ELSE 0 END)
                    AS missing_signup_countries,
                SUM(CASE WHEN account_created_at IS NULL THEN 1 ELSE 0 END)
                    AS missing_account_creation_dates
            FROM customers
        """,
        "payment_type_wildcard_search": """
            SELECT transaction_id, sender_account, payment_type, amount
            FROM transactions
            WHERE UPPER(payment_type) LIKE '%WIRE%'
               OR UPPER(payment_type) LIKE 'WIRE_%'
               OR UPPER(payment_type) LIKE '%CRYPTO%'
            ORDER BY amount DESC
        """,
        "similar_transaction_patterns": """
            SELECT sender_account, payment_type,
                   COUNT(*) AS transaction_count,
                   COUNT(DISTINCT receiver_account) AS receiver_count,
                   SUM(amount) AS total_amount
            FROM transactions
            GROUP BY sender_account, payment_type
            HAVING COUNT(*) >= 2
            ORDER BY transaction_count DESC, total_amount DESC
        """,
        "model_prediction_summary": """
            SELECT prediction, COUNT(*) AS transaction_count,
                   AVG(label) AS observed_positive_rate
            FROM model_predictions
            GROUP BY prediction
            ORDER BY prediction
        """,
        "model_label_comparison": """
            SELECT label AS actual_label, prediction,
                   COUNT(*) AS transaction_count
            FROM model_predictions
            GROUP BY label, prediction
            ORDER BY actual_label, prediction
        """,
    }


def run_sql_analytics(
    spark: SparkSession,
    near_threshold: float = 10000.0,
) -> dict[str, DataFrame]:
    """Execute the reusable SQL collection against registered pipeline views."""
    queries = build_sql_queries(near_threshold)
    optional_queries = {"model_prediction_summary", "model_label_comparison"}
    results = {}
    for name, query in queries.items():
        if name in optional_queries and not spark.catalog.tableExists(
            "model_predictions"
        ):
            continue
        results[name] = spark.sql(query)
    return results
