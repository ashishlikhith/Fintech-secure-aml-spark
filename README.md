# Fintech Secure AML Spark

An academic, Spark SQL and PySpark demonstration for transaction analysis,
illustrative structuring signals, identity indicators, and machine learning.
The default pipeline downloads the labeled IBM AMLSim Kaggle benchmark; the
records in `data/` remain available as a small synthetic example. AMLSim
transactions are simulated benchmark data, not live or verified customer
activity. Labels are not verified fraud determinations. Risk signals are
investigation leads, not proof of money laundering or identity theft. The
configurable near-threshold amount is illustrative and is not a legal reporting
threshold.

## Existing project audit

The original implementation already had a Spark session, an RDD-to-DataFrame
customer conversion, a CSV transaction input, a Spark SQL aggregation with a
rolling window, vector assembly/correlation, and a decision-tree
TrainValidationSplit. The actual customer and transaction files were empty,
however, so the original pipeline had no runnable input.

The audit also found that the CSV parser assumed all rows were valid and
comma-safe, joins and labels were not checked, malformed inputs could fail
inside a worker, the feature set was aggregated before modeling, and only
Decision Tree classification was implemented. There were no reusable SQL
queries, model comparison, regression/clustering, tests, dependencies, or
execution guidance. The old configuration claimed off-heap memory and large
parallelism without evidence that the host could support those settings. The
README, Spark config, dashboard JSON, and deployment shell script were empty or
placeholders; the dashboard and deployment files remain intentionally unused.

## Structure

```text
.
├── config/
│   └── spark_config.conf
├── dashboard/
│   └── zeppelin_aml_ui.json   # empty legacy placeholder; unused
├── data/
│   ├── aml_transactions.csv
│   └── mock_users.txt
├── deploy_cluster.sh         # empty placeholder; not a deployment claim
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── engine_init.py
│   ├── feature_engineering.py
│   ├── model_pipeline.py
│   └── sql_analytics.py
├── tests/
│   └── test_pipeline.py
├── artifacts/                 # created when the pipeline runs
│   └── correlation_heatmap.png
├── requirements.txt
├── run_pipeline.py
└── README.md
```

There is no separate account input file in the repository. Account identifiers
are derived from transaction sender/receiver fields; customer profiles are
joined to sender accounts by `user_id`.

## Requirements and run

- Python 3.10 or newer
- Java 8 or newer
- PySpark 3.5.9, Matplotlib, Seaborn, and KaggleHub (declared in `requirements.txt`)

From the repository root:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The default run downloads and uses the Kaggle IBM AML dataset, preferring its
large transaction file. Choose a dataset size or let the pipeline take the
largest transaction CSV present:

```powershell
python run_pipeline.py --ibm-size large
python run_pipeline.py --ibm-size largest
```

To run the repository's small synthetic example instead:

```powershell
python run_pipeline.py --dataset-source sample
```

KaggleHub downloads the dataset on demand. To use an already downloaded file or
directory instead:

```powershell
python run_pipeline.py --dataset-source ibm `
  --ibm-data-path "C:\path\to\HI-Small_Trans.csv"
```

If a requested size is absent, the pipeline reports available files; `largest`
selects the largest transaction CSV by file size. Large-scale training can
require substantial memory and runtime; use a Spark cluster with
`--master` for data that exceeds local resources. Kaggle access/network
availability may affect download.

The default Spark master is `local[*]`; this is a local Spark demonstration,
not a claim of execution on a cluster. To use a configured Spark cluster, set
its master explicitly and ensure the selected runtime, data paths, Java, and
Spark dependencies are available there:

```powershell
$env:SPARK_MASTER = "spark://your-master:7077"
python run_pipeline.py
```

The master may also be set with `--master`. The local Spark settings are in
`config/spark_config.conf`; `SPARK_SQL_SHUFFLE_PARTITIONS` and
`SPARK_DEFAULT_PARALLELISM` override the two corresponding values. Example
options:

```powershell
python run_pipeline.py --users data/mock_users.txt `
  --transactions data/aml_transactions.csv `
  --near-threshold 10000 --lookback-hours 24 --output-dir artifacts
```

Input paths and output paths may be absolute or relative to the project root.
The sample data are small enough to run locally. For other inputs:

- `mock_users.txt` is UTF-8, pipe-delimited, with columns
  `user_id|full_name|device_id|contact_email|signup_country|account_created_at`.
- `aml_transactions.csv` has columns
  `transaction_id,timestamp,sender_account,receiver_account,amount,payment_type,is_laundering`.
- The IBM option accepts the AMLSim transaction CSV schema, including repeated
  `Account` headers. It maps bank/account pairs to unique account identifiers,
  uses `Amount Paid`, `Payment Currency`, and `Payment Format`, and uses
  `Is Laundering` as the binary label. Currency and payment format are learned
  as categorical model features. No synthetic customer profiles are attached
  in this mode. Monetary values and rolling sums are not FX-normalized, so
  amount-derived features and the configurable threshold are illustrative
  across currencies.
- Timestamps should be parseable by Spark's `to_timestamp`; labels may be null
  or binary (`0`/`1`). Rows with missing required transaction fields, invalid
  timestamps, negative amounts, or non-binary labels are skipped with a count.
- Exact duplicate transactions are removed. Conflicting rows for one
  transaction ID and conflicting profile rows for one user ID are excluded
  rather than arbitrarily selecting a record. Invalid/missing input files and
  empty usable inputs produce explicit errors.
- The `is_laundering` column must be a suitable label before supervised
  metrics are interpreted. The example labels are synthetic; unknown or
  unlabelled data are not silently treated as ground truth.

## Processing and model behavior

`run_data_engineering` uses Spark's text RDD API to parse profile records into a
declared `StructType` DataFrame, and Spark's schema-driven CSV reader for
transactions. It cleans duplicates and invalid rows, joins the sender profile,
aggregates shared device/contact indicators, and computes account transaction
history and preceding-24-hour counts/amounts with Spark windows. A transaction
earlier than its known account creation date is an inconsistency signal.

The rule-based score combines near-threshold, repeated near-threshold,
shared-device/contact, high-velocity, and profile-timing indicators. The score
is capped at 100 and retains a text explanation in `risk_indicators`.
Indicators are not proofs and shared identifiers alone do not establish
misconduct.

The supervised models use the earliest 80% of transaction timestamps for
training and the latest 20% as a future-period holdout. Preprocessing is fitted
on training data only. Decision Tree and class-weighted Random Forest report
accuracy, positive-class precision, recall, F1, PR-AUC, and ROC-AUC; PR-AUC and
recall are especially informative when laundering labels are rare. K-means is
an unsupervised, label-free third model and reports test cluster sizes; it is
not a fraud classifier and has no classification accuracy. Class weighting
addresses skew during training but does not guarantee higher real-world
performance. Metrics are computed from held-out data; no accuracy target is
forced. A future-time split is more cautious than a random split, but it is
not a substitute for validation on a later, independently sourced dataset.

The model feature vector includes learned one-hot encoding of payment format
and payment currency. If both labels or a usable future holdout are not
available, supervised classification is skipped. IBM benchmark results measure
held-out performance on simulated AMLSim records and should not be treated as
evidence of performance on live bank transactions.

Visualizations use Spark-side aggregations and transfer only small summaries to
Matplotlib/Seaborn—not the transaction-level dataset. The run writes
`correlation_heatmap.png`, `transaction_overview.png`, `model_comparison.png`,
`confusion_matrices.png`, `roc_precision_recall.png`, and
`transaction_flow_map.png` plus `model_metrics.csv` in the output directory.
The flow map is a network diagram of account-to-account transfers, not a
geographic map: the source data provide no geographic coordinates. ROC and
precision-recall curves are approximated from 100 score bins for visualization;
the reported AUC metrics are calculated by Spark on the full held-out set.
Correlation describes linear association, not causation, and highly correlated
features may be redundant.

## Spark SQL query collection

`src/sql_analytics.py` exposes `build_sql_queries()` and
`run_sql_analytics()`. The pipeline registers `customers`, `transactions`,
`transactions_raw`, `aml_analysis`, and (when Random Forest predictions are
available) `model_predictions` views. The SQL results and their principal
output fields are:

| Query key | Purpose | Principal output |
|---|---|---|
| `transaction_totals` | Count, SUM, AVG, MIN, MAX | counts and amount aggregates |
| `highest_value_transactions` | Highest-value records | transaction, account, amount, type |
| `customer_transaction_summary` | Customer summary and HAVING filter | customer, count, total, average |
| `account_transaction_frequency` | Frequency and DISTINCT receivers | account, count, receiver count |
| `suspicious_transactions_by_risk` | Rule score categories | category, count, score average, total |
| `highest_risk_transactions` | Explainable transaction ranking | transaction, accounts, score, indicators |
| `account_risk_ranking` | Account ranking from transaction signals | account, max/average score, counts |
| `repeated_near_threshold_accounts` | Configurable illustrative range | account, count, total near threshold |
| `shared_contact_information` | Reused contact signal | contact and profile count |
| `customer_account_join` | Profile/transaction join | transaction and sender profile fields |
| `inbound_outbound_totals` | In/out account flows | account and inbound/outbound totals |
| `daily_transaction_summary` | Daily aggregation | date, count, sum, average |
| `duplicate_transaction_ids` | Duplicates in raw parsed input | ID and row count |
| `missing_data_summary` | Missing transaction fields/labels | missing-field counts |
| `missing_customer_profile_data` | Missing customer attributes | missing-field counts |
| `payment_type_wildcard_search` | `LIKE`, `%`, and `_` examples | matching transaction fields |
| `similar_transaction_patterns` | Repeated account/type patterns | account, type, counts, total |
| `model_prediction_summary` | Optional prediction summary | prediction and count/rate |
| `model_label_comparison` | Optional confusion-table summary | actual label, prediction, count |

The raw-input quality queries run against `transactions_raw` before the cleaned
view is deduplicated. Near-threshold SQL uses 90% of the configurable
illustrative threshold as its lower bound.

## Testing

Run the standard-library test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

Tests cover the sample input row counts, joined profile/transaction data,
repeated near-threshold and shared-contact signals, SQL summaries/wildcards,
missing input handling, IBM AML CSV ingestion, generated visualizations, and
model task outputs. The exact runtime and model metrics depend on the
Spark/Java environment; the repository does not claim precomputed performance
benchmarks or fraud accuracy.

## Syllabus mapping

| Syllabus requirement | Implemented in |
|---|---|
| Spark SQL/DataFrames and schema-driven ingestion | `src/data_pipeline.py`, `src/sql_analytics.py` |
| RDD-to-DataFrame conversion | `run_data_engineering()` |
| Column operations, filters, sorting, transformations, null/duplicate handling | `src/data_pipeline.py` |
| SQL SELECT/WHERE/GROUP BY/HAVING/ORDER BY/DISTINCT and aggregates | `build_sql_queries()` |
| Customer/account/transaction joins and SQL wildcards | `src/sql_analytics.py` |
| Identity and transaction risk indicators; transparent score | `run_data_engineering()` |
| Feature assembly, train/test split, classification metrics | `src/feature_engineering.py`, `src/model_pipeline.py` |
| Decision Tree and Random Forest | class-weighted binary classification |
| K-means | unlabeled behavioral clustering; no fraud metric assigned |
| Confusion matrix, accuracy, precision, recall, F1, PR-AUC, ROC-AUC | future-period classification results |
| Correlation matrix and heatmap | `process_ml_features()` |
| Execution timing and local/cluster configuration | `src/engine_init.py`, pipeline modules |
| Empty and invalid input handling | `src/data_pipeline.py`, `tests/test_pipeline.py` |

## Demonstration workflow

1. Load synthetic profiles using an RDD and transactions using a declared CSV
   schema.
2. Validate, clean, deduplicate, and register Spark SQL temporary views.
3. Join sender profiles, calculate shared-identifier and rolling-window
   signals, then attach a transparent rule-based score.
4. Run the named SQL queries for transaction totals, joins, identity signals,
   near-threshold patterns, data quality, and account flows.
5. Prepare model features (including payment format and currency) and render
   transaction overview, correlation, model metric, confusion matrix, score
   curve, and transaction-flow network visualizations.
6. Train a Decision Tree and Random Forest on earlier transactions, evaluate
   on the latest period, and run K-means without labels.
7. Review precision/recall and PR-AUC alongside confusion matrices, inspect the
   SQL model-prediction summary when labels allow, and investigate high-scoring
   records as signals requiring human review.
