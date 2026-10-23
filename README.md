# Fintech Secure AML Spark

A local PySpark demonstration that creates a fixed-seed, stratified sample of
500,000 transactions from the existing IBM AMLSim **HI-Small** CSV, builds
illustrative risk signals, trains three classifiers on chronological splits,
runs Spark SQL summaries, and saves metrics and visualizations.

AMLSim records and labels are simulated benchmark data. Predictions and risk
scores are not verified fraud determinations and must not be used as proof of
misconduct. The near-threshold signal is illustrative, not a legal reporting
threshold.

## Requirements

- Use the existing Python, Java, and PySpark setup; this project does not
  upgrade or replace them.
- Project dependencies are in `requirements.txt`.
- The default local Spark master is `local[2]`, with 32 SQL shuffle partitions.
- The sample seed defaults to `42`; Python, Java, and Spark versions are not
  modified by this project.

## Run on Windows PowerShell

From the project root:

```powershell
python main.py
```

The project reads the original `data\HI-Small_Trans.csv` without changing it,
then creates or reuses `data\HI-Small_Trans_sample_500000.csv`. Sampling is
stratified on the AML label and uses seed 42 by default. Training, validation,
and test data are made from this saved sample.

To choose a different saved-sample path, or explicitly set the sample size
and seed:

```powershell
python main.py --data-path "data\HI-Small_Trans.csv" `
  --sample-path "data\HI-Small_Trans_sample_500000.csv" `
  --sample-size 500000 --sample-seed 42 `
  --output-dir "artifacts\hi-small-500k"
```

Optional settings:

```powershell
python main.py --help
```

## Pipeline and outputs

The pipeline validates and cleans transaction fields, derives account IDs
from bank/account pairs, computes prior-transaction and near-threshold
features, and creates explainable rule-based risk scores. Missing numeric
features are imputed with zero; rows with invalid timestamps, missing required
fields, negative amounts, or non-binary labels are excluded from modeling.
Unlabeled rows can remain in transaction analysis but are not included in
supervised training or evaluation.

Training, validation, and test data are disjoint chronological 60/20/20 time
periods. The run prints and saves each split's row count, AML-positive count,
and positive proportion. Preprocessing is fitted on training data only. The
transaction ID and timestamp determine split membership but are excluded from
model features; rolling features use only prior transactions. Class weights
are applied to training data. The validation period selects a probability
threshold for F1. Test metrics include precision, recall, F1, PR-AUC, ROC-AUC,
and all four confusion-matrix counts. Each split must contain both labels or
the run stops with its class counts.

Exactly three supervised classifiers are used:

1. `DecisionTreeClassifier`
2. `RandomForestClassifier`
3. `GBTClassifier`

No other classifier or unsupervised model is included. A model training,
evaluation, SQL, or output-writing error stops the run rather than being
reported as a successful pipeline.

Outputs are written to `artifacts\hi-small\` by default:

- `correlation_heatmap.png`
- `transaction_overview.png`
- `model_comparison.png` (precision, recall, F1, and PR-AUC)
- `confusion_matrices.png`
- `feature_importance.png`
- `transaction_flow_map.png`
- `model_metrics.csv`
- `metrics\model_metrics.csv`
- `metrics\evaluation_summary.json`
- `sql\*.csv` (bounded summaries and top-risk/pattern rows)

Spark stores the enriched data and model feature/prediction frames with
disk-only persistence and unpersists them after use. The Random Forest uses
12 trees, depth 4, and square-root feature selection; Decision Tree depth is
4 and GBT uses 8 iterations at depth 3. SQL summaries and chart inputs are
bounded before transfer to Python. No Spark model objects are saved.

Spark `MemoryStore` warnings indicate that some blocks could not remain in
memory and were written to disk. Such warnings alone do not establish a Java
heap failure. Keep sufficient free disk space for Spark's cache and spill
files; inspect the run log for any actual exception.

## Spark SQL analyses

The pipeline executes and saves bounded outputs for transaction totals,
suspicious transactions by risk category, highest-risk transactions, payment
type summaries, near-threshold sender patterns, and sender/receiver patterns.
It also demonstrates beginner SQL concepts on at most 100 transaction rows:
DataFrame-to-RDD-to-DataFrame conversion using `RDD.toDF()`, temporary views,
`SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, aggregate functions, `INNER JOIN`,
and `LIKE '%WIRE%'`. The example is deliberately bounded; the AML summaries
run against the saved 500,000-row sample. The entry point prints the DataFrame
schema and five example rows without sorting or collecting the full dataset.

## Python files

- `main.py`: command-line entry point; runs the stages in order and
  reads the Spark settings, validates transactions, builds account activity
  and illustrative risk features, and writes the bounded SQL query results.
- `ml_models.py`: selects model inputs, handles missing numeric values,
  creates the Spark-computed correlation heatmap, makes chronological
  train/validation/test splits, trains the three classifiers, and saves
  evaluation metrics.
- `spark_sql.py`: registers SQL views, creates the small RDD lesson,
  and holds the AML and syllabus query examples.
- `visualizations.py`: creates the transaction, model-comparison,
  confusion-matrix, feature-importance, and transaction-flow plots.
- `tests/test_pipeline.py`: exercises the pipeline with small generated data.

The AML task is a classification problem, so the project reports a confusion
matrix, precision, recall, F1, PR-AUC, and ROC-AUC. RMSE, MAE, and R-squared
are regression metrics and are not applicable to these classifiers. Linear
and logistic regression, K-means, SVM, and Naive Bayes are not mixed into the
three-model AML comparison; they can be practised as separate syllabus
exercises. K-means is not part of this project.

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
```

The tests use small generated fixtures for deterministic checks of input
validation, feature engineering, class weighting and evaluation, SQL, and
visualizations. The end-to-end HI-Small run is the integration check against
the supplied benchmark.
