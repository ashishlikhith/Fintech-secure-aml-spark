# Fintech Secure AML Spark

A local PySpark demonstration that analyzes the IBM AMLSim **HI-Small**
transaction CSV, creates illustrative risk signals, trains exactly three
classifiers, evaluates them on a chronological holdout, runs Spark SQL
summaries, and writes metrics and visualizations.

AMLSim records and labels are simulated benchmark data. Predictions and risk
scores are not verified fraud determinations and must not be used as proof of
misconduct. The near-threshold signal is illustrative, not a legal reporting
threshold.

## Requirements

- Use the existing Python, Java, and PySpark setup; this project does not
  upgrade or replace them.
- Project dependencies are in `requirements.txt`.
- The default local Spark master is `local[2]`, with 32 SQL shuffle partitions
  to moderate per-partition work on the supplied dataset.

## Run on Windows PowerShell

From the project root:

```powershell
python main.py
```

This reads the smaller `data\HI-Small_Trans.csv` benchmark (about 476 MB),
not the 3 GB HI-Medium file. The command deliberately
accepts only a file named `HI-Small_Trans.csv`; it cannot select the medium or
large benchmark. To specify another copy of the small CSV:

```powershell
python main.py --data-path "C:\data\HI-Small_Trans.csv" `
  --output-dir "artifacts\hi-small"
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

Training, validation, and test data are chronological 60/20/20 time periods.
Preprocessing is learned on training data only. Class weights are applied to
the training period to address label imbalance. The validation period selects
a probability threshold for F1; the final period reports precision, recall,
F1, false-positive/negative counts, PR-AUC, and ROC-AUC. Each split must
contain both labels or the run stops with its class counts.

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

Spark caches the enriched data and model feature/prediction frames on disk
rather than keeping the full dataset in JVM memory. SQL summaries and chart
inputs are bounded before they are transferred to Python. Spark model objects
are not persisted; metrics, evaluation summaries, SQL results, and plots are
the saved deliverables.

The HI-Small run emitted Spark `MemoryStore` cache warnings; the corresponding
blocks fell back to disk, and the pipeline completed without a heap or Py4J
error. Keep sufficient free disk space for Spark's local spill/cache files.

## Spark SQL analyses

The pipeline executes and saves bounded outputs for transaction totals,
suspicious transactions by risk category, highest-risk transactions, payment
type summaries, near-threshold sender patterns, and sender/receiver patterns.
It also demonstrates beginner SQL concepts on at most 100 transaction rows:
DataFrame-to-RDD-to-DataFrame conversion using `RDD.toDF()`, temporary views,
`SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, aggregate functions, `INNER JOIN`,
and `LIKE '%WIRE%'`. The example is deliberately bounded; the AML summaries
run against the full HI-Small input. The entry point prints the DataFrame
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
