# Fintech Secure AML Spark

A local PySpark batch-analytics project for exploring simulated anti-money
laundering (AML) transactions. It demonstrates data preparation, Spark SQL,
imbalanced binary classification, chronological model evaluation, and result
visualization using IBM's AMLSim HI-Small dataset.

> **Important:** AMLSim data is simulated. A positive label or model prediction
> is not proof of money laundering, and this project is not a production
> banking, real-time monitoring, or automated decision system.

## Project overview

The pipeline reads the source transaction CSV, creates a reproducible
class-stratified sample of 500,000 rows, cleans and prepares the data, and
compares three Spark ML classifiers. It also runs seven Spark SQL examples and
saves metrics, query results, and eight visualizations.

At a high level:

```text
IBM AMLSim CSV
      |
      v
Exact-size stratified sample (500,000 rows, seed 42)
      |
      v
Parse, validate, and prepare transaction columns
      |
      +---------------------> Spark SQL summaries and pattern queries
      |
      v
Chronological train / validation / test splits
      |
      v
Decision tree | Random forest | Gradient-boosted trees
      |
      v
Test metrics, CSV files, and visualizations
```

## Requirements

- Python with a working Java installation compatible with Apache Spark 3.5.
- The Python packages in `requirements.txt`.
- The original AMLSim HI-Small transaction CSV.
- A desktop Matplotlib backend to display charts during the normal run.

The project does not install or change Python, Java, PySpark, Apache Spark, or
Spark Shell. The core dependency versions are:

- `pyspark==3.5.9`
- `matplotlib>=3.7,<4`
- `pandas>=2,<3`
- `seaborn>=0.13,<1`

## Setup and run

From the repository root, install the Python dependencies in your chosen
environment:

```powershell
python -m pip install -r requirements.txt
```

Place the original dataset at:

```text
data\HI-Small_Trans.csv
```

Then run:

```powershell
python main.py
```

The program creates the sample at
`data\HI-Small_Trans_sample_500000.csv` and writes run outputs under
`artifacts\hi-small-500k\`. It does not edit the original dataset.

The normal run displays each of the eight charts in sequence. Close a chart
window to continue to the next one. On Windows, a Tk installation with
Matplotlib's `TkAgg` backend is supported. If the selected backend is not a
desktop-window backend, the program stops with an actionable error rather than
silently skipping the interactive charts.

To show existing plots without rerunning Spark or training the models:

```powershell
python main.py --show-saved-plots
```

This option expects all eight PNG files to already exist in the output
directory.

## Data and preparation

The input is the IBM AMLSim HI-Small transaction CSV. The reader locates the
required columns by header name and explicitly handles the dataset's two
columns both named `Account`.

### Sampling

The pipeline first counts the source rows and the two AML label classes. It
then selects an exact-size sample of 500,000 labeled rows while preserving the
source class proportions as closely as integer row counts allow. The
class-specific row selection uses seed `42` for reproducibility. The output is
written separately from the original, and its row count and class counts are
verified before replacing the sample file.

The source dataset must contain at least 500,000 validly labeled rows and both
label classes. If it does not, the run fails with an explicit error.

### Cleaning and derived columns

The data preparation code:

- Parses supported timestamp formats.
- Converts the amount paid to a numeric value.
- Combines bank and account values into sender and receiver IDs, for example
  `bank:account`.
- Generates a deterministic SHA-256 transaction ID from the source row.
- Removes records with an invalid timestamp, missing account or payment type,
  invalid or negative amount, or a label other than `0` or `1`.
- Adds `is_near_threshold`, equal to 1 when the amount paid is at least 9,000
  and less than 10,000, and 0 otherwise.

The amount range is a simple demonstration feature; it should not be presented
as a universal or regulatory AML threshold.

## Spark processing and SQL

The application creates a local Spark session using `local[2]`, two default
parallel workers, and eight SQL shuffle partitions. Prepared transactions and
model prediction data are cached on disk where appropriate. The main pipeline
releases its cached transaction data and stops Spark in a `finally` block.

The cleaned data is registered as a temporary Spark SQL view named
`transactions`. The seven queries in `spark_sql.py` demonstrate:

1. High-value transactions with `SELECT`, `WHERE`, sorting, and `LIMIT`.
2. Active senders with `GROUP BY`, `HAVING`, `COUNT`, and `SUM`.
3. Distinct payment types.
4. Transaction count, sum, average, minimum, and maximum.
5. A join between transactions and a sender-account summary.
6. Payment-format pattern matching with `LIKE` wildcards.
7. AML-positive counts and rates by payment type.

Each query's name and result count are printed, the first ten rows are shown,
and the complete query result is saved as a CSV under `artifacts\hi-small-500k\sql\`.

## Modeling and evaluation

### Features

The models use only these three input features:

- `amount`
- `is_near_threshold`
- `payment_type_index`

The payment-format index is learned from the training split only. The
transaction timestamp, sender and receiver IDs, transaction ID, and target
label are not model inputs. The label is used as the binary classification
target. The training data also supplies the class weights used to account for
the imbalanced classes.

### Chronological splits

The pipeline calculates timestamp quantiles and creates approximately:

- 60% earliest transactions for training.
- The next 20% for validation.
- The latest 20% for testing.

Repeated timestamps can affect the exact row percentages at a cutoff. The
pipeline reports the actual counts and positive-label prevalence for each
split. It deliberately does not stratify these chronological splits: a change
in AML-positive rate across time is retained and reported as possible temporal
distribution shift.

### Classifiers and threshold selection

The three Spark ML classifiers are:

- `DecisionTreeClassifier` (maximum depth 4).
- `RandomForestClassifier` (8 trees, maximum depth 4).
- `GBTClassifier` (5 iterations, maximum depth 3).

All use seed `42` and training-derived class weights. Each model independently
selects a probability threshold that maximizes F1 on the validation split.
That threshold is then applied to the model's future test predictions for
confusion counts, accuracy, precision, recall, and F1. PR-AUC and ROC-AUC are
calculated from the model's continuous test probability scores.

The test data is not used to select the threshold. Different models can select
different thresholds and make different numbers of false-positive and
false-negative predictions. Identical metrics, if they occur, are not changed
to force a difference.

### Metric interpretation

- **Precision:** Of the transactions flagged, what fraction are positive
  labels? Low precision means many alerts are false positives.
- **Recall:** Of the positive labels, what fraction were flagged? Low recall
  means more positive cases were missed.
- **F1:** The harmonic mean of precision and recall at the selected threshold.
- **PR-AUC:** The area under the precision-recall curve. It is particularly
  informative for rare positive classes; compare it with the test-set positive
  prevalence.
- **ROC-AUC:** The area under the receiver operating characteristic curve.
  It measures ranking across thresholds and can look strong even when
  precision is low for a rare class.
- **Accuracy:** The fraction of all rows classified correctly. It can be
  misleading when almost all rows are negative.
- **TP, TN, FP, FN:** True positives, true negatives, false positives, and
  false negatives at the selected validation threshold.

Threshold choice is an operational trade-off. A higher recall can come with
more false alerts; an alerting process needs to consider investigation
capacity and the relative cost of missed and unnecessary reviews.

## Saved example results

The checked-in example output in `artifacts\hi-small-500k\model_metrics.csv`
contains the following test metrics. In that run, the test set has 100,389
transactions, of which 195 (about 0.19%) are AML-positive.

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision tree | 0.8947 | 0.0170 | 0.9385 | 0.0335 | 0.0165 | 0.9129 |
| Random forest | 0.9805 | 0.0472 | 0.4718 | 0.0857 | 0.0315 | 0.9428 |
| Gradient-boosted trees | 0.9639 | 0.0332 | 0.6256 | 0.0631 | 0.0291 | 0.9368 |

For this saved run, the random forest has the highest F1, precision, and
PR-AUC, while the decision tree has the highest recall. The decision tree's
high recall comes with substantially lower precision. The results illustrate
why accuracy or ROC-AUC alone is not enough for a rare-event detection task.
The PR-AUC baseline is roughly the positive prevalence (about 0.0019), so
PR-AUC should be interpreted in that context.

These are example results from one saved run, not a performance guarantee.
Sampling, the source data, temporal distribution shift, and modeling choices
all affect the measured results. Re-run the pipeline to generate a new
sample/output from the available source data.

## Output files

By default, generated outputs are placed under
`artifacts\hi-small-500k\`:

- `model_metrics.csv` — one test metrics row per classifier, including
  threshold and confusion counts.
- `aml_class_distribution.png` — label counts and percentages.
- `correlation_heatmap.png` — correlation of the two numeric preparation
  columns, `amount` and `is_near_threshold`.
- `precision_recall_curves.png` — test precision-recall curves.
- `roc_curves.png` — test ROC curves and chance baseline.
- `top10_feature_importance.png` — Random Forest feature importances. There
  are only three model features, so this plot displays at most three.
- `model_comparison.png` — accuracy, PR-AUC, and ROC-AUC.
- `precision_recall_f1_comparison.png` — precision, recall, and F1.
- `confusion_matrices.png` — test confusion counts at each model's selected
  threshold.
- `sql\*.csv` — the seven Spark SQL query results.

The curves, confusion matrices, and threshold-based metrics all use the
chronological test split. The correlation plot is descriptive, not causal.

## Project files

- `main.py` — command-line entry point and end-to-end orchestration.
- `data_processing.py` — source-column discovery, sampling, cleaning, and
  feature creation.
- `ml_models.py` — chronological splitting, model training, threshold
  selection, metrics, and metrics CSV output.
- `spark_sql.py` — seven SQL query definitions and execution.
- `visualizations.py` — chart generation and display of saved charts.
- `tests\test_pipeline.py` — tests using generated transaction data.
- `requirements.txt` — Python dependencies.

The source CSV, generated sample, and artifacts may be excluded from version
control by `.gitignore`; provide the source CSV at the documented path before
running the application.

## Jupyter notebook workflow

The repository includes
[`notebooks/01_aml_spark_workflow.ipynb`](notebooks/01_aml_spark_workflow.ipynb),
a single end-to-end notebook for presentation and experimentation. It imports
the existing Python modules rather than copying their data preparation,
feature engineering, SQL, or model-training logic. Run cells from top to
bottom from the repository root in one kernel.

The notebook defaults to reusing the existing 500,000-row sample, running all
three existing model fits and seven SQL queries, and displaying fresh scores,
bounded prediction previews, SQL result tables, and charts inline as their
cells execute. It does not load stale saved metrics or plots. Run cells from
top to bottom. Model training can take several minutes. Sampling is the only
expensive processing stage disabled by default because the prepared sample
already exists.

Notebook-run metrics are written by the existing training helper into a
temporary directory that is removed after training. SQL tables and charts are
displayed directly in notebook outputs and are not written to project artifact
files. The notebook shows a small prediction preview per model rather than
dumping every test prediction; the full test scores and confusion counts are
calculated by the existing evaluation logic. Transaction previews are also
bounded to keep driver memory and notebook output manageable.

### Install and launch on Windows

Use the same Python environment that has the project dependencies installed.
From PowerShell, change to the repository root:

```powershell
Set-Location D:\Projects\Fintech-secure-aml-spark
```

If Jupyter is not already installed in that environment, install it there
without changing the project's `requirements.txt`:

```powershell
python -m pip install notebook ipykernel
```

Register that Python environment as a notebook kernel (one-time per
environment), then launch JupyterLab from the repository root:

```powershell
python -m ipykernel install --user --name fintech-aml-spark --display-name "Python (Fintech AML Spark)"
python -m jupyter lab
```

Open `notebooks\01_aml_spark_workflow.ipynb` and select the **Python (Fintech
AML Spark)** kernel. The selected kernel must be the same environment in which
PySpark and the other project dependencies are installed. The notebook uses
the same local Spark settings as `main.py` and does not change Python, Java,
Spark, or PySpark versions.

The notebook locates the project by searching the current working directory
and its parents for `main.py`; launching Jupyter from the repository root
ensures imports and data paths resolve as documented. If the prepared sample
is missing, provide `data\HI-Small_Trans.csv` and set `RUN_SAMPLING = True` in
the configuration cell. `RUN_MODEL_TRAINING` and `RUN_SQL_QUERIES` default to
`True`, so their live outputs appear when those cells run. Set either to
`False` to skip that stage. Training uses local CPU, memory, and temporary
disk. Run the final notebook cell to release cached transactions and stop
Spark after exploration.

### Regenerate the project report

From the repository root, run:

```powershell
python reports\generate_report.py
```

This regenerates the editable DOCX and PDF under `reports\` using only the
notebook's code cells and embedded outputs. It does not execute the notebook;
rerun the notebook first if the report should reflect new results.

The first notebook cell adds the discovered project root to Python's import
path before importing the project modules. If the notebook is opened from a
location outside the project tree (for example, a copied notebook attachment),
set `AML_SPARK_PROJECT_DIR` to the repository root before launching Jupyter:

```powershell
$env:AML_SPARK_PROJECT_DIR = "D:\Projects\Fintech-secure-aml-spark"
python -m jupyter lab
```

## Testing

Run the generated-data tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover exact and reproducible sampling, preservation of the input,
cleaning and account-ID construction, feature behavior, SQL query execution,
threshold and metric calculations, the three-model pipeline, plot generation,
and the interactive-backend error.

## Limitations and next steps

This project is an educational prototype. It does not include streaming
ingestion, a deployed service or API, model persistence, probability
calibration, hyperparameter search, real-bank data validation, case management,
human review, production monitoring, or a regulatory decision workflow. The
three-feature design omits behavioral history, customer context, and account
network structure. Simulated-data evaluation cannot establish real-world
effectiveness.

Reasonable next steps for a production-oriented research effort would include
validating on appropriately governed real data, building richer temporal and
network features, establishing simple baselines, testing stability across
multiple time windows, calibrating and monitoring scores, and defining alert
thresholds with investigators and compliance stakeholders. Any operational
use would also require appropriate governance, privacy, security, and human
review.

## Interview-ready explanation

> “I built a local PySpark batch pipeline around IBM's simulated AMLSim
> HI-Small transaction data. It creates an exact-size, reproducible,
> class-stratified sample, validates and prepares the transaction columns,
> and adds a simple amount-range indicator. I use Spark SQL for transaction
> summaries and pattern queries, then compare a decision tree, random forest,
> and gradient-boosted tree. To better approximate future use, I split the data
> chronologically into training, validation, and test periods. Each model
> chooses its classification threshold on validation F1, and I report
> precision, recall, F1, PR-AUC, ROC-AUC, and confusion counts on the future
> test period. The key lesson is that AML labels are rare: accuracy and
> ROC-AUC can look good even when precision is low, so the alert trade-off
> matters. This is a prototype on simulated data, not a production AML
> decision system.”

### Common interview follow-ups

**Why use a time-based split instead of a random split?**
A time-based split better approximates training on past transactions and
evaluating on later ones. It also makes changes in class prevalence visible
instead of distributing them away through random stratification.

**Why select a threshold instead of always using 0.5?**
The default cutoff may not fit a rare-positive task or the desired balance
between missed cases and false alerts. This project chooses each model's
threshold using validation F1, then measures the result on a separate test
period.

**Which model is best?**
There is no universal answer. In the saved example, the random forest has the
best F1, precision, and PR-AUC, while the decision tree has higher recall.
The right operating point depends on the cost of false positives and false
negatives, as well as investigation capacity.

**What would you improve?**
I would add richer behavioral and graph-based features, establish baselines,
evaluate across multiple time windows and governed real datasets, and
investigate threshold calibration and alert workload. I would also add
appropriate monitoring and a human-review process before considering any
operational use.
