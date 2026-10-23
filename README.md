# Simple PySpark AML Project

This project uses the existing IBM AMLSim **HI-Small** transaction dataset to
demonstrate Spark data preparation, Spark SQL, and three supervised
classifiers. AMLSim is simulated data; a model prediction is not proof of
money laundering or a decision for real banking activity.

## Run on Windows

Place the original `HI-Small_Trans.csv` at `data\HI-Small_Trans.csv`, then run
from the project folder:

```powershell
python main.py
```

The eight plots are saved and opened one at a time; close each chart window to
continue. To reopen the existing PNGs without rerunning Spark or model training,
use:

```powershell
python main.py --show-saved-plots
```

Interactive chart windows require a desktop Matplotlib backend. On Windows,
`TkAgg` is supported by the standard Tk installation; if another backend is
selected, the program stops with instructions instead of silently saving
without displaying the charts.

The program creates the reproducible, class-stratified 500,000-row sample at
`data\HI-Small_Trans_sample_500000.csv` with seed `42`. The original dataset is
not edited. This project does not install or change Python, Java, PySpark,
Apache Spark, or Spark Shell.

## Pipeline and evaluation

1. Reads the IBM CSV, reports the source row and AML class counts, then writes
   an exact-size sample preserving the source class proportions as closely as
   integer row counts allow.
2. Parses timestamps and amounts, removes invalid records, and creates sender
   and receiver account IDs from bank and account fields. The only engineered
   feature is an amount-range indicator for values between 9,000 and 10,000.
3. Splits by chronological timestamp quantiles into training, validation, and
   future test periods. The run reports each split's row count and AML-positive
   prevalence. Changing prevalence is retained and reported as temporal
   distribution shift rather than hidden by random stratification.
4. Uses only `amount`, `is_near_threshold`, and a payment-format index as model
   features. Timestamps, account IDs, transaction IDs, and the label are
   excluded. The payment index and class weights are fit from training data
   only.
5. Trains exactly `DecisionTreeClassifier`, `RandomForestClassifier`, and
   `GBTClassifier`. Each model selects its own probability threshold by
   validation F1, then computes its threshold metrics and confusion counts
   from its own future test predictions. PR-AUC and ROC-AUC use that model's
   own continuous test scores. Identical threshold metrics can occur naturally
   and are not altered to force differences.
6. Prints and saves one test-score table with accuracy, precision, recall,
   F1-score, PR-AUC, ROC-AUC, TP/TN/FP/FN, and the selected threshold. The
   beginning and end reports include source/sample/split class counts,
   prevalences, feature names, evaluated rows, per-model training times, and
   total runtime.
7. Executes all seven Spark SQL queries against the registered `transactions`
   Spark view. Each query's name and result row count are printed, followed by
   the actual first ten rows; all results are also saved as CSV.

The fixed local Spark settings use two worker threads and eight shuffle
partitions. Prepared transactions are cached on disk and released before Spark
shuts down.

## Output files

The run writes all outputs under `artifacts\hi-small-500k\`:

- `model_metrics.csv`
- `aml_class_distribution.png` (counts and percentages)
- `correlation_heatmap.png`
- `model_comparison.png` (accuracy, PR-AUC, and ROC-AUC)
- `confusion_matrices.png`
- `precision_recall_curves.png`
- `roc_curves.png`
- `top10_feature_importance.png`
- `precision_recall_f1_comparison.png`
- `sql\*.csv` for the seven query results

Every generated visualization and the metrics/SQL CSV paths are printed.

## Python files and tests

- `main.py` runs the stages and always stops the Spark session.
- `data_processing.py` samples the original data, cleans transaction fields,
  creates account IDs, and adds the amount feature.
- `spark_sql.py` contains and executes the seven SQL queries.
- `ml_models.py` makes chronological splits, trains the three classifiers,
  selects validation thresholds, and calculates test metrics.
- `visualizations.py` creates the AML and model evaluation plots.

Run the generated-data checks with:

```powershell
python -m unittest discover -s tests -v
```
