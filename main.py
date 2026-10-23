"""Run the complete 500,000-row IBM AMLSim PySpark project."""

import argparse
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from data_processing import add_features, create_stratified_sample, load_and_clean
from ml_models import print_data_usage_report, train_and_evaluate_models
from spark_sql import run_sql_queries
from visualizations import (
    create_visualizations,
    display_saved_visualizations,
    require_interactive_backend,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "HI-Small_Trans.csv"
SAMPLE_PATH = PROJECT_DIR / "data" / "HI-Small_Trans_sample_500000.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "hi-small-500k"
SAMPLE_SIZE = 500_000
SAMPLE_SEED = 42


def main() -> None:
    """Sample, prepare, analyze, model, plot, and release Spark resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-saved-plots",
        action="store_true",
        help="Display existing charts in order without rerunning Spark training.",
    )
    args = parser.parse_args()
    if args.show_saved_plots:
        display_saved_visualizations(str(OUTPUT_DIR))
        return

    require_interactive_backend()
    run_started = perf_counter()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Place the original IBM dataset at {DATA_PATH} before running this project."
        )

    spark = (
        SparkSession.builder.appName("Simple IBM AML Analysis")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "2")
        .getOrCreate()
    )
    transactions = None
    model_metrics = None
    source_stats: dict[str, int] = {}
    plot_paths = []
    try:
        sample_path = create_stratified_sample(
            spark,
            str(DATA_PATH),
            str(SAMPLE_PATH),
            SAMPLE_SIZE,
            SAMPLE_SEED,
            source_stats,
        )
        transactions = add_features(
            load_and_clean(spark, sample_path)
        ).persist(StorageLevel.DISK_ONLY)
        row_count = transactions.count()
        if row_count != SAMPLE_SIZE:
            raise ValueError(
                f"Expected {SAMPLE_SIZE:,} valid sample rows, got {row_count:,}."
            )
        print(f"Prepared {row_count:,} clean transactions from {sample_path}.")
        sample_counts = {
            int(row["is_laundering"]): int(row["count"])
            for row in transactions.groupBy("is_laundering").count().collect()
        }
        transactions.select(
            "timestamp",
            "sender_account",
            "receiver_account",
            "amount",
            "payment_type",
            "is_laundering",
        ).show(5, truncate=False)

        model_metrics = train_and_evaluate_models(
            transactions,
            str(OUTPUT_DIR),
            source_stats=source_stats,
            sample_size=SAMPLE_SIZE,
            sample_seed=SAMPLE_SEED,
        )
        plot_paths = create_visualizations(
            transactions,
            model_metrics,
            str(OUTPUT_DIR),
            show_interactive=True,
        )

        sql_output = OUTPUT_DIR / "sql"
        sql_output.mkdir(parents=True, exist_ok=True)
        for name, result in run_sql_queries(spark, transactions).items():
            print(f"\nSQL query: {name}")
            print(f"Result row count: {result.count():,}")
            result.show(10, truncate=False)
            result.toPandas().to_csv(
                sql_output / f"{name}.csv", index=False, encoding="utf-8"
            )
            print(f"Saved SQL result: {sql_output / f'{name}.csv'}")
    finally:
        if transactions is not None:
            transactions.unpersist()
        spark.stop()

    if model_metrics is not None:
        print_data_usage_report(
            source_stats,
            SAMPLE_SIZE,
            SAMPLE_SEED,
            {float(label): count for label, count in sample_counts.items()},
            model_metrics[0]["_split_stats"],
            model_metrics=model_metrics,
            total_runtime_seconds=perf_counter() - run_started,
        )
        print(f"\nProject complete. Output directory: {OUTPUT_DIR}")
        print(f"Metrics CSV: {OUTPUT_DIR / 'model_metrics.csv'}")
        for plot_path in plot_paths:
            print(f"Visualization: {plot_path}")


if __name__ == "__main__":
    main()
