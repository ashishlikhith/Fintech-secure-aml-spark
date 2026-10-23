import argparse
import csv
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel

from src.data_pipeline import run_ibm_data_engineering
from src.engine_init import get_spark_session
from src.feature_engineering import process_ml_features
from src.model_pipeline import train_and_evaluate_models
from src.sql_analytics import run_sql_analytics
from src.visualization import create_pipeline_visualizations


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _resolve_transaction_file(value: str) -> str:
    dataset_path = Path(_project_path(value))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Transaction CSV file does not exist: {dataset_path}")
    if dataset_path.name.casefold() != "hi-small_trans.csv":
        raise ValueError(
            "This run is configured for the HI-Small_Trans.csv dataset only."
        )
    return str(dataset_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AML analysis and three classifiers on the IBM HI-Small CSV."
    )
    parser.add_argument(
        "--data-path",
        default="data/HI-Small_Trans.csv",
        help="Path to HI-Small_Trans.csv (relative paths are based at the project root).",
    )
    parser.add_argument("--output-dir", default="artifacts/hi-small")
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=10000.0,
        help="Illustrative amount used for educational near-threshold signals.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Transaction lookback window used for account-level signals.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be greater than zero")
    if args.near_threshold <= 0:
        raise ValueError("--near-threshold must be greater than zero")

    transaction_file = _resolve_transaction_file(args.data_path)
    output_dir = Path(_project_path(args.output_dir))
    spark = get_spark_session()
    model_results = None
    base_df = None
    try:
        print(f"Using IBM AML transaction file: {transaction_file}")
        base_df = run_ibm_data_engineering(
            spark,
            transaction_file,
            near_threshold=args.near_threshold,
            lookback_seconds=args.lookback_hours * 3600,
        ).persist(StorageLevel.DISK_ONLY)
        transaction_count = base_df.count()
        if not transaction_count:
            raise ValueError("HI-Small input contains no valid transactions.")
        print(f"Loaded {transaction_count:,} valid transactions.")

        ml_df = process_ml_features(base_df, output_dir=str(output_dir))
        model_results = train_and_evaluate_models(ml_df, output_dir=str(output_dir))
        if set(model_results["models"]) != {
            "Decision Tree",
            "Random Forest",
            "Gradient-Boosted Trees",
        }:
            raise RuntimeError("The pipeline did not complete exactly three classifiers.")

        create_pipeline_visualizations(base_df, model_results, str(output_dir))
        forest_predictions = model_results["predictions"]["Random Forest"]
        forest_predictions.select(
            "transaction_id", "label", "prediction"
        ).createOrReplaceTempView("model_predictions")

        sql_output_dir = output_dir / "sql"
        sql_output_dir.mkdir(parents=True, exist_ok=True)
        print("\nSpark SQL AML analytics:")
        for name, result in run_sql_analytics(
            spark, near_threshold=args.near_threshold
        ).items():
            query_started = perf_counter()
            rows = result.limit(20).collect()
            print(f"\n{name}:")
            if rows:
                print(" | ".join(result.columns))
                for row in rows:
                    print(" | ".join(str(value) for value in row))
            else:
                print("(no rows)")
            with (sql_output_dir / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                writer = csv.writer(destination)
                writer.writerow(result.columns)
                writer.writerows(tuple(row) for row in rows)
            print(f"Query/display/save time: {perf_counter() - query_started:.2f}s")
    finally:
        if model_results is not None:
            for predictions in model_results["predictions"].values():
                predictions.unpersist()
        if base_df is not None:
            base_df.unpersist()
        spark.stop()

    print(
        f"\nPipeline completed. Outputs saved under {output_dir}. "
        "The IBM AMLSim benchmark is simulated data; "
        "held-out predictions are not validated on live banking activity. "
        "Risk signals are not proof of money laundering."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
