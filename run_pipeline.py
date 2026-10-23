import argparse
import os
from pathlib import Path
from time import perf_counter

from src.data_pipeline import run_data_engineering, run_ibm_data_engineering
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


def _resolve_ibm_transactions(value: str | None, size: str) -> str:
    if value:
        dataset_path = Path(_project_path(value))
    else:
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "The IBM dataset download requires kagglehub. Install the project "
                "requirements or pass --ibm-data-path to a local CSV/folder."
            ) from error
        dataset_path = Path(
            kagglehub.dataset_download(
                "ealtman2019/ibm-transactions-for-anti-money-laundering"
            )
        )

    if dataset_path.is_file():
        return str(dataset_path.resolve())
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"IBM dataset path does not exist: {dataset_path}")

    csv_files = sorted(
        path
        for path in dataset_path.rglob("*.csv")
        if "trans" in path.stem.casefold()
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No transaction CSV was found in the IBM dataset: {dataset_path}"
        )
    if size != "largest":
        matching_files = [path for path in csv_files if size in path.stem.casefold()]
        if not matching_files:
            available = ", ".join(path.name for path in csv_files)
            raise FileNotFoundError(
                f"No IBM {size} transaction CSV found. Available files: {available}"
            )
        return str(max(matching_files, key=lambda path: path.stat().st_size).resolve())
    return str(max(csv_files, key=lambda path: path.stat().st_size).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Spark financial-crime analysis and prediction pipeline."
    )
    parser.add_argument(
        "--dataset-source",
        choices=("sample", "ibm"),
        default="ibm",
        help="Use the local educational sample or the IBM AMLSim Kaggle dataset.",
    )
    parser.add_argument(
        "--ibm-data-path",
        default=None,
        help="Optional IBM transaction CSV or downloaded dataset directory.",
    )
    parser.add_argument(
        "--ibm-size",
        choices=("small", "medium", "large", "xlarge", "largest"),
        default="large",
        help="Dataset file to use from a Kaggle directory; falls back to the largest available file.",
    )
    parser.add_argument("--users", default="data/mock_users.txt")
    parser.add_argument("--transactions", default="data/aml_transactions.csv")
    parser.add_argument("--output-dir", default="artifacts")
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
    parser.add_argument(
        "--master",
        default=None,
        help="Optional Spark master URL; defaults to local[*].",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be greater than zero")
    if args.near_threshold <= 0:
        raise ValueError("--near-threshold must be greater than zero")
    if args.master:
        os.environ["SPARK_MASTER"] = args.master

    ibm_transactions = (
        _resolve_ibm_transactions(args.ibm_data_path, args.ibm_size)
        if args.dataset_source == "ibm"
        else None
    )
    spark = get_spark_session()
    try:
        if ibm_transactions is not None:
            print(f"Using IBM AML transaction file: {ibm_transactions}")
            base_df = run_ibm_data_engineering(
                spark,
                ibm_transactions,
                near_threshold=args.near_threshold,
                lookback_seconds=args.lookback_hours * 3600,
            ).cache()
        else:
            base_df = run_data_engineering(
                spark,
                _project_path(args.users),
                _project_path(args.transactions),
                near_threshold=args.near_threshold,
                lookback_seconds=args.lookback_hours * 3600,
            ).cache()
        model_results = None
        try:
            print("\nHighest rule-based risk signals (not fraud determinations):")
            base_df.orderBy("risk_score", ascending=False).select(
                "transaction_id",
                "sender_account",
                "amount",
                "risk_score",
                "risk_category",
                "risk_indicators",
            ).show(20, truncate=False)

            ml_df = process_ml_features(
                spark,
                base_df,
                output_dir=_project_path(args.output_dir),
            )
            model_results = train_and_evaluate_models(ml_df)
            create_pipeline_visualizations(
                base_df,
                model_results,
                output_dir=_project_path(args.output_dir),
            )
            forest_predictions = model_results["predictions"].get("Random Forest")
            if forest_predictions is not None:
                forest_predictions.select(
                    "transaction_id", "label", "prediction"
                ).createOrReplaceTempView("model_predictions")

            print("\nSpark SQL fraud analytics:")
            for name, result in run_sql_analytics(
                spark, near_threshold=args.near_threshold
            ).items():
                print(f"\n{name}:")
                query_started = perf_counter()
                result.show(20, truncate=False)
                print(f"Query/display time: {perf_counter() - query_started:.2f}s")
        finally:
            if model_results is not None:
                for predictions in model_results["predictions"].values():
                    predictions.unpersist()
            base_df.unpersist()
    finally:
        spark.stop()

    if args.dataset_source == "ibm":
        print(
            "\nPipeline completed. The IBM AMLSim benchmark is simulated data; "
            "held-out predictions are not validated on live banking activity. "
            "Risk signals are not proof of money laundering."
        )
    else:
        print(
            "\nPipeline completed. Inputs are synthetic educational data; "
            "risk signals are not proof of fraud or money laundering."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
