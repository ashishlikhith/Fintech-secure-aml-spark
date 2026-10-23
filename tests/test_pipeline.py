import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession

from ml_models import (
    _classification_metrics,
    process_ml_features,
    train_and_evaluate_models,
)
from spark_sql import run_sql_analytics
from visualizations import create_pipeline_visualizations
from main import (
    _resolve_transaction_file,
    run_data_engineering,
    run_ibm_data_engineering,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_sample_transactions(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as transaction_file:
        writer = csv.writer(transaction_file)
        writer.writerow(
            [
                "transaction_id",
                "timestamp",
                "sender_account",
                "receiver_account",
                "amount",
                "payment_type",
                "is_laundering",
            ]
        )
        start = datetime(2025, 1, 1)
        for index in range(48):
            sender_number = index % 8 + 1
            sender = f"U{sender_number:03d}"
            receiver = f"U{(sender_number % 8) + 1:03d}"
            writer.writerow(
                [
                    f"T{index + 1:04d}",
                    (start + timedelta(hours=index)).isoformat(),
                    sender,
                    receiver,
                    9500 if sender == "U002" else 100 + index,
                    "WIRE_TRANSFER" if index % 3 == 0 else "ACH",
                    int(sender in {"U002", "U004", "U007"}),
                ]
            )


class SparkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("FintechSecureAmlTests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls._test_data_directory = tempfile.TemporaryDirectory()
        cls._transactions_path = (
            Path(cls._test_data_directory.name) / "transactions.csv"
        )
        _write_sample_transactions(cls._transactions_path)
        cls.base_df = run_data_engineering(
            cls.spark,
            str(PROJECT_ROOT / "data" / "mock_users.txt"),
            str(cls._transactions_path),
        ).cache()
        cls.base_df.count()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.base_df.unpersist()
        cls.spark.stop()
        cls._test_data_directory.cleanup()

    def test_sample_data_and_identity_structuring_signals(self) -> None:
        self.assertEqual(self.base_df.count(), 48)
        self.assertEqual(self.spark.table("customers").count(), 8)
        self.assertGreater(
            self.base_df.filter(
                "sender_account = 'U002' AND near_threshold_count_24h >= 2"
            ).count(),
            0,
        )
        self.assertGreater(
            self.base_df.filter(
                "sender_account = 'U007' AND shared_contact_count > 1"
            ).count(),
            0,
        )
        self.assertTrue(
            {"risk_score", "risk_category", "risk_indicators"}.issubset(
                self.base_df.columns
            )
        )

    def test_ibm_aml_csv_maps_duplicate_account_headers_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transaction_file = Path(directory) / "HI-Small_Trans.csv"
            transaction_file.write_text(
                "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
                "Receiving Currency,Amount Paid,Payment Currency,Payment Format,"
                "Is Laundering\n"
                "2022/09/01 00:20,1,1001,2,2001,100,US Dollar,100,"
                "US Dollar,ACH,0\n"
                "2022/09/01 01:20,1,1001,3,3001,250,US Dollar,250,"
                "US Dollar,Cash,1\n",
                encoding="utf-8",
            )

            isolated_spark = self.spark.newSession()
            enriched = run_ibm_data_engineering(
                isolated_spark, str(transaction_file)
            )
            rows = enriched.select(
                "sender_account",
                "receiver_account",
                "payment_type",
                "payment_currency",
                "is_laundering",
            ).orderBy("is_laundering").collect()
            limited = run_ibm_data_engineering(
                isolated_spark, str(transaction_file), max_rows=1
            )
            limited_count = limited.count()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sender_account"], "1:1001")
        self.assertEqual(rows[0]["receiver_account"], "2:2001")
        self.assertEqual(rows[0]["payment_type"], "ACH")
        self.assertEqual(rows[0]["payment_currency"], "US Dollar")
        self.assertEqual(rows[0]["is_laundering"], 0)
        self.assertEqual(rows[1]["is_laundering"], 1)
        self.assertEqual(limited_count, 1)

    def test_pipeline_resolves_only_the_small_ibm_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory)
            small = data_directory / "HI-Small_Trans.csv"
            medium = data_directory / "HI-Medium_Trans.csv"
            small.write_text("small\n", encoding="utf-8")
            medium.write_text("medium\n", encoding="utf-8")

            self.assertEqual(_resolve_transaction_file(str(small)), str(small.resolve()))
            with self.assertRaisesRegex(ValueError, "HI-Small_Trans.csv"):
                _resolve_transaction_file(str(medium))

    def test_reusable_bounded_aml_sql_queries(self) -> None:
        results = run_sql_analytics(self.spark, transactions=self.base_df)
        self.assertEqual(results["transaction_totals"].count(), 1)
        self.assertLessEqual(results["highest_risk_transactions"].count(), 20)
        self.assertGreater(results["payment_type_summary"].count(), 0)
        self.assertGreater(results["near_threshold_summary"].count(), 0)
        self.assertGreater(results["sender_receiver_patterns"].count(), 0)
        self.assertGreater(results["like_wildcard_example"].count(), 0)
        self.assertLessEqual(results["dataframe_join_sql_example"].count(), 20)
        self.assertEqual(
            set(results),
            {
                "transaction_totals",
                "suspicious_transactions_by_risk",
                "highest_risk_transactions",
                "payment_type_summary",
                "near_threshold_summary",
                "sender_receiver_patterns",
                "like_wildcard_example",
                "dataframe_join_sql_example",
            },
        )
        self.assertEqual(self.spark.table("rdd_transactions").count(), 48)
        self.assertGreater(self.spark.table("rdd_sender_summary").count(), 0)

    def test_feature_pipeline_models_and_visualizations(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            ml_df = process_ml_features(
                self.base_df, output_dir=output_directory
            )
            self.assertTrue((Path(output_directory) / "correlation_heatmap.png").is_file())
            results = train_and_evaluate_models(
                ml_df,
                output_dir=output_directory,
            )
            create_pipeline_visualizations(self.base_df, results, output_directory)

            for artifact in (
                "transaction_overview.png",
                "model_comparison.png",
                "confusion_matrices.png",
                "feature_importance.png",
                "transaction_flow_map.png",
                "model_metrics.csv",
            ):
                self.assertTrue((Path(output_directory) / artifact).is_file())
            self.assertTrue(
                (Path(output_directory) / "metrics" / "model_metrics.csv").is_file()
            )
            self.assertTrue(
                (
                    Path(output_directory)
                    / "metrics"
                    / "evaluation_summary.json"
                ).is_file()
            )
        expected_models = {
            "Decision Tree",
            "Random Forest",
            "Gradient-Boosted Trees",
        }
        self.assertTrue(expected_models.issubset(results["models"]))
        self.assertEqual(
            set(results["models"]),
            {"Decision Tree", "Random Forest", "Gradient-Boosted Trees"},
        )
        comparison = results["comparison"]
        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison.filter("task = 'binary classification'").count(), 3)
        self.assertTrue(
            {
                "area_under_pr",
                "area_under_roc",
                "positive_rate",
                "threshold",
                "default_precision",
                "false_positives",
                "false_negatives",
            }.issubset(comparison.columns)
        )
        self.assertEqual(
            results["predictions"]["Random Forest"]
            .select("label")
            .distinct()
            .count(),
            2,
        )
        self.assertEqual(
            set(results["models"]),
            {"Decision Tree", "Random Forest", "Gradient-Boosted Trees"},
        )
        for predictions in results["predictions"].values():
            predictions.unpersist()

    def test_classification_metrics_use_laundering_as_positive_label(self) -> None:
        predictions = self.spark.createDataFrame(
            [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)],
            ["label", "prediction"],
        )
        metrics = _classification_metrics(predictions)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["positive_rate"], 2 / 3)

    def test_y_invalid_and_duplicate_transaction_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            customer_file = Path(directory) / "customers.txt"
            transaction_file = Path(directory) / "transactions.csv"
            customer_file.write_text(
                "user_id|full_name|device_id|contact_email|signup_country|account_created_at\n"
                "U001|Synthetic Customer|device-1|||2024-01-01\n",
                encoding="utf-8",
            )
            transaction_file.write_text(
                "transaction_id,timestamp,sender_account,receiver_account,"
                "amount,payment_type,is_laundering\n"
                "T0001,2025-01-01T09:00:00,U001,U001,12.00,CARD,0\n"
                "T0001,2025-01-01T09:00:00,U001,U001,12.00,CARD,0\n"
                "T0002,not-a-timestamp,U001,U001,15.00,ACH,0\n"
                "T0003,2025-01-01T11:00:00,U001,U001,,ACH,0\n",
                encoding="utf-8",
            )
            cleaned = run_data_engineering(
                self.spark, str(customer_file), str(transaction_file)
            )
            self.assertEqual(cleaned.count(), 1)
            raw_transactions = self.spark.table("transactions_raw")
            duplicate_ids = (
                raw_transactions.groupBy("transaction_id")
                .count()
                .filter("count > 1")
                .count()
            )
            self.assertEqual(duplicate_ids, 1)
            self.assertEqual(
                raw_transactions.filter("amount IS NULL").count(),
                1,
            )

    def test_model_training_fails_when_a_time_split_lacks_both_labels(self) -> None:
        data = self.spark.createDataFrame(
            [
                (datetime(2025, 1, 1), 0.0),
                (datetime(2025, 1, 2), 0.0),
                (datetime(2025, 1, 3), 0.0),
                (datetime(2025, 1, 4), 0.0),
                (datetime(2025, 1, 5), 0.0),
            ],
            ["timestamp", "label"],
        )
        with tempfile.TemporaryDirectory() as output_directory:
            with self.assertRaisesRegex(ValueError, "both labels"):
                train_and_evaluate_models(data, output_dir=output_directory)

    def test_z_missing_and_empty_inputs_fail_explicitly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_data_engineering(
                self.spark,
                "missing-customer-file.txt",
                str(self._transactions_path),
            )

        with tempfile.TemporaryDirectory() as directory:
            customer_file = Path(directory) / "customers.txt"
            transaction_file = Path(directory) / "transactions.csv"
            customer_file.write_text(
                "user_id|full_name|device_id|contact_email|signup_country|account_created_at\n"
                "U001|Synthetic Customer|device-1|contact@example.test|US|2024-01-01\n",
                encoding="utf-8",
            )
            transaction_file.write_text(
                "transaction_id,timestamp,sender_account,receiver_account,"
                "amount,payment_type,is_laundering\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no valid transaction"):
                run_data_engineering(
                    self.spark, str(customer_file), str(transaction_file)
                )


if __name__ == "__main__":
    unittest.main()
