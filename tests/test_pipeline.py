import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

from src.data_pipeline import run_data_engineering, run_ibm_data_engineering
from src.feature_engineering import process_ml_features
from src.model_pipeline import train_and_evaluate_models
from src.sql_analytics import run_sql_analytics
from src.visualization import create_pipeline_visualizations
from run_pipeline import _resolve_ibm_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        cls.base_df = run_data_engineering(
            cls.spark,
            str(PROJECT_ROOT / "data" / "mock_users.txt"),
            str(PROJECT_ROOT / "data" / "aml_transactions.csv"),
        ).cache()
        cls.base_df.count()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.base_df.unpersist()
        cls.spark.stop()

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

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sender_account"], "1:1001")
        self.assertEqual(rows[0]["receiver_account"], "2:2001")
        self.assertEqual(rows[0]["payment_type"], "ACH")
        self.assertEqual(rows[0]["payment_currency"], "US Dollar")
        self.assertEqual(rows[0]["is_laundering"], 0)
        self.assertEqual(rows[1]["is_laundering"], 1)

    def test_ibm_directory_selects_requested_or_largest_transaction_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory)
            small = data_directory / "HI-Small_Trans.csv"
            large = data_directory / "HI-Large_Trans.csv"
            small.write_text("small\n", encoding="utf-8")
            large.write_text("large\n" * 10, encoding="utf-8")

            self.assertEqual(
                _resolve_ibm_transactions(str(data_directory), "large"),
                str(large.resolve()),
            )
            self.assertEqual(
                _resolve_ibm_transactions(str(data_directory), "largest"),
                str(large.resolve()),
            )

    def test_reusable_sql_queries_and_optional_model_view(self) -> None:
        results = run_sql_analytics(self.spark)
        self.assertEqual(results["transaction_totals"].count(), 1)
        self.assertEqual(results["highest_value_transactions"].count(), 20)
        self.assertGreater(results["payment_type_wildcard_search"].count(), 0)
        self.assertGreater(results["shared_contact_information"].count(), 0)
        self.assertGreater(results["repeated_near_threshold_accounts"].count(), 0)
        self.assertIn("duplicate_transaction_ids", results)
        self.assertEqual(results["duplicate_transaction_ids"].count(), 0)

        self.spark.createDataFrame(
            [("T0001", 0.0, 0.0)], ["transaction_id", "label", "prediction"]
        ).createOrReplaceTempView("model_predictions")
        prediction_results = run_sql_analytics(self.spark)
        self.assertIn("model_label_comparison", prediction_results)
        self.assertEqual(prediction_results["model_label_comparison"].count(), 1)

    def test_feature_pipeline_models_and_visualizations(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            ml_df = process_ml_features(
                self.spark, self.base_df, output_dir=output_directory
            )
            self.assertTrue((Path(output_directory) / "correlation_heatmap.png").is_file())
            results = train_and_evaluate_models(ml_df)
            create_pipeline_visualizations(self.base_df, results, output_directory)

            for artifact in (
                "transaction_overview.png",
                "model_comparison.png",
                "confusion_matrices.png",
                "roc_precision_recall.png",
                "transaction_flow_map.png",
                "model_metrics.csv",
            ):
                self.assertTrue((Path(output_directory) / artifact).is_file())
        expected_models = {
            "Decision Tree",
            "Random Forest",
            "K-means",
        }
        self.assertTrue(expected_models.issubset(results["models"]))
        comparison = results["comparison"]
        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison.filter("task = 'binary classification'").count(), 2)
        self.assertTrue(
            {"area_under_pr", "area_under_roc", "positive_rate"}.issubset(
                comparison.columns
            )
        )
        self.assertEqual(
            results["predictions"]["Random Forest"]
            .select("label")
            .distinct()
            .count(),
            2,
        )

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
            self.assertEqual(
                run_sql_analytics(self.spark)["duplicate_transaction_ids"].count(),
                1,
            )
            missing = run_sql_analytics(self.spark)["missing_data_summary"].first()
            self.assertEqual(missing["missing_amounts"], 1)
            self.assertEqual(missing["missing_timestamps"], 0)

    def test_z_missing_and_empty_inputs_fail_explicitly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_data_engineering(
                self.spark,
                "missing-customer-file.txt",
                str(PROJECT_ROOT / "data" / "aml_transactions.csv"),
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
