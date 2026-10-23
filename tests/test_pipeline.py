import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from data_processing import add_features, create_stratified_sample, load_and_clean
from ml_models import (
    _best_validation_threshold,
    _classification_metrics,
    train_and_evaluate_models,
)
from spark_sql import build_queries, run_sql_queries
from visualizations import (
    PLOT_TITLES,
    create_visualizations,
    display_saved_visualizations,
    require_interactive_backend,
)


def _write_ibm_transactions(path: Path, row_count: int = 120) -> None:
    header = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    start = datetime(2022, 9, 1)
    payment_types = ["Wire", "Cheque", "ACH"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for index in range(row_count):
            sender = index % 6
            writer.writerow(
                [
                    (start + timedelta(minutes=index)).strftime("%Y/%m/%d %H:%M"),
                    1,
                    1000 + sender,
                    2,
                    2000 + index % 9,
                    100 + index,
                    "US Dollar",
                    9_500 if index % 13 == 0 else 100 + index,
                    "US Dollar",
                    payment_types[index % len(payment_types)],
                    int(index % 7 == 0),
                ]
            )


class AmlPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("SimpleAmlPipelineTests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_interactive_backend_error_is_actionable(self) -> None:
        with patch("visualizations.matplotlib.get_backend", return_value="Agg"):
            with self.assertRaisesRegex(RuntimeError, "TkAgg.*QtAgg"):
                require_interactive_backend()

    def test_stratified_sample_is_exact_reproducible_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "HI-Small_Trans.csv"
            first_sample = Path(directory) / "sample_a.csv"
            second_sample = Path(directory) / "sample_b.csv"
            _write_ibm_transactions(source)
            original_bytes = source.read_bytes()

            create_stratified_sample(
                self.spark, str(source), str(first_sample), sample_size=60, seed=42
            )
            create_stratified_sample(
                self.spark, str(source), str(second_sample), sample_size=60, seed=42
            )

            first = self.spark.read.option("header", "true").csv(str(first_sample))
            positives = first.filter("`Is Laundering` = '1'").count()
            self.assertEqual(first.count(), 60)
            self.assertEqual(positives, 9)
            self.assertEqual(first_sample.read_bytes(), second_sample.read_bytes())
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_cleaning_builds_bank_account_ids_and_no_label_feature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "HI-Small_Trans.csv"
            _write_ibm_transactions(source, row_count=10)
            transactions = add_features(load_and_clean(self.spark, str(source)))
            first = transactions.orderBy("timestamp").first()

            self.assertEqual(transactions.count(), 10)
            self.assertEqual(first["sender_account"], "1:1000")
            self.assertEqual(first["receiver_account"], "2:2000")
            self.assertEqual(first["is_near_threshold"], 1)
            self.assertNotIn("is_laundering", ["amount", "is_near_threshold"])

    def test_all_seven_sql_queries_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "HI-Small_Trans.csv"
            _write_ibm_transactions(source)
            transactions = add_features(load_and_clean(self.spark, str(source)))

            queries = run_sql_queries(self.spark, transactions)
            self.assertEqual(len(build_queries()), 7)
            self.assertEqual(len(queries), 7)
            self.assertEqual(queries["transaction_statistics"].count(), 1)
            self.assertEqual(
                queries["transaction_statistics"].first()["transaction_count"], 120
            )
            self.assertGreater(queries["sender_account_join"].count(), 0)
            self.assertGreater(queries["wildcard_payment_search"].count(), 0)

    def test_metrics_and_all_three_models_and_plots(self) -> None:
        self.assertEqual(
            _best_validation_threshold(
                np.array([1, 0, 0, 0]),
                np.array([0.9, 0.4, 0.8, 0.1]),
            ),
            0.9,
        )
        sample_predictions = self.spark.createDataFrame(
            [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
            ["label", "prediction"],
        )
        self.assertEqual(_classification_metrics(sample_predictions)["tp"], 1)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "HI-Small_Trans.csv"
            output = Path(directory) / "outputs"
            _write_ibm_transactions(source)
            transactions = add_features(load_and_clean(self.spark, str(source)))

            metrics = train_and_evaluate_models(transactions, str(output))
            create_visualizations(
                transactions, metrics, str(output), show_interactive=False
            )

            self.assertEqual(
                {row["model"] for row in metrics},
                {
                    "DecisionTreeClassifier",
                    "RandomForestClassifier",
                    "GBTClassifier",
                },
            )
            for row in metrics:
                self.assertTrue(
                    {
                        "precision",
                        "recall",
                        "f1",
                        "pr_auc",
                        "roc_auc",
                        "accuracy",
                        "threshold",
                        "rows_evaluated",
                        "tp",
                        "tn",
                        "fp",
                        "fn",
                    }.issubset(row)
                )
            for plot in (
                "aml_class_distribution.png",
                "correlation_heatmap.png",
                "model_comparison.png",
                "confusion_matrices.png",
                "precision_recall_curves.png",
                "roc_curves.png",
                "top10_feature_importance.png",
                "precision_recall_f1_comparison.png",
            ):
                self.assertTrue((output / plot).is_file(), plot)
            self.assertTrue((output / "model_metrics.csv").is_file())
            with (output / "model_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as metrics_file:
                saved_metrics = list(csv.DictReader(metrics_file))
            self.assertEqual(len(saved_metrics), 3)
            self.assertEqual(
                {row["Model"] for row in saved_metrics},
                {
                    "DecisionTreeClassifier",
                    "RandomForestClassifier",
                    "GBTClassifier",
                },
            )
            displayed_titles = []
            with patch(
                "visualizations.plt.show",
                side_effect=lambda **kwargs: displayed_titles.append(
                    plt.gcf().axes[0].get_title()
                ),
            ) as show, patch("visualizations.plt.close") as close:
                displayed_paths = display_saved_visualizations(str(output))
            self.assertEqual(
                [path.name for path in displayed_paths],
                [
                    "aml_class_distribution.png",
                    "correlation_heatmap.png",
                    "precision_recall_curves.png",
                    "roc_curves.png",
                    "top10_feature_importance.png",
                    "model_comparison.png",
                    "precision_recall_f1_comparison.png",
                    "confusion_matrices.png",
                ],
            )
            self.assertEqual(displayed_titles, PLOT_TITLES)
            self.assertEqual(show.call_count, 8)
            show.assert_has_calls([call(block=True)] * 8)
            self.assertEqual(close.call_count, 8)


if __name__ == "__main__":
    unittest.main()
