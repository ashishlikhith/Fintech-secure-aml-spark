import os
from pathlib import Path

from pyspark.sql import SparkSession


def _read_spark_config() -> dict[str, str]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "spark_config.conf"
    config = {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Spark configuration file not found: {config_path}")
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid Spark configuration at {config_path}:{line_number}"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(
                f"Invalid Spark configuration at {config_path}:{line_number}"
            )
        config[key] = value
    return config


def get_spark_session() -> SparkSession:
    """Create the local-by-default Spark session, with environment overrides."""
    config = _read_spark_config()
    config["spark.master"] = os.environ.get(
        "SPARK_MASTER", config.get("spark.master", "local[*]")
    )
    config["spark.sql.shuffle.partitions"] = os.environ.get(
        "SPARK_SQL_SHUFFLE_PARTITIONS",
        config.get("spark.sql.shuffle.partitions", "8"),
    )
    config["spark.default.parallelism"] = os.environ.get(
        "SPARK_DEFAULT_PARALLELISM",
        config.get("spark.default.parallelism", "8"),
    )
    for environment_name, spark_property in (
        ("SPARK_DRIVER_MEMORY", "spark.driver.memory"),
        ("SPARK_EXECUTOR_MEMORY", "spark.executor.memory"),
    ):
        memory_value = os.environ.get(environment_name)
        if memory_value:
            config[spark_property] = memory_value
    builder = SparkSession.builder.appName("Fintech-Secure-AML-Spark")
    for key, value in config.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()
