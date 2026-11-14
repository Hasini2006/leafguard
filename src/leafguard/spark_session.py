"""Build a local Spark session sized for a 4-core, 10 GB virtual machine."""

from __future__ import annotations

import os
import sys
from typing import Any

from pyspark.sql import SparkSession


def make_spark(app_name: str, config: dict[str, Any]) -> SparkSession:
    """Start (or reuse) a SparkSession using the 'spark' section of params.yaml.

    Spark's Python workers are told to use the same interpreter as the driver, so they
    see the packages installed in the project's virtual environment.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .master(str(config.get("master", "local[4]")))
        .config("spark.driver.memory", str(config.get("driver_memory", "2g")))
        .config("spark.sql.shuffle.partitions", str(config.get("partitions", 8)))
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "64")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
