"""Time series data loader for spacecraft thruster prediction."""

from pathlib import Path

import pandas as pd
from loguru import logger


class TimeSeriesLoader:
    """Loads time series CSV files and metadata"""

    def __init__(self, metadata_path: Path, train_dir: Path, test_dir: Path):
        self.metadata_path = metadata_path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.metadata = None

    def load_metadata(self) -> pd.DataFrame:
        """Load and prepare metadata with train/test split"""
        logger.info(f"Loading metadata from {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)

        self.metadata["split"] = self.metadata["sn"].apply(
            lambda sn: "train" if sn <= 12 else "test"
        )

        train_count = (self.metadata["split"] == "train").sum()
        test_count = (self.metadata["split"] == "test").sum()

        logger.info(f"Train tests: {train_count} (SN01-12)")
        logger.info(f"Test tests: {test_count} (SN13-24)")

        return self.metadata

    def load_time_series(self, filename: str, split: str) -> pd.DataFrame:
        """Load a single time series CSV file"""
        data_dir = self.train_dir if split == "train" else self.test_dir
        filepath = data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})

        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

        return df
