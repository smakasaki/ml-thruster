from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from loguru import logger

from ..common import config

STATS_SUFFIXES = ["mean", "std", "min", "max", "median"]
DEFAULT_FLOAT_VALUE = 0.0
DEFAULT_INT_VALUE = 0

TON_COLUMN = "ton"
TIMESTAMP_COLUMN = "timestamp"
ANOMALY_CODE_COLUMN = "anomaly_code"
THRUST_COLUMN = "thrust"
MFR_COLUMN = "mfr"

CONCAT_AXIS_COLUMNS = 1


@dataclass
class TimeSeriesAggregates:
    thrust_mean: float
    thrust_std: float
    thrust_min: float
    thrust_max: float
    thrust_median: float
    mfr_mean: float
    mfr_std: float
    mfr_min: float
    mfr_max: float
    mfr_median: float
    on_time_test: float
    pulses_count: int
    duration: float
    anomaly_duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "thrust_mean": self.thrust_mean,
            "thrust_std": self.thrust_std,
            "thrust_min": self.thrust_min,
            "thrust_max": self.thrust_max,
            "thrust_median": self.thrust_median,
            "mfr_mean": self.mfr_mean,
            "mfr_std": self.mfr_std,
            "mfr_min": self.mfr_min,
            "mfr_max": self.mfr_max,
            "mfr_median": self.mfr_median,
            "on_time_test": self.on_time_test,
            "pulses_count": self.pulses_count,
            "duration": self.duration,
            "anomaly_duration": self.anomaly_duration,
        }


class DataAggregator:
    def aggregate_time_series(self, time_series_df: pd.DataFrame) -> dict:
        aggregates = {}

        if len(time_series_df) == 0:
            logger.warning("Empty DataFrame - returning default aggregates")
            return self._get_default_aggregates()

        aggregates.update(
            self._aggregate_column_statistics(
                time_series_df, THRUST_COLUMN, self._get_default_column_aggregates
            )
        )
        aggregates.update(
            self._aggregate_column_statistics(
                time_series_df, MFR_COLUMN, self._get_default_column_aggregates
            )
        )

        if TON_COLUMN in time_series_df.columns:
            is_thruster_active = time_series_df[TON_COLUMN] == 1
            thruster_on_rows = time_series_df[is_thruster_active]
            has_active_thrust_data = len(thruster_on_rows) > 0
            aggregates["on_time_test"] = (
                len(thruster_on_rows) / config.SAMPLING_RATE_HZ
                if has_active_thrust_data
                else DEFAULT_FLOAT_VALUE
            )
            ton_series = cast(pd.Series, time_series_df[TON_COLUMN])
            aggregates["pulses_count"] = self._count_pulses(ton_series)
        else:
            aggregates["on_time_test"] = DEFAULT_FLOAT_VALUE
            aggregates["pulses_count"] = DEFAULT_INT_VALUE

        if TIMESTAMP_COLUMN in time_series_df.columns and len(time_series_df) > 0:
            aggregates["duration"] = (
                time_series_df[TIMESTAMP_COLUMN].max() - time_series_df[TIMESTAMP_COLUMN].min()
            )
        else:
            aggregates["duration"] = DEFAULT_FLOAT_VALUE

        if ANOMALY_CODE_COLUMN in time_series_df.columns:
            aggregates["anomaly_duration"] = (
                time_series_df[ANOMALY_CODE_COLUMN] > 0
            ).sum() / config.SAMPLING_RATE_HZ
        else:
            aggregates["anomaly_duration"] = DEFAULT_FLOAT_VALUE

        return aggregates

    def _aggregate_column_statistics(
        self, df: pd.DataFrame, column_name: str, default_method: Callable[[str], dict[str, Any]]
    ) -> dict[str, Any]:
        if column_name not in df.columns:
            logger.debug(f"No valid {column_name} data")
            return default_method(column_name)

        column_data = cast(pd.Series, df[column_name])
        has_valid_data = bool(column_data.notna().any())
        if not has_valid_data:
            logger.debug(f"No valid {column_name} data")
            return default_method(column_name)

        return {
            f"{column_name}_mean": column_data.mean(),
            f"{column_name}_std": column_data.std(),
            f"{column_name}_min": column_data.min(),
            f"{column_name}_max": column_data.max(),
            f"{column_name}_median": column_data.median(),
        }

    def _get_default_column_aggregates(self, column_prefix: str) -> dict:
        return {f"{column_prefix}_{suffix}": DEFAULT_FLOAT_VALUE for suffix in STATS_SUFFIXES}

    def _get_default_aggregates(self) -> dict:
        aggregates = {}
        aggregates.update(self._get_default_column_aggregates("thrust"))
        aggregates.update(self._get_default_column_aggregates("mfr"))
        aggregates["on_time_test"] = DEFAULT_FLOAT_VALUE
        aggregates["pulses_count"] = DEFAULT_INT_VALUE
        aggregates["duration"] = DEFAULT_FLOAT_VALUE
        aggregates["anomaly_duration"] = DEFAULT_FLOAT_VALUE
        return aggregates

    def _count_pulses(self, ton_series: pd.Series) -> int:
        """Counts rising edges (0→1 transitions) in the thruster on/off signal to determine pulse count."""
        state_changes = ton_series.diff()
        rising_edges = (state_changes == 1).sum()
        return int(rising_edges)

    def create_aggregated_dataset(
        self, metadata: pd.DataFrame, time_series_aggregates: list[dict]
    ) -> pd.DataFrame:
        logger.info("Creating aggregated dataset")

        if len(metadata) != len(time_series_aggregates):
            msg = (
                f"Metadata length ({len(metadata)}) does not match "
                f"aggregates length ({len(time_series_aggregates)})"
            )
            raise ValueError(msg)

        aggregates_df = pd.DataFrame(time_series_aggregates)
        combined = pd.concat(
            [metadata.reset_index(drop=True), aggregates_df], axis=CONCAT_AXIS_COLUMNS
        )

        logger.info(f"Created aggregated dataset with {len(combined)} tests")
        return combined
