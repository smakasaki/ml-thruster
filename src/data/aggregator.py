from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
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
    avg_pulse_duration: float
    max_pulse_duration: float
    min_pulse_duration: float
    duty_cycle: float

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
            "avg_pulse_duration": self.avg_pulse_duration,
            "max_pulse_duration": self.max_pulse_duration,
            "min_pulse_duration": self.min_pulse_duration,
            "duty_cycle": self.duty_cycle,
        }


class DataAggregator:
    def aggregate_time_series(self, time_series_df: pd.DataFrame) -> dict:
        aggregates = {}

        if len(time_series_df) == 0:
            logger.warning("Empty DataFrame - returning default aggregates")
            return self._get_default_aggregates()

        if TON_COLUMN not in time_series_df.columns:
            logger.warning("No ton column found - cannot filter active periods")
            return self._get_default_aggregates()

        active_mask = time_series_df[TON_COLUMN] == 1
        active_df = time_series_df[active_mask]

        if len(active_df) > 0:
            aggregates.update(
                self._aggregate_column_statistics(
                    active_df, THRUST_COLUMN, self._get_default_column_aggregates
                )
            )
            aggregates.update(
                self._aggregate_column_statistics(
                    active_df, MFR_COLUMN, self._get_default_column_aggregates
                )
            )

            aggregates["on_time_test"] = len(active_df) / config.SAMPLING_RATE_HZ
        else:
            logger.debug("No active periods (ton=1) found in sequence")
            aggregates.update(self._get_default_column_aggregates("thrust"))
            aggregates.update(self._get_default_column_aggregates("mfr"))
            aggregates["on_time_test"] = DEFAULT_FLOAT_VALUE

        ton_series = cast(pd.Series, time_series_df[TON_COLUMN])
        aggregates["pulses_count"] = self._count_pulses(ton_series)

        ton_features = self._extract_ton_features(ton_series, config.SAMPLING_RATE_HZ)
        aggregates.update(ton_features)

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
        return {f"{column_prefix}_{suffix}": np.nan for suffix in STATS_SUFFIXES}

    def _get_default_aggregates(self) -> dict:
        aggregates = {}
        aggregates.update(self._get_default_column_aggregates("thrust"))
        aggregates.update(self._get_default_column_aggregates("mfr"))
        aggregates["on_time_test"] = np.nan  # ← CHANGED
        aggregates["pulses_count"] = 0  # Keep as 0 (count can be zero)
        aggregates["duration"] = np.nan  # ← CHANGED
        aggregates["anomaly_duration"] = np.nan  # ← CHANGED
        aggregates["avg_pulse_duration"] = np.nan  # ← CHANGED
        aggregates["max_pulse_duration"] = np.nan  # ← CHANGED
        aggregates["min_pulse_duration"] = np.nan  # ← CHANGED
        aggregates["duty_cycle"] = np.nan  # ← CHANGED
        return aggregates

    def _count_pulses(self, ton_series: pd.Series) -> int:
        state_changes = ton_series.diff()
        rising_edges = (state_changes == 1).sum()
        return int(rising_edges)

    def _extract_ton_features(self, ton_series: pd.Series, sampling_rate: float) -> dict:
        if len(ton_series) == 0:
            return {
                "avg_pulse_duration": np.nan,
                "max_pulse_duration": np.nan,
                "min_pulse_duration": np.nan,
                "duty_cycle": np.nan,
            }

        duty_cycle = (ton_series == 1).mean()

        state_changes = ton_series.diff()
        rising_edges_mask = state_changes == 1
        falling_edges_mask = state_changes == -1

        rising_edges = ton_series.index[rising_edges_mask].tolist()
        falling_edges = ton_series.index[falling_edges_mask].tolist()

        if len(ton_series) > 0 and ton_series.iloc[0] == 1:
            rising_edges.insert(0, ton_series.index[0])

        if len(ton_series) > 0 and ton_series.iloc[-1] == 1:
            falling_edges.append(ton_series.index[-1])

        pulse_durations = []
        min_length = min(len(rising_edges), len(falling_edges))
        for i in range(min_length):
            start = rising_edges[i]
            end = falling_edges[i]
            if end > start:
                duration_seconds = (end - start) / sampling_rate
                pulse_durations.append(duration_seconds)

        if len(pulse_durations) == 0:
            return {
                "avg_pulse_duration": np.nan,
                "max_pulse_duration": np.nan,
                "min_pulse_duration": np.nan,
                "duty_cycle": duty_cycle,  # Keep as-is
            }

        return {
            "avg_pulse_duration": float(np.mean(pulse_durations)),
            "max_pulse_duration": float(np.max(pulse_durations)),
            "min_pulse_duration": float(np.min(pulse_durations)),
            "duty_cycle": duty_cycle,
        }

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
