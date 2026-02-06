from dataclasses import dataclass
from typing import cast

import pandas as pd
from loguru import logger

from ..common import config

DEFAULT_ANOMALY_CODE = 0
DEFAULT_FILL_VALUE = 0.0
UNKNOWN_VALUE = "unknown"
Q1_PERCENTILE = 0.25
Q3_PERCENTILE = 0.75
IQR_MULTIPLIER = 1.5
REPORT_SEPARATOR = "=" * 50

NUMERIC_DTYPES = ["int64", "float64"]
OUTLIER_CHECK_COLUMNS = ["thrust", "mfr"]

THRUST_MIN_PHYSICAL = -0.1
THRUST_MAX_PHYSICAL = 15.0
MFR_MIN_PHYSICAL = -10.0
MFR_MAX_PHYSICAL = 3000.0

METADATA_TYPE_MAPPING = {
    "uid": int,
    "test_id": int,
    "sn": int,
    "test_pressure": float,
    "anomaly_code": float,
    "cumulated_throughput": float,
    "cumulated_on_time": float,
    "cumulated_pulses": float,
}

TIME_SERIES_TYPE_MAPPING = {
    "timestamp": float,
    "ton": int,
    "thrust": float,
    "mfr": float,
    "vl": int,
    "anomaly_code": float,
}


@dataclass
class CleaningStatistics:
    duplicates_removed: int = 0
    missing_values_handled: int = 0
    type_corrections: int = 0
    outliers_detected: int = 0
    physical_outliers_removed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicates_removed": self.duplicates_removed,
            "missing_values_handled": self.missing_values_handled,
            "type_corrections": self.type_corrections,
            "outliers_detected": self.outliers_detected,
            "physical_outliers_removed": self.physical_outliers_removed,
        }


class DataCleaner:
    def __init__(self):
        self.cleaning_stats = CleaningStatistics()

    def clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning metadata")
        df = df.copy()

        df = self._ensure_correct_types(df)
        df = self._handle_missing_values(df)
        df = self._remove_duplicates(df)
        df = self._add_split_column(df)

        logger.info(f"Metadata cleaned: {len(df)} records remain")
        return df

    def clean_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})

        if "timestamp" in df.columns and df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

        df = self._ensure_time_series_types(df)
        df = self._handle_missing_values_time_series(df)
        df = self._remove_duplicates(df)
        # df = self._remove_physical_outliers(df)
        return self._log_outlier_statistics(df)

    def _ensure_correct_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_anomaly_code_column(df)

        for col, dtype in METADATA_TYPE_MAPPING.items():
            if col in df.columns and df[col].dtype != dtype:
                df = self._safe_type_conversion(df, col, dtype)

        bool_columns = ["vl1", "vl2", "vl3", "anomalous"]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        return df

    def _ensure_time_series_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_anomaly_code_column(df)
        df = self._normalize_numeric_column(df, "vl", DEFAULT_ANOMALY_CODE)

        for col, dtype in TIME_SERIES_TYPE_MAPPING.items():
            if col in df.columns and df[col].dtype != dtype:
                df = self._safe_type_conversion(df, col, dtype, "time series column")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_before = df.isna().sum().sum()

        if missing_before > 0:
            for col in df.columns:
                has_missing_values = bool(df[col].isna().any())
                if has_missing_values:
                    fill_value = self._get_fill_strategy(df, col)
                    df[col] = df[col].fillna(fill_value)

            self.cleaning_stats.missing_values_handled += missing_before

        return df

    def _handle_missing_values_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_before = df.isna().sum().sum()

        if missing_before > 0:
            for col in df.columns:
                if not df[col].isna().any():
                    continue

                if col in ["thrust", "mfr"]:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                    df[col] = df[col].fillna(method="bfill")
                    df[col] = df[col].fillna(method="ffill")
                else:
                    fill_value = self._get_fill_strategy(df, col)
                    df[col] = df[col].fillna(fill_value)

            self.cleaning_stats.missing_values_handled += missing_before
            logger.debug(f"Handled {missing_before} missing values in time series")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        row_count_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = row_count_before_dedup - len(df)

        if duplicates_removed > 0:
            self.cleaning_stats.duplicates_removed += duplicates_removed

        return df

    def _remove_physical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)

        if "thrust" in df.columns:
            thrust_mask = (df["thrust"] >= THRUST_MIN_PHYSICAL) & (
                df["thrust"] <= THRUST_MAX_PHYSICAL
            )
            df = df[thrust_mask]

        if "mfr" in df.columns:
            mfr_mask = (df["mfr"] >= MFR_MIN_PHYSICAL) & (df["mfr"] <= MFR_MAX_PHYSICAL)
            df = df[mfr_mask]

        removed = initial_count - len(df)
        if removed > 0:
            self.cleaning_stats.physical_outliers_removed += removed
            logger.info(f"Removed {removed} physically impossible measurements")

        return df

    def _log_outlier_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in OUTLIER_CHECK_COLUMNS:
            if col in df.columns:
                column_series = cast(pd.Series, df[col])
                lower_bound, upper_bound = self._calculate_iqr_bounds(column_series)

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outlier_mask.sum()

                if outliers_count > 0:
                    self.cleaning_stats.outliers_detected += outliers_count
                    outlier_values = df.loc[outlier_mask, col]
                    min_outlier = outlier_values.min()
                    max_outlier = outlier_values.max()
                    logger.info(
                        f"Outliers in {col}: {outliers_count} detected "
                        f"(range: {min_outlier:.4f} to {max_outlier:.4f}, "
                        f"IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}])"
                    )

        return df

    def _add_split_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df["split"] = df["sn"].apply(self._determine_split)
        logger.info(
            f"Added split column: {(df['split'] == 'train').sum()} train, {(df['split'] == 'test').sum()} test"
        )
        return df

    @staticmethod
    def _determine_split(sn: int) -> str:
        return "train" if sn in config.TRAIN_SNS else "test"

    def _normalize_anomaly_code_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "anomaly_code" in df.columns:
            df["anomaly_code"] = df["anomaly_code"].replace("", DEFAULT_ANOMALY_CODE)
            numeric_series = cast(pd.Series, pd.to_numeric(df["anomaly_code"], errors="coerce"))
            df["anomaly_code"] = numeric_series.fillna(DEFAULT_ANOMALY_CODE)
        return df

    def _normalize_numeric_column(
        self, df: pd.DataFrame, column_name: str, default_value: int | float
    ) -> pd.DataFrame:
        if column_name in df.columns:
            df[column_name] = df[column_name].replace("", default_value)
            numeric_series = cast(pd.Series, pd.to_numeric(df[column_name], errors="coerce"))
            df[column_name] = numeric_series.fillna(default_value)
        return df

    def _safe_type_conversion(
        self, df: pd.DataFrame, col: str, dtype: type, col_type_desc: str = "column"
    ) -> pd.DataFrame:
        try:
            df[col] = df[col].astype(dtype)
            self.cleaning_stats.type_corrections += 1
        except ValueError as e:
            logger.warning(f"Failed to convert {col_type_desc} {col} to {dtype}: {e}")
        return df

    def _get_fill_strategy(self, df: pd.DataFrame, column_name: str) -> int | float | bool | str:
        if column_name == "anomaly_code":
            return DEFAULT_ANOMALY_CODE

        column_dtype = df[column_name].dtype
        if column_dtype in NUMERIC_DTYPES:
            median_value = df[column_name].median()
            is_valid_median = bool(pd.notna(median_value))
            return float(median_value) if is_valid_median else DEFAULT_FILL_VALUE

        if column_dtype == "bool":
            return False

        return UNKNOWN_VALUE

    def _calculate_iqr_bounds(self, series: pd.Series) -> tuple[float, float]:
        q1 = series.quantile(Q1_PERCENTILE)
        q3 = series.quantile(Q3_PERCENTILE)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        return lower_bound, upper_bound

    def get_cleaning_report(self) -> str:
        report = "Data Cleaning Report\n"
        report += REPORT_SEPARATOR + "\n"
        report += f"Duplicates removed: {self.cleaning_stats.duplicates_removed}\n"
        report += f"Missing values handled: {self.cleaning_stats.missing_values_handled}\n"
        report += f"Type corrections: {self.cleaning_stats.type_corrections}\n"
        report += f"Statistical outliers detected (NOT removed): {self.cleaning_stats.outliers_detected}\n"
        report += "\nNote: Outliers were identified but NOT removed for Midterm 1.\n"
        report += "Decision on outlier handling will be made in Midterm 2 during modeling phase.\n"
        return report
