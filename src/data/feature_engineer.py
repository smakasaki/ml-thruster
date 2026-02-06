import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import linregress


class FeatureEngineer:
    ACCEPTANCE_TEST_IDS = list(range(1, 13))
    TEST_SNS = list(range(13, 25))

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering")
        df = df.copy()

        df = self._encode_test_mode(df)
        df = self._add_acceptance_test_features(df)
        df = self._add_interaction_features(df)
        df = self._add_ageing_ratio_features(df)

        logger.info(f"Feature engineering complete - {len(df.columns)} total features")
        return df

    def _encode_test_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        if "test_mode" not in df.columns:
            logger.warning("test_mode column not found - skipping encoding")
            return df

        logger.info("Encoding test_mode with one-hot encoding")
        test_mode_dummies = pd.get_dummies(df["test_mode"], prefix="mode", drop_first=False)

        df = pd.concat([df, test_mode_dummies], axis=1)
        df = df.drop(columns=["test_mode"])

        logger.info(f"Created {len(test_mode_dummies.columns)} test_mode features")
        return df

    def _add_acceptance_test_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding acceptance test features for test set")

        acceptance_features = [
            "acceptance_thrust_mean",
            "acceptance_thrust_std",
            "acceptance_thrust_trend",
            "acceptance_mfr_mean",
            "acceptance_mfr_std",
            "acceptance_mfr_trend",
        ]

        for feature in acceptance_features:
            df[feature] = np.nan

        test_mask = df["split"] == "test"
        test_sns = df.loc[test_mask, "sn"].unique()

        for sn in test_sns:
            acceptance_mask = (
                (df["sn"] == sn)
                & (df["test_id"] >= self.ACCEPTANCE_TEST_IDS[0])
                & (df["test_id"] <= self.ACCEPTANCE_TEST_IDS[-1])
            )
            acceptance_data = df[acceptance_mask]

            if len(acceptance_data) == 0:
                logger.warning(f"No acceptance test data found for SN{sn:02d}")
                continue

            thrust_mean = acceptance_data["thrust_mean"].mean()
            thrust_std = acceptance_data["thrust_mean"].std()
            mfr_mean = acceptance_data["mfr_mean"].mean()
            mfr_std = acceptance_data["mfr_mean"].std()

            thrust_trend = self._compute_linear_trend(
                acceptance_data["test_id"].values, acceptance_data["thrust_mean"].values
            )
            mfr_trend = self._compute_linear_trend(
                acceptance_data["test_id"].values, acceptance_data["mfr_mean"].values
            )

            sn_mask = df["sn"] == sn
            df.loc[sn_mask, "acceptance_thrust_mean"] = thrust_mean
            df.loc[sn_mask, "acceptance_thrust_std"] = thrust_std
            df.loc[sn_mask, "acceptance_thrust_trend"] = thrust_trend
            df.loc[sn_mask, "acceptance_mfr_mean"] = mfr_mean
            df.loc[sn_mask, "acceptance_mfr_std"] = mfr_std
            df.loc[sn_mask, "acceptance_mfr_trend"] = mfr_trend

            logger.debug(f"Added acceptance features for SN{sn:02d}")

        logger.info(f"Added {len(acceptance_features)} acceptance test features")
        return df

    def _compute_linear_trend(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0

        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 2:
            return 0.0

        try:
            slope, _, _, _, _ = linregress(x_valid, y_valid)
            return float(slope)
        except Exception as e:
            logger.debug(f"Failed to compute trend: {e}")
            return 0.0

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding interaction features")

        required_columns = ["test_pressure", "cumulated_throughput"]
        if all(col in df.columns for col in required_columns):
            df["pressure_throughput_interaction"] = df["test_pressure"] * df["cumulated_throughput"]

        if "test_pressure" in df.columns and "duty_cycle" in df.columns:
            df["pressure_duty_interaction"] = df["test_pressure"] * df["duty_cycle"]

        if "cumulated_pulses" in df.columns and "avg_pulse_duration" in df.columns:
            df["pulses_duration_interaction"] = df["cumulated_pulses"] * df["avg_pulse_duration"]

        logger.info("Interaction features added")
        return df

    def _add_ageing_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding ageing ratio features")

        ageing_columns = ["cumulated_throughput", "cumulated_on_time", "cumulated_pulses"]
        if all(col in df.columns for col in ageing_columns):
            df["throughput_per_hour"] = df["cumulated_throughput"] / (
                df["cumulated_on_time"] + 1e-6
            )
            df["pulses_per_hour"] = df["cumulated_pulses"] / (df["cumulated_on_time"] + 1e-6)

        logger.info("Ageing ratio features added")
        return df

    def prepare_features_for_modeling(
        self, df: pd.DataFrame, exclude_test_acceptance: bool = False
    ) -> pd.DataFrame:
        logger.info("Preparing features for modeling")
        df = df.copy()

        columns_to_drop = [
            "uid",
            "filename",
            "anomaly_code",
            "anomalous",
            "vl1",
            "vl2",
            "vl3",
            "anomaly_duration",
        ]

        if exclude_test_acceptance:
            test_acceptance_mask = (df["split"] == "test") & (
                df["test_id"].isin(self.ACCEPTANCE_TEST_IDS)
            )
            logger.info(f"Excluding {test_acceptance_mask.sum()} acceptance test records")
            df = df[~test_acceptance_mask]

        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        logger.info(f"Prepared dataset with {len(df.columns)} features")
        return df
