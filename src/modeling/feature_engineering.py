"""Feature engineering for time series thrust prediction."""

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Creates lag features from time series data"""

    def __init__(self, lag: int = 10, rolling_window: int = 10):
        self.lag = lag
        self.rolling_window = rolling_window

    def create_features(
        self, ts_df: pd.DataFrame, metadata_row: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create features for thrust prediction.

        18 features selected via backward elimination (BIC-optimal subset):
        - ton[t]: current command (0/1)
        - thrust[t-2 to t-lag]: autoregressive history (thrust[t-1] redundant with rolling_mean)
        - ton[t-3], ton[t-5]: sparse command history
        - mfr[t-1]: previous mass flow rate
        - rolling_mean_thrust, rolling_std_thrust: last 10 points
        - test_pressure: inlet pressure (constant per test)
        - cumulated_on_time, cumulated_pulses: aging factors

        Target:
        - thrust[t]: current thrust value
        """
        n_samples = len(ts_df) - self.lag
        if n_samples <= 0:
            return np.array([]), np.array([])

        # 1 (ton[t]) + (lag-1) thrust lags + 2 ton lags + 1 (mfr) + 2 (rolling) + 3 (aging) = 18
        n_features = 1 + (self.lag - 1) + 2 + 1 + 2 + 3
        features = np.zeros((n_samples, n_features))
        targets = np.zeros(n_samples)

        ton = ts_df["ton"].values
        thrust = ts_df["thrust"].values
        mfr = ts_df["mfr"].values

        pressure = metadata_row["test_pressure"]
        on_time = metadata_row["cumulated_on_time"]
        pulses = metadata_row["cumulated_pulses"]

        for i in range(self.lag, len(ts_df)):
            idx = i - self.lag
            col = 0

            features[idx, col] = ton[i]  # ton[t]
            col += 1

            for j in range(2, self.lag + 1):  # thrust[t-2 .. t-lag], skip t-1
                features[idx, col] = thrust[i - j]
                col += 1

            features[idx, col] = ton[i - 3]  # ton[t-3]
            col += 1
            features[idx, col] = ton[i - 5]  # ton[t-5]
            col += 1

            features[idx, col] = mfr[i - 1]  # mfr[t-1]
            col += 1

            recent_thrust = thrust[max(0, i - self.rolling_window) : i]
            features[idx, col] = np.mean(recent_thrust)  # type: ignore
            col += 1
            features[idx, col] = np.std(recent_thrust)  # type: ignore
            col += 1

            features[idx, col] = pressure  # test_pressure
            col += 1
            features[idx, col] = on_time  # cumulated_on_time
            col += 1
            features[idx, col] = pulses  # cumulated_pulses

            targets[idx] = thrust[i]

        return features, targets
