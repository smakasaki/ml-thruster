"""Hyperparameter tuning for thrust prediction models.

Uses RandomizedSearchCV on data subsamples to find optimal parameters.
Does NOT retrain on full data - just reports best parameters for later use.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


class HyperparameterTuner:
    """Hyperparameter tuning using RandomizedSearchCV on a data subset.

    NOTE: This class finds optimal parameters but does NOT retrain on full data.
    Use the best_params in your model config and retrain separately.
    """

    def __init__(self, output_dir: Path, sample_size: int = 500_000):
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.test_sample_size = 100_000
        self.results: list[dict] = []

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, size: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take a random subsample for faster tuning."""
        target_size = size or self.sample_size
        if len(X) <= target_size:
            return X, y

        np.random.seed(42)
        idx = np.random.choice(len(X), target_size, replace=False)
        return X[idx], y[idx]

    def tune_ridge(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Tune Ridge regression hyperparameters on subsample."""
        logger.info("=" * 60)
        logger.info("TUNING RIDGE REGRESSION")
        logger.info("=" * 60)

        X_sub, y_sub = self._subsample(X_train, y_train)
        logger.info(f"Train subsample: {len(X_sub):,} samples")

        X_test_sub, y_test_sub = self._subsample(X_test, y_test, self.test_sample_size)
        logger.info(f"Test subsample: {len(X_test_sub):,} samples")

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_test_scaled = scaler.transform(X_test_sub)

        param_dist = {
            "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            "solver": ["lsqr", "sag", "cholesky"],  # memory-efficient solvers (no svd)
        }

        ridge = Ridge(random_state=42)

        logger.info(f"Running RandomizedSearchCV with {len(param_dist['alpha'])} alpha values...")
        start_time = time.time()

        search = RandomizedSearchCV(
            ridge,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_sub_scaled, y_sub)

        tuning_time = time.time() - start_time
        logger.info(f"Tuning completed in {tuning_time:.1f}s")
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV RMSE: {-search.best_score_:.4f}")

        # Evaluate best model on test subsample (not full data)
        y_pred = search.best_estimator_.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test_sub, y_pred))
        mae = mean_absolute_error(y_test_sub, y_pred)
        r2 = r2_score(y_test_sub, y_pred)

        result = {
            "model": "Ridge (Tuned)",
            "best_params": search.best_params_,
            "cv_rmse": -search.best_score_,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "tuning_time": tuning_time,
            "note": "Evaluated on 100K test subsample",
        }

        logger.info(f"Test RMSE (subsample): {rmse:.4f} N")
        logger.info(f"Test MAE (subsample):  {mae:.4f} N")
        logger.info(f"Test R² (subsample):   {r2:.4f}")

        self.results.append(result)
        return result

    def tune_hist_gradient_boosting(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Tune HistGradientBoosting hyperparameters on subsample."""
        logger.info("=" * 60)
        logger.info("TUNING HIST GRADIENT BOOSTING")
        logger.info("=" * 60)

        X_sub, y_sub = self._subsample(X_train, y_train)
        logger.info(f"Train subsample: {len(X_sub):,} samples")

        X_test_sub, y_test_sub = self._subsample(X_test, y_test, self.test_sample_size)
        logger.info(f"Test subsample: {len(X_test_sub):,} samples")

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_test_scaled = scaler.transform(X_test_sub)

        param_dist = {
            "max_iter": [50, 100, 150, 200, 300],
            "max_depth": [5, 10, 15, 20, 25, None],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "min_samples_leaf": [10, 20, 50, 100],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        }

        hgb = HistGradientBoostingRegressor(random_state=42, early_stopping=False, verbose=0)

        logger.info("Running RandomizedSearchCV (this may take a while)...")
        start_time = time.time()

        search = RandomizedSearchCV(
            hgb,
            param_distributions=param_dist,
            n_iter=30,
            cv=3,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_sub_scaled, y_sub)

        tuning_time = time.time() - start_time
        logger.info(f"Tuning completed in {tuning_time:.1f}s")
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV RMSE: {-search.best_score_:.4f}")

        # Evaluate best model on test subsample (not full data)
        y_pred = search.best_estimator_.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test_sub, y_pred))
        mae = mean_absolute_error(y_test_sub, y_pred)
        r2 = r2_score(y_test_sub, y_pred)

        result = {
            "model": "HistGradientBoosting (Tuned)",
            "best_params": search.best_params_,
            "cv_rmse": -search.best_score_,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "tuning_time": tuning_time,
            "note": "Evaluated on 100K test subsample",
        }

        logger.info(f"Test RMSE (subsample): {rmse:.4f} N")
        logger.info(f"Test MAE (subsample):  {mae:.4f} N")
        logger.info(f"Test R² (subsample):   {r2:.4f}")

        self.results.append(result)
        return result

    def save_results(self):
        """Save tuning results to CSV."""
        if not self.results:
            logger.warning("No tuning results to save")
            return

        rows = []
        for r in self.results:
            rows.append(
                {
                    "Model": r["model"],
                    "Best Params": str(r["best_params"]),
                    "CV RMSE": r["cv_rmse"],
                    "Test RMSE (subsample)": r["test_rmse"],
                    "Test MAE (subsample)": r["test_mae"],
                    "Test R² (subsample)": r["test_r2"],
                    "Tuning Time (s)": r["tuning_time"],
                }
            )

        df = pd.DataFrame(rows)
        filepath = self.output_dir / "hyperparameter_tuning_results.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Tuning results saved to: {filepath}")

        logger.info("\n" + "=" * 60)
        logger.info("HYPERPARAMETER TUNING SUMMARY")
        logger.info("=" * 60)
        logger.info("(Metrics based on 500K train / 100K test subsamples)")
        for r in self.results:
            logger.info(f"\n{r['model']}:")
            logger.info(f"  Best params: {r['best_params']}")
            logger.info(f"  CV RMSE: {r['cv_rmse']:.4f} N")
            logger.info(f"  Test RMSE: {r['test_rmse']:.4f} N")
            logger.info(f"  Test R²: {r['test_r2']:.4f}")

        logger.info("\n" + "-" * 60)
        logger.info("TO USE THESE PARAMETERS:")
        logger.info("Update your model config and run training again, e.g.:")
        logger.info(
            "  python -m src.scripts.second_midterm --use-cache --models ridge hist_gradient_boosting"
        )
        logger.info("with updated --alpha, --hgb-max-iter, etc.")
