"""Thrust prediction models."""

import time

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class ThrustPredictor:
    """Model for thrust prediction - supports Ridge, Linear, Random Forest, and HistGradientBoosting"""

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        solver: str = "auto",
        n_estimators: int = 100,
        max_depth: int | None = 20,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        n_jobs: int = -1,
    ):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.training_time = None
        self.training_metrics = {}

        if model_type == "ridge":
            self.model = Ridge(alpha=alpha, solver=solver, random_state=42)
        elif model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=n_jobs,
                verbose=0,
            )
        elif model_type == "hist_gradient_boosting":
            self.model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42,
                verbose=1,
                early_stopping="auto",
                validation_fraction=0.1,
                n_iter_no_change=10,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type.upper()} model")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        logger.info(
            f"Training {self.model_type.upper()} on {len(X_train):,} samples with {X_train.shape[1]} features"
        )

        X_train_scaled = self.scaler.fit_transform(X_train)

        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        self.training_time = time.time() - start_time

        y_train_pred = self.model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        self.training_metrics = {
            "rmse": train_rmse,
            "mae": train_mae,
            "r2": train_r2,
        }

        logger.info(f"Training completed in {self.training_time:.2f}s")
        logger.info(f"Training RMSE: {train_rmse:.4f} N")
        logger.info(f"Training MAE:  {train_mae:.4f} N")
        logger.info(f"Training R²:   {train_r2:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on test set"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_in_batches(self, X: np.ndarray, batch_size: int = 1000000) -> np.ndarray:
        """Predict in batches to avoid memory issues (for large datasets)"""
        n_samples = len(X)
        predictions = np.empty(n_samples, dtype=np.float32)

        logger.info(f"Predicting in batches of {batch_size:,} samples...")
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_num = i // batch_size + 1

            if batch_num % 10 == 0 or batch_num == n_batches:
                logger.info(f"  Batch {batch_num}/{n_batches}: samples {i:,}-{batch_end:,}")

            X_batch_scaled = self.scaler.transform(X[i:batch_end])
            predictions[i:batch_end] = self.model.predict(X_batch_scaled)

        logger.info(f"Completed batched prediction for {n_samples:,} samples")
        return predictions

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance (coefficients for Ridge/Linear, feature_importances_ for tree-based)"""
        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                if importance is not None:
                    return np.array(importance)
        except Exception:
            pass

        try:
            if hasattr(self.model, "coef_"):
                return np.abs(self.model.coef_)
        except Exception:
            pass

        return None

    def save(self, filepath: str):
        """Save model and scaler to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "training_time": self.training_time,
            "training_metrics": self.training_metrics,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model and scaler from disk"""
        model_data = joblib.load(filepath)

        instance = cls.__new__(cls)
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.model_type = model_data["model_type"]
        instance.training_time = model_data.get("training_time")
        instance.training_metrics = model_data.get("training_metrics", {})

        logger.info(f"Model loaded from: {filepath}")
        return instance
