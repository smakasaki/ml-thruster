"""Main pipeline for thrust prediction."""

import time
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

from ..common import config
from ..common.time import format_elapsed_time
from .data_loader import TimeSeriesLoader
from .evaluator import Evaluator
from .feature_engineering import FeatureEngineering
from .feature_selector import FeatureSelector
from .predictor import ThrustPredictor
from .tuner import HyperparameterTuner

DATA_CACHE_DIR = config.OUTPUTS_DIR / "cache"


class Pipeline:
    """Main pipeline for thrust prediction - supports comparing multiple models"""

    def __init__(
        self,
        lag: int = 10,
        rolling_window: int = 10,
        model_configs: list[dict] | None = None,
        max_files: int | None = None,
        skip_visualizations: bool = False,
        rf_sample_ratio: float = 1.0,
        load_ridge_path: str | None = None,
        load_random_forest_path: str | None = None,
        load_hist_gb_path: str | None = None,
        run_feature_selection: bool = False,
        use_cached_data: bool = False,
        save_data_cache: bool = False,
        run_tuning: bool = False,
        tuning_sample_size: int = 500_000,
    ):
        self.loader = TimeSeriesLoader(config.METADATA_PATH, config.TRAIN_DIR, config.TEST_DIR)
        self.feature_eng = FeatureEngineering(lag=lag, rolling_window=rolling_window)
        self.evaluator = Evaluator(config.FIGURES_DIR)
        self.feature_selector = FeatureSelector(config.FIGURES_DIR)
        self.tuner = HyperparameterTuner(config.FIGURES_DIR.parent / "reports", tuning_sample_size)
        self.max_files = max_files
        self.skip_visualizations = skip_visualizations
        self.rf_sample_ratio = rf_sample_ratio
        self.load_ridge_path = load_ridge_path
        self.load_random_forest_path = load_random_forest_path
        self.load_hist_gb_path = load_hist_gb_path
        self.run_feature_selection = run_feature_selection
        self.use_cached_data = use_cached_data
        self.save_data_cache = save_data_cache
        self.run_tuning = run_tuning

        if model_configs is None:
            model_configs = [{"model_type": "ridge", "alpha": 1.0}]
        self.model_configs = model_configs
        self.output_dir = config.FIGURES_DIR.parent / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self, split: str):
        """Load all files and create X, y arrays"""
        logger.info(f"Loading {split} data...")

        metadata = self.loader.metadata
        subset = metadata[metadata["split"] == split]

        if self.max_files is not None:
            subset = subset.head(self.max_files)
            logger.warning(f"LIMITED TO {self.max_files} FILES FOR TESTING!")

        X_list = []
        y_list = []
        file_indices = []

        total_files = len(subset)
        processed = 0
        failed = 0
        start_time = time.time()

        for idx, row in subset.iterrows():
            try:
                if processed < 5:
                    logger.info(f"Processing file {processed + 1}: {row['filename']}")

                ts_df = self.loader.load_time_series(str(row["filename"]), split)

                if processed < 5:
                    logger.info(f"  Loaded {len(ts_df)} rows, creating features...")

                X, y = self.feature_eng.create_features(ts_df, row)

                if processed < 5:
                    logger.info(f"  Created {len(X)} feature samples")

                X_list.append(X)
                y_list.append(y)
                file_indices.extend([idx] * len(X))

                processed += 1

                if processed > 5 and processed % 10 != 0:
                    print(".", end="", flush=True)

                if processed % 10 == 0 or processed == 5:
                    if processed > 5 and processed % 10 == 0:
                        print()
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    percent = (processed / total_files) * 100
                    logger.info(
                        f"Progress: {percent:.1f}% ({processed}/{total_files}) | "
                        f"Speed: {speed:.1f} files/s | Errors: {failed}"
                    )

            except Exception as e:
                failed += 1
                logger.warning(f"Failed to process {row['filename']}: {e}")
                continue

        if len(X_list) == 0:
            raise ValueError(f"No data loaded for split={split}")

        logger.info(f"Concatenating {len(X_list)} feature arrays...")
        X = np.vstack(X_list)
        logger.info(f"Concatenating {len(y_list)} target arrays...")
        y = np.concatenate(y_list)

        logger.info(f"Loaded {split}: {len(X):,} samples from {processed} files")
        logger.info(f"Features: {X.shape[1]}")

        return X, y, file_indices

    def sample_data_for_rf(self, X: np.ndarray, y: np.ndarray, ratio: float):
        """Sample data for Random Forest training (memory optimization)"""
        if ratio >= 1.0:
            return X, y

        n_samples = len(X)
        n_sampled = int(n_samples * ratio)

        logger.info(
            f"Sampling {ratio * 100:.0f}% of data for Random Forest: {n_sampled:,} / {n_samples:,} samples"
        )

        np.random.seed(42)
        indices = np.random.choice(n_samples, n_sampled, replace=False)
        indices = np.sort(indices)

        return X[indices], y[indices]

    def _get_cache_path(self, split: str) -> Path:
        """Get cache file path for a data split."""
        suffix = f"_max{self.max_files}" if self.max_files else ""
        return DATA_CACHE_DIR / f"{split}_data{suffix}.joblib"

    def _load_cached_data(self, split: str) -> tuple[np.ndarray, np.ndarray, list] | None:
        """Load cached data if available."""
        cache_path = self._get_cache_path(split)
        if cache_path.exists():
            logger.info(f"Loading cached {split} data from {cache_path}")
            start = time.time()
            data = joblib.load(cache_path)
            elapsed = time.time() - start
            logger.info(f"Loaded {split} cache in {elapsed:.1f}s: {len(data['X']):,} samples")
            return data["X"], data["y"], data["indices"]
        return None

    def _save_data_cache(self, split: str, X: np.ndarray, y: np.ndarray, indices: list):
        """Save processed data to cache."""
        cache_path = self._get_cache_path(split)
        logger.info(f"Saving {split} data cache to {cache_path}")
        start = time.time()
        joblib.dump({"X": X, "y": y, "indices": indices}, cache_path, compress=3)
        elapsed = time.time() - start
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {split} cache in {elapsed:.1f}s ({size_mb:.1f} MB)")

    def load_or_prepare_data(self, split: str) -> tuple[np.ndarray, np.ndarray, list]:
        """Load from cache or prepare data."""
        if self.use_cached_data:
            cached = self._load_cached_data(split)
            if cached is not None:
                return cached
            logger.warning(f"Cache not found for {split}, loading from files...")

        X, y, indices = self.load_and_prepare_data(split)

        if self.save_data_cache:
            self._save_data_cache(split, X, y, indices)

        return X, y, indices

    def run(self):
        """Run full pipeline with model comparison"""
        logger.info("=" * 70)
        logger.info("MIDTERM 2: THRUST PREDICTION - MODEL COMPARISON")
        logger.info("=" * 70)

        start_time = time.time()

        try:
            logger.info("\n[1/5] Loading metadata...")
            self.loader.load_metadata()

            logger.info("\n[2/5] Loading training data (SN01-12)...")
            X_train, y_train, train_indices = self.load_or_prepare_data("train")

            logger.info(f"\n[3/5] Training {len(self.model_configs)} model(s)...")
            models_results = {}

            for config in self.model_configs:
                model_name = config["model_type"].upper()
                model_type = config["model_type"]

                if model_type == "ridge" and self.load_ridge_path:
                    logger.info(f"\n--- Loading pre-trained RIDGE from {self.load_ridge_path} ---")
                    predictor = ThrustPredictor.load(self.load_ridge_path)

                    models_results[model_name] = {
                        "predictor": predictor,
                        "training_time": predictor.training_time,
                    }
                    continue

                if model_type == "random_forest" and self.load_random_forest_path:
                    logger.info(
                        f"\n--- Loading pre-trained RANDOM_FOREST from {self.load_random_forest_path} ---"
                    )
                    predictor = ThrustPredictor.load(self.load_random_forest_path)

                    models_results[model_name] = {
                        "predictor": predictor,
                        "training_time": predictor.training_time,
                    }
                    continue

                if model_type == "hist_gradient_boosting" and self.load_hist_gb_path:
                    logger.info(
                        f"\n--- Loading pre-trained HIST_GRADIENT_BOOSTING from {self.load_hist_gb_path} ---"
                    )
                    predictor = ThrustPredictor.load(self.load_hist_gb_path)

                    models_results[model_name] = {
                        "predictor": predictor,
                        "training_time": predictor.training_time,
                    }
                    continue

                logger.info(f"\n--- Training {model_name} ---")

                if model_type == "random_forest" and self.rf_sample_ratio < 1.0:
                    X_train_model, y_train_model = self.sample_data_for_rf(
                        X_train, y_train, self.rf_sample_ratio
                    )
                else:
                    X_train_model, y_train_model = X_train, y_train

                predictor = ThrustPredictor(**config)
                predictor.fit(X_train_model, y_train_model)

                model_path = self.output_dir / f"{model_type}_model.joblib"
                predictor.save(str(model_path))

                models_results[model_name] = {
                    "predictor": predictor,
                    "training_time": predictor.training_time,
                }

            logger.info("\n[4/7] Loading test data (SN13-24)...")
            X_test, y_test, test_indices = self.load_or_prepare_data("test")

            if self.run_feature_selection:
                logger.info(
                    "\n[5/7] Running Feature Selection / Dimensionality Reduction Analysis..."
                )
                self.feature_selector.run_analysis(
                    X_train, y_train, X_test, y_test, sample_size=500000
                )
                logger.info(self.feature_selector.get_conclusion())
            else:
                logger.info(
                    "\n[5/7] Skipping feature selection analysis (use --feature-selection to enable)"
                )

            if self.run_tuning:
                logger.info("\n[6/7] Running Hyperparameter Tuning (Model Boosting)...")
                model_types = [c["model_type"] for c in self.model_configs]

                if "ridge" in model_types:
                    self.tuner.tune_ridge(X_train, y_train, X_test, y_test)

                if "hist_gradient_boosting" in model_types:
                    self.tuner.tune_hist_gradient_boosting(X_train, y_train, X_test, y_test)

                self.tuner.save_results()
            else:
                logger.info("\n[6/7] Skipping hyperparameter tuning (use --tune to enable)")

            logger.info("\n[7/9] Evaluating models on test set...")

            for model_name, result in models_results.items():
                logger.info(f"\n--- Evaluating {model_name} ---")
                predictor = result["predictor"]

                if model_name in ("RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"):
                    y_pred = predictor.predict_in_batches(X_test, batch_size=1000000)
                else:
                    y_pred = predictor.predict(X_test)

                metrics = self.evaluator.evaluate(y_test, y_pred, f"Test - {model_name}")

                result["y_pred"] = y_pred
                result["y_true"] = y_test
                result["metrics"] = metrics

            if len(models_results) > 1 and self.loader.metadata is not None:
                logger.info("\n[8/9] Comparing models...")
                self.evaluator.compare_models(models_results, self.loader.metadata, test_indices)
            else:
                if len(models_results) == 1:
                    logger.info("\n[8/9] Single model - skipping comparison")
                else:
                    logger.warning("\n[8/9] Metadata not available - skipping comparison")

            if not self.skip_visualizations:
                logger.info("\n[9/9] Generating individual model visualizations...")

                for model_name, result in models_results.items():
                    logger.info(f"\nGenerating visualizations for {model_name}...")
                    y_pred = result["y_pred"]
                    y_test = result["y_true"]
                    metrics = result["metrics"]
                    predictor = result["predictor"]

                    model_output_dir = self.evaluator.output_dir.parent / model_name.lower()
                    model_output_dir.mkdir(parents=True, exist_ok=True)

                    original_output_dir = self.evaluator.output_dir
                    self.evaluator.output_dir = model_output_dir

                    self.evaluator.plot_predictions_vs_actual(y_test, y_pred, metrics)
                    self.evaluator.plot_residuals(y_test, y_pred)
                    self.evaluator.plot_time_series_examples(
                        self.loader, predictor, self.feature_eng, n_examples=3
                    )

                    if self.loader.metadata is not None:
                        self.evaluator.per_thruster_analysis(
                            self.loader.metadata, test_indices, y_test, y_pred
                        )

                    self.evaluator.output_dir = original_output_dir
            else:
                logger.info("\n[9/9] Skipping visualizations (--skip-viz flag)")

            elapsed = format_elapsed_time(time.time() - start_time)
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Pipeline completed successfully in {elapsed}")
            logger.info(f"{'=' * 70}")

        except Exception as e:
            elapsed = format_elapsed_time(time.time() - start_time)
            logger.error(f"Pipeline failed after {elapsed}: {e}")
            raise
