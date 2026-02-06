"""
Second Midterm: Thrust Prediction - Model Comparison
Time Series Forecasting with Exogenous Inputs

Task: Predict thrust[t] given ton[t] and historical data
Models: Ridge Regression & Random Forest with lag features
Comparison: Side-by-side evaluation of 2 modeling approaches
Train: SN01-12 | Test: SN13-24

Feature Selection: SelectKBest, PCA comparison
"""

import argparse

from loguru import logger

from ..common import config
from ..common.logger import setup_logger
from ..modeling import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Thrust Prediction - Model Comparison (Ridge & Random Forest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with feature selection analysis
  python -m src.scripts.second_midterm --max-files 50 --feature-selection

  # First run: save data to cache (takes ~20min, but only once!)
  python -m src.scripts.second_midterm --save-cache --models ridge --skip-viz

  # Fast subsequent runs: load from cache (~30sec to load data)
  python -m src.scripts.second_midterm --use-cache --feature-selection

  # Use HistGradientBoosting instead of Random Forest (much faster)
  python -m src.scripts.second_midterm --use-cache --models ridge hist_gradient_boosting

  # Load pre-trained models
  python -m src.scripts.second_midterm --use-cache --load-ridge outputs/models/ridge_model.joblib
        """,
    )
    parser.add_argument("--lag", type=int, default=10, help="Number of lag features (default: 10)")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--ridge-solver",
        type=str,
        default="auto",
        choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        help="Ridge solver algorithm (default: auto)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["ridge", "linear", "random_forest", "hist_gradient_boosting", "all"],
        default=["ridge", "hist_gradient_boosting"],
        help="Models to train (default: ridge hist_gradient_boosting). Use 'all' for all models.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=100,
        help="Random Forest: number of trees (default: 100)",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=20,
        help="Random Forest: max tree depth (default: 20)",
    )
    parser.add_argument(
        "--rf-sample-ratio",
        type=float,
        default=0.2,
        help="Random Forest: fraction of data to use for training (default: 0.2 = 20%%). Use 1.0 for all data.",
    )
    parser.add_argument(
        "--load-ridge",
        type=str,
        default=None,
        help="Path to pre-trained Ridge model to load (skip Ridge training)",
    )
    parser.add_argument(
        "--load-random-forest",
        type=str,
        default=None,
        help="Path to pre-trained Random Forest model to load (skip RF training)",
    )
    parser.add_argument(
        "--load-hist-gb",
        type=str,
        default=None,
        help="Path to pre-trained HistGradientBoosting model to load",
    )
    parser.add_argument(
        "--hgb-max-iter",
        type=int,
        default=100,
        help="HistGradientBoosting: max iterations (default: 100)",
    )
    parser.add_argument(
        "--hgb-max-depth",
        type=int,
        default=None,
        help="HistGradientBoosting: max tree depth (default: None = unlimited)",
    )
    parser.add_argument(
        "--hgb-learning-rate",
        type=float,
        default=0.1,
        help="HistGradientBoosting: learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--hgb-min-samples-leaf",
        type=int,
        default=20,
        help="HistGradientBoosting: min samples per leaf (default: 20)",
    )
    parser.add_argument(
        "--hgb-l2-reg",
        type=float,
        default=0.0,
        help="HistGradientBoosting: L2 regularization (default: 0.0)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files to process per split (for testing, e.g., 100)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load train/test data from cache (much faster if available)",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Save train/test data to cache for future runs",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Run feature selection / dimensionality reduction analysis (SelectKBest, PCA)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip time-consuming visualizations (faster run)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (RandomizedSearchCV) to boost model performance",
    )
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=500_000,
        help="Sample size for hyperparameter tuning (default: 500000)",
    )
    parser.add_argument("--log-file", action="store_true", help="Enable logging to file")

    args = parser.parse_args()

    log_path = config.OUTPUTS_DIR / "second_midterm.log" if args.log_file else None
    setup_logger(enable_file_logging=args.log_file, log_path=log_path)

    if "all" in args.models:
        models_to_train = ["ridge", "linear", "hist_gradient_boosting", "random_forest"]
    else:
        models_to_train = args.models

    model_configs = []
    for model_type in models_to_train:
        if model_type == "ridge":
            model_configs.append(
                {"model_type": "ridge", "alpha": args.alpha, "solver": args.ridge_solver}
            )
        elif model_type == "linear":
            model_configs.append({"model_type": "linear"})
        elif model_type == "random_forest":
            model_configs.append(
                {
                    "model_type": "random_forest",
                    "n_estimators": args.rf_n_estimators,
                    "max_depth": args.rf_max_depth,
                    "n_jobs": -1,
                }
            )
        elif model_type == "hist_gradient_boosting":
            model_configs.append(
                {
                    "model_type": "hist_gradient_boosting",
                    "max_iter": args.hgb_max_iter,
                    "max_depth": args.hgb_max_depth,
                    "learning_rate": args.hgb_learning_rate,
                    "min_samples_leaf": args.hgb_min_samples_leaf,
                    "l2_regularization": args.hgb_l2_reg,
                }
            )

    logger.info(f"Models to train: {[c['model_type'] for c in model_configs]}")

    if args.rf_sample_ratio < 1.0:
        logger.info(
            f"Random Forest will use {args.rf_sample_ratio * 100:.0f}% of training data (memory optimization)"
        )
    if args.load_ridge:
        logger.info(f"Will load pre-trained Ridge model from: {args.load_ridge}")
    if args.load_random_forest:
        logger.info(f"Will load pre-trained Random Forest model from: {args.load_random_forest}")
    if args.load_hist_gb:
        logger.info(f"Will load pre-trained HistGradientBoosting model from: {args.load_hist_gb}")
    if args.feature_selection:
        logger.info("Feature selection analysis ENABLED (SelectKBest, PCA)")
    if args.use_cache:
        logger.info("Data caching ENABLED - will load from cache if available")
    if args.save_cache:
        logger.info("Data caching ENABLED - will save data to cache")
    if args.tune:
        logger.info(f"Hyperparameter tuning ENABLED (sample size: {args.tune_sample_size:,})")

    pipeline = Pipeline(
        lag=args.lag,
        rolling_window=10,
        model_configs=model_configs,
        max_files=args.max_files,
        skip_visualizations=args.skip_viz,
        rf_sample_ratio=args.rf_sample_ratio,
        load_ridge_path=args.load_ridge,
        load_random_forest_path=args.load_random_forest,
        load_hist_gb_path=args.load_hist_gb,
        run_feature_selection=args.feature_selection,
        use_cached_data=args.use_cache,
        save_data_cache=args.save_cache,
        run_tuning=args.tune,
        tuning_sample_size=args.tune_sample_size,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
