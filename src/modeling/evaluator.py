"""Model evaluation and visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.1)

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "neutral": "#3B3B3B",
}


class Evaluator:
    """Evaluation and visualization with professional styling."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "axes.titleweight": "bold",
                "axes.labelweight": "medium",
                "figure.facecolor": "white",
                "axes.facecolor": "#FAFAFA",
                "axes.edgecolor": "#CCCCCC",
                "grid.color": "#E0E0E0",
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
            }
        )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str = "Test") -> dict:
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

        logger.info(f"\n{'=' * 60}")
        logger.info(f"{split_name.upper()} SET RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"RMSE: {rmse:.4f} N")
        logger.info(f"MAE:  {mae:.4f} N")
        logger.info(f"R²:   {r2:.4f}")

        return metrics

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict):
        """Scatter plot: predictions vs actual (colored by error)."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        sample_size = min(10000, len(y_true))
        np.random.seed(42)
        indices = np.random.choice(len(y_true), sample_size, replace=False)

        p_low = np.percentile(np.concatenate([y_true, y_pred]), 0.1)
        p_high = np.percentile(np.concatenate([y_true, y_pred]), 99.9)
        margin = (p_high - p_low) * 0.05
        axis_min = max(0, p_low - margin)
        axis_max = p_high + margin
        scatter = ax.scatter(
            y_true[indices],
            y_pred[indices],
            c=np.abs(y_true[indices] - y_pred[indices]),
            cmap="RdYlGn_r",
            alpha=0.6,
            s=15,
            edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Absolute Error (N)", fontsize=10)

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color=COLORS["success"],
            linewidth=2.5,
            linestyle="--",
            label="Perfect prediction",
            zorder=10,
        )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        ax.set_xlabel("Actual Thrust (N)", fontsize=12, fontweight="medium")
        ax.set_ylabel("Predicted Thrust (N)", fontsize=12, fontweight="medium")
        ax.set_title("Predictions vs Actual (colored by error)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)

        textstr = (
            f"RMSE: {metrics['rmse']:.4f} N\nMAE: {metrics['mae']:.4f} N\nR²: {metrics['r2']:.4f}"
        )
        props = dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["primary"], alpha=0.9
        )
        ax.text(
            0.95,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        # ax = axes[1]
        # h = ax.hist2d(
        #     y_true[indices],
        #     y_pred[indices],
        #     bins=50,
        #     cmap="Blues",
        #     cmin=1,
        #     range=[[axis_min, axis_max], [axis_min, axis_max]],
        # )
        # plt.colorbar(h[3], ax=ax, shrink=0.8, label="Count")
        # ax.plot(
        #     [axis_min, axis_max],
        #     [axis_min, axis_max],
        #     color=COLORS["success"],
        #     linewidth=2.5,
        #     linestyle="--",
        #     label="Perfect prediction",
        # )
        # ax.set_xlim(axis_min, axis_max)
        # ax.set_ylim(axis_min, axis_max)
        # ax.set_xlabel("Actual Thrust (N)", fontsize=12, fontweight="medium")
        # ax.set_ylabel("Predicted Thrust (N)", fontsize=12, fontweight="medium")
        # ax.set_title("Prediction Density Heatmap", fontsize=13, fontweight="bold")
        # ax.legend(loc="upper left", fontsize=10)

        # plt.suptitle(
        #     f"Thrust Prediction Quality Analysis (n={len(y_true):,} test points)",
        #     fontsize=14,
        #     fontweight="bold",
        #     y=1.01,
        # )
        plt.tight_layout()
        filepath = self.output_dir / "thrust_predictions_scatter.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Residual analysis plot with enhanced diagnostics."""
        residuals = y_true - y_pred

        sample_size = min(10000, len(residuals))
        np.random.seed(42)
        idx = np.random.choice(len(residuals), sample_size, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(
            y_pred[idx], residuals[idx], alpha=0.4, s=8, c=COLORS["primary"], edgecolors="none"
        )
        ax.axhline(y=0, color=COLORS["success"], linestyle="--", linewidth=2.5, label="Zero line")
        # ax.axhline(
        #     y=residuals.std() * 2,
        #     color=COLORS["accent"],
        #     linestyle=":",
        #     linewidth=1.5,
        #     alpha=0.7,
        #     label="±2σ",
        # )
        # ax.axhline(
        #     y=-residuals.std() * 2, color=COLORS["accent"], linestyle=":", linewidth=1.5, alpha=0.7
        # )
        ax.set_xlabel("Predicted Thrust (N)", fontsize=11, fontweight="medium")
        ax.set_ylabel("Residuals (N)", fontsize=11, fontweight="medium")
        ax.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

        # ax = axes[0, 1]
        # n, bins, patches = ax.hist(
        #     residuals, bins=80, edgecolor="white", alpha=0.8, color=COLORS["primary"], density=True
        # )

        # from scipy.stats import norm

        # mu, std = norm.fit(residuals)
        # x = np.linspace(residuals.min(), residuals.max(), 100)
        # ax.plot(
        #     x,
        #     norm.pdf(x, mu, std),
        #     color=COLORS["success"],
        #     linewidth=2.5,
        #     label=f"Normal fit (μ={mu:.4f}, σ={std:.4f})",
        # )
        # ax.axvline(0, color=COLORS["accent"], linestyle="--", linewidth=2, label="Zero")
        # ax.set_xlabel("Residual (N)", fontsize=11, fontweight="medium")
        # ax.set_ylabel("Density", fontsize=11, fontweight="medium")
        # ax.set_title("Residual Distribution (Normality Check)", fontsize=12, fontweight="bold")
        # ax.legend(loc="upper right", fontsize=9)

        # ax = axes[1, 0]
        # stats.probplot(residuals[idx], dist="norm", plot=ax)
        # ax.get_lines()[0].set_markerfacecolor(COLORS["primary"])
        # ax.get_lines()[0].set_markeredgecolor("white")
        # ax.get_lines()[0].set_markersize(4)
        # ax.get_lines()[1].set_color(COLORS["success"])
        # ax.get_lines()[1].set_linewidth(2)
        # ax.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold")

        ax = axes[1]
        abs_residuals = np.abs(residuals[idx])
        percentiles = [50, 90, 95, 99]
        percentile_values = [np.percentile(abs_residuals, p) for p in percentiles]

        bars = ax.bar(
            [f"{p}th" for p in percentiles],
            percentile_values,
            color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]],
            edgecolor="white",
            linewidth=2,
        )
        ax.set_xlabel("Percentile", fontsize=11, fontweight="medium")
        ax.set_ylabel("Absolute Error (N)", fontsize=11, fontweight="medium")
        ax.set_title("Error Distribution by Percentile", fontsize=12, fontweight="bold")

        for bar, val in zip(bars, percentile_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="medium",
            )

        stats_text = (
            f"Mean: {residuals.mean():.6f} N\n"
            f"Std: {residuals.std():.6f} N\n"
            f"Skew: {stats.skew(residuals):.3f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["neutral"], alpha=0.9),
        )

        plt.suptitle(
            "Residual Analysis & Model Diagnostics", fontsize=15, fontweight="bold", y=0.995
        )

        plt.tight_layout()
        filepath = self.output_dir / "residual_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_time_series_examples(
        self,
        loader,
        predictor,
        feature_eng,
        n_examples: int = 3,
    ):
        """Plot example predictions on actual time series with enhanced styling."""
        metadata = loader.metadata
        test_samples = metadata[metadata["split"] == "test"].sample(n=n_examples, random_state=42)

        fig, axes = plt.subplots(n_examples, 1, figsize=(16, 4.5 * n_examples))
        if n_examples == 1:
            axes = [axes]

        for idx, (_, row) in enumerate(test_samples.iterrows()):
            try:
                ts_df = loader.load_time_series(str(row["filename"]), "test")
                X_test, y_test = feature_eng.create_features(ts_df, row)
                y_pred = predictor.predict(X_test)

                time = ts_df.iloc[feature_eng.lag :]["timestamp"].values

                ax = axes[idx]

                ton_signal = ts_df.iloc[feature_eng.lag :]["ton"].values
                ax.fill_between(
                    time,
                    0,
                    ton_signal * y_test.max() * 1.1,
                    alpha=0.15,
                    color=COLORS["neutral"],
                    label="Command ON period",
                )

                ax.plot(
                    time,
                    y_test,
                    label="Actual Thrust",
                    linewidth=1.8,
                    alpha=0.9,
                    color=COLORS["primary"],
                )
                ax.plot(
                    time,
                    y_pred,
                    label="Predicted Thrust",
                    linewidth=1.5,
                    alpha=0.85,
                    color=COLORS["accent"],
                    linestyle="--",
                )

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                ax.set_xlabel("Time (s)", fontsize=11, fontweight="medium")
                ax.set_ylabel("Thrust (N)", fontsize=11, fontweight="medium")

                title = (
                    f"SN{row['sn']:02d} | {row['test_mode']} | "
                    f"Pressure: {row['test_pressure']:.0f} bars | "
                    f"Throughput: {row['cumulated_throughput']:.1f} kg"
                )
                ax.set_title(title, fontsize=12, fontweight="bold", loc="left")

                metrics_text = f"RMSE: {rmse:.4f} N\nMAE: {mae:.4f} N\nR²: {r2:.4f}"
                ax.text(
                    0.99,
                    0.97,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor=COLORS["primary"],
                        alpha=0.95,
                    ),
                )

                ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
                ax.set_xlim(time.min(), time.max())

            except Exception as e:
                logger.warning(f"Could not plot {row['filename']}: {e}")
                axes[idx].text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    ha="center",
                    va="center",
                    transform=axes[idx].transAxes,
                    fontsize=12,
                )

        plt.suptitle(
            "Time Series Prediction Examples (Test Set: SN13-24)",
            fontsize=14,
            fontweight="bold",
            y=1.005,
        )
        plt.tight_layout()
        filepath = self.output_dir / "time_series_examples.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def per_thruster_analysis(
        self, metadata: pd.DataFrame, test_indices: list, y_true: np.ndarray, y_pred: np.ndarray
    ):
        """Analyze performance per thruster (SN13-24)"""
        logger.info(f"\n{'=' * 60}")
        logger.info("PER-THRUSTER PERFORMANCE (SN13-24)")
        logger.info(f"{'=' * 60}")

        results = []

        sn_array = np.array([metadata.loc[idx, "sn"] for idx in test_indices])

        for sn in range(13, 25):
            sn_mask = sn_array == sn

            if sn_mask.sum() == 0:
                continue

            y_true_sn = y_true[sn_mask]
            y_pred_sn = y_pred[sn_mask]

            rmse_sn = np.sqrt(mean_squared_error(y_true_sn, y_pred_sn))
            mae_sn = mean_absolute_error(y_true_sn, y_pred_sn)
            r2_sn = r2_score(y_true_sn, y_pred_sn)

            results.append(
                {
                    "SN": f"SN{sn:02d}",
                    "RMSE": rmse_sn,
                    "MAE": mae_sn,
                    "R2": r2_sn,
                    "Points": len(y_true_sn),
                }
            )

            logger.info(
                f"SN{sn:02d}: RMSE={rmse_sn:.4f}, MAE={mae_sn:.4f}, R²={r2_sn:.4f} "
                f"({len(y_true_sn):,} points)"
            )

        results_df = pd.DataFrame(results)
        filepath = self.output_dir.parent.parent / "reports" / "per_thruster_results.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(filepath, index=False)
        logger.info(f"\nPer-thruster results saved to: {filepath}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        sns.barplot(data=results_df, x="SN", y="RMSE", ax=axes[0], color="#1f77b4")
        axes[0].set_title("RMSE per Thruster")
        axes[0].set_ylabel("RMSE (N)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(alpha=0.3, axis="y")

        sns.barplot(data=results_df, x="SN", y="MAE", ax=axes[1], color="#ff7f0e")
        axes[1].set_title("MAE per Thruster")
        axes[1].set_ylabel("MAE (N)")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(alpha=0.3, axis="y")

        sns.barplot(data=results_df, x="SN", y="R2", ax=axes[2], color="#2ca02c")
        axes[2].set_title("R² per Thruster")
        axes[2].set_ylabel("R² Score")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(alpha=0.3, axis="y")
        axes[2].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="0.8")

        plt.tight_layout()
        filepath = self.output_dir / "per_thruster_performance.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def compare_models(
        self,
        models_results: dict,
        metadata: pd.DataFrame,
        test_indices: list,
    ):
        """Compare multiple models performance"""
        logger.info(f"\n{'=' * 60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'=' * 60}")

        comparison_data = []
        for model_name, result in models_results.items():
            comparison_data.append(
                {
                    "Model": model_name,
                    "RMSE": result["metrics"]["rmse"],
                    "MAE": result["metrics"]["mae"],
                    "R²": result["metrics"]["r2"],
                    "Training Time (s)": result.get("training_time", 0),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        logger.info("\n" + comparison_df.to_string(index=False))

        filepath = self.output_dir.parent.parent / "reports" / "model_comparison.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(filepath, index=False)
        logger.info(f"\nModel comparison saved to: {filepath}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.barplot(
            data=comparison_df,
            x="Model",
            y="RMSE",
            hue="Model",
            ax=axes[0, 0],
            palette="Set2",
            legend=False,
        )
        axes[0, 0].set_title("RMSE Comparison", fontsize=14)
        axes[0, 0].set_ylabel("RMSE (N)")
        axes[0, 0].grid(alpha=0.3, axis="y")

        sns.barplot(
            data=comparison_df,
            x="Model",
            y="MAE",
            hue="Model",
            ax=axes[0, 1],
            palette="Set2",
            legend=False,
        )
        axes[0, 1].set_title("MAE Comparison", fontsize=14)
        axes[0, 1].set_ylabel("MAE (N)")
        axes[0, 1].grid(alpha=0.3, axis="y")

        sns.barplot(
            data=comparison_df,
            x="Model",
            y="R²",
            hue="Model",
            ax=axes[1, 0],
            palette="Set2",
            legend=False,
        )
        axes[1, 0].set_title("R² Comparison", fontsize=14)
        axes[1, 0].set_ylabel("R² Score")
        axes[1, 0].grid(alpha=0.3, axis="y")

        sns.barplot(
            data=comparison_df,
            x="Model",
            y="Training Time (s)",
            hue="Model",
            ax=axes[1, 1],
            palette="Set2",
            legend=False,
        )
        axes[1, 1].set_title("Training Time Comparison", fontsize=14)
        axes[1, 1].set_ylabel("Time (s)")
        axes[1, 1].grid(alpha=0.3, axis="y")

        plt.suptitle("Model Performance Comparison", fontsize=16, y=0.995)
        plt.tight_layout()
        filepath = self.output_dir / "model_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

        self._plot_per_thruster_comparison(models_results, metadata, test_indices)
        self._plot_feature_importance_comparison(models_results)

    def _plot_per_thruster_comparison(
        self,
        models_results: dict,
        metadata: pd.DataFrame,
        test_indices: list,
    ):
        """Compare per-thruster performance across models"""
        all_results = []

        sn_array = np.array([metadata.loc[idx, "sn"] for idx in test_indices])

        for model_name, result in models_results.items():
            y_true = result["y_true"]
            y_pred = result["y_pred"]

            for sn in range(13, 25):
                sn_mask = sn_array == sn
                if sn_mask.sum() == 0:
                    continue

                y_true_sn = y_true[sn_mask]
                y_pred_sn = y_pred[sn_mask]

                rmse_sn = np.sqrt(mean_squared_error(y_true_sn, y_pred_sn))
                mae_sn = mean_absolute_error(y_true_sn, y_pred_sn)
                r2_sn = r2_score(y_true_sn, y_pred_sn)

                all_results.append(
                    {
                        "Model": model_name,
                        "SN": f"SN{sn:02d}",
                        "RMSE": rmse_sn,
                        "MAE": mae_sn,
                        "R²": r2_sn,
                    }
                )

        results_df = pd.DataFrame(all_results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(data=results_df, x="SN", y="RMSE", hue="Model", ax=axes[0], palette="Set2")
        axes[0].set_title("RMSE per Thruster - Model Comparison", fontsize=14)
        axes[0].set_ylabel("RMSE (N)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(alpha=0.3, axis="y")
        axes[0].get_legend().remove()

        sns.barplot(data=results_df, x="SN", y="MAE", hue="Model", ax=axes[1], palette="Set2")
        axes[1].set_title("MAE per Thruster - Model Comparison", fontsize=14)
        axes[1].set_ylabel("MAE (N)")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].get_legend().remove()

        sns.barplot(data=results_df, x="SN", y="R²", hue="Model", ax=axes[2], palette="Set2")
        axes[2].set_title("R² per Thruster - Model Comparison", fontsize=14)
        axes[2].set_ylabel("R² Score")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(alpha=0.3, axis="y")
        axes[2].get_legend().remove()

        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            ncol=len(models_results),
            bbox_to_anchor=(0.5, 1.02),
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        filepath = self.output_dir / "per_thruster_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def _plot_feature_importance_comparison(self, models_results: dict):
        """Compare feature importance across models"""
        feature_names = [
            "ton[t]",
            "thrust[t-2]",
            "thrust[t-3]",
            "thrust[t-4]",
            "thrust[t-5]",
            "thrust[t-6]",
            "thrust[t-7]",
            "thrust[t-8]",
            "thrust[t-9]",
            "thrust[t-10]",
            "ton[t-3]",
            "ton[t-5]",
            "mfr[t-1]",
            "rolling_mean",
            "rolling_std",
            "pressure",
            "on_time",
            "pulses",
        ]

        fig, axes = plt.subplots(1, len(models_results), figsize=(8 * len(models_results), 10))
        if len(models_results) == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(models_results.items()):
            importance = result["predictor"].get_feature_importance()

            if importance is not None:
                sorted_idx = np.argsort(importance)
                sorted_importance = importance[sorted_idx]
                sorted_names = [feature_names[i] for i in sorted_idx]

                axes[idx].barh(sorted_names, sorted_importance, color="#1f77b4")
                axes[idx].set_xlabel("Importance")
                axes[idx].set_title(f"Feature Importance - {model_name}", fontsize=14)
                axes[idx].grid(alpha=0.3, axis="x")
            else:
                axes[idx].text(
                    0.5,
                    0.5,
                    "Feature importance not available",
                    ha="center",
                    va="center",
                    transform=axes[idx].transAxes,
                )
                axes[idx].set_title(f"Feature Importance - {model_name}", fontsize=14)

        plt.tight_layout()
        filepath = self.output_dir / "feature_importance_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()
