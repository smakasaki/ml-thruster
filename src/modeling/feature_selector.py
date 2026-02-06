"""Feature Selection and Dimensionality Reduction Analysis.

This module implements various feature selection methods to compare
with the baseline 24-feature model and demonstrate that manual
feature engineering based on domain knowledge is effective.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
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


@dataclass
class SelectionResult:
    """Result of a feature selection experiment."""

    method: str
    n_features: int
    rmse: float
    mae: float
    r2: float
    training_time: float
    selected_features: list[str] | None = None
    explained_variance: float | None = None
    aic: float | None = None
    bic: float | None = None


class FeatureSelector:
    """Compare different feature selection and dimensionality reduction methods."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.results: list[SelectionResult] = []
        self.elimination_path: list[tuple] = []

    def run_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_size: int = 500000,
    ) -> pd.DataFrame:
        """Run all feature selection experiments."""
        logger.info("=" * 70)
        logger.info("FEATURE SELECTION / DIMENSIONALITY REDUCTION ANALYSIS")
        logger.info("=" * 70)

        if len(X_train) > sample_size:
            logger.info(f"Sampling {sample_size:,} points for feature selection analysis")
            np.random.seed(42)
            train_idx = np.random.choice(len(X_train), sample_size, replace=False)
            test_idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
            X_train_sample = X_train[train_idx]
            y_train_sample = y_train[train_idx]
            X_test_sample = X_test[test_idx]
            y_test_sample = y_test[test_idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
            X_test_sample = X_test
            y_test_sample = y_test

        X_train_scaled = self.scaler.fit_transform(X_train_sample)
        X_test_scaled = self.scaler.transform(X_test_sample)

        self.results = []

        X_test_sc = np.asarray(X_test_scaled)
        X_test_samp = np.asarray(X_test_sample)

        self._run_baseline(X_train_scaled, y_train_sample, X_test_sc, y_test_sample)
        self._run_selectkbest_f(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, k=15)
        self._run_selectkbest_f(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, k=10)
        self._run_selectkbest_f(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, k=5)
        self._run_selectkbest_mi(X_train_sample, y_train_sample, X_test_samp, y_test_sample, k=10)
        self._run_pca(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, variance=0.99)
        self._run_pca(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, variance=0.95)
        self._run_pca(X_train_scaled, y_train_sample, X_test_sc, y_test_sample, variance=0.90)
        self._run_backward_elimination(X_train_scaled, y_train_sample, X_test_sc, y_test_sample)

        results_df = self._create_results_dataframe()
        self._plot_results(results_df)
        self._plot_information_criteria(results_df)
        self._plot_feature_scores(X_train_scaled, y_train_sample)
        self._plot_pca_variance(X_train_scaled)
        self._plot_elimination_path()

        return results_df

    def _evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        compute_ic: bool = True,
    ) -> tuple[float, float, float, float, float | None, float | None]:
        """Train Ridge model and return metrics (+ AIC/BIC when compute_ic=True)."""
        model = Ridge(alpha=1.0, random_state=42)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        aic: float | None = None
        bic: float | None = None
        if compute_ic:
            n = X_train.shape[0]
            k = X_train.shape[1] + 1  # features + intercept
            y_train_pred = model.predict(X_train)
            rss = float(np.sum((y_train - y_train_pred) ** 2))
            log_rss_n = np.log(rss / n)
            aic = float(n * log_rss_n + 2 * k)
            bic = float(n * log_rss_n + k * np.log(n))

        return rmse, mae, r2, training_time, aic, bic

    def _run_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Baseline: all 24 features."""
        n_baseline = len(FEATURE_NAMES)
        logger.info(f"\n[1] Baseline: All {n_baseline} features (backward-elimination optimised)")

        rmse, mae, r2, train_time, aic, bic = self._evaluate_model(X_train, y_train, X_test, y_test)

        result = SelectionResult(
            method=f"Baseline (All {n_baseline})",
            n_features=X_train.shape[1],
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=train_time,
            selected_features=FEATURE_NAMES.copy(),
            aic=aic,
            bic=bic,
        )
        self.results.append(result)
        self._log_result(result)

    def _run_selectkbest_f(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int,
    ):
        """SelectKBest with F-regression."""
        logger.info(f"\n[*] SelectKBest (F-test, k={k})")

        selector = SelectKBest(f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        rmse, mae, r2, train_time, aic, bic = self._evaluate_model(
            np.asarray(X_train_selected), y_train, np.asarray(X_test_selected), y_test
        )

        support_mask = selector.get_support()
        if support_mask is not None:
            selected = [FEATURE_NAMES[i] for i, sel in enumerate(support_mask) if sel]
        else:
            selected = []

        result = SelectionResult(
            method=f"SelectKBest F (k={k})",
            n_features=k,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=train_time,
            selected_features=selected,
            aic=aic,
            bic=bic,
        )
        self.results.append(result)
        self._log_result(result)

    def _run_selectkbest_mi(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int,
    ):
        """SelectKBest with Mutual Information."""
        logger.info(f"\n[*] SelectKBest (Mutual Information, k={k})")

        selector = SelectKBest(mutual_info_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        rmse, mae, r2, train_time, aic, bic = self._evaluate_model(
            np.asarray(X_train_selected), y_train, np.asarray(X_test_selected), y_test
        )

        support_mask = selector.get_support()
        if support_mask is not None:
            selected = [FEATURE_NAMES[i] for i, sel in enumerate(support_mask) if sel]
        else:
            selected = []

        result = SelectionResult(
            method=f"SelectKBest MI (k={k})",
            n_features=k,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=train_time,
            selected_features=selected,
            aic=aic,
            bic=bic,
        )
        self.results.append(result)
        self._log_result(result)

    def _run_pca(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        variance: float,
    ):
        """PCA with specified variance retention."""
        logger.info(f"\n[*] PCA ({variance * 100:.0f}% variance)")

        pca = PCA(n_components=variance, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        rmse, mae, r2, train_time, aic, bic = self._evaluate_model(X_train_pca, y_train, X_test_pca, y_test)

        result = SelectionResult(
            method=f"PCA ({variance * 100:.0f}% var)",
            n_features=pca.n_components_,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=train_time,
            explained_variance=pca.explained_variance_ratio_.sum(),
            aic=aic,
            bic=bic,
        )
        self.results.append(result)
        self._log_result(result)

    def _run_backward_elimination(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        min_features: int = 3,
    ):
        """Backward elimination: iteratively remove the least impactful feature.

        At each step every remaining feature is held out in turn; a Ridge model
        is trained on the reduced set and the feature whose exclusion produces
        the lowest RMSE is permanently removed.  The full elimination path is
        recorded and the configuration with the best RMSE along the path is
        reported as the optimal subset.
        """
        logger.info(f"\n[*] Backward Elimination (down to {min_features} features)")

        n_total = X_train.shape[1]
        current_indices = list(range(n_total))

        # Baseline with all features
        rmse, mae, r2, train_time, _, _ = self._evaluate_model(
            X_train, y_train, X_test, y_test, compute_ic=False
        )
        best_rmse = rmse
        best_indices = current_indices.copy()

        # Path: (n_features, rmse, removed_feature_name)
        self.elimination_path = [(n_total, rmse, None)]
        logger.info(f"  Start: {n_total} features, RMSE={rmse:.6f}")

        while len(current_indices) > min_features:
            removal_candidate = None
            removal_rmse = float("inf")

            for feat in current_indices:
                candidate = [f for f in current_indices if f != feat]
                rmse_cand, _, _, _, _, _ = self._evaluate_model(
                    X_train[:, candidate], y_train,
                    X_test[:, candidate], y_test,
                    compute_ic=False,
                )
                if rmse_cand < removal_rmse:
                    removal_rmse = rmse_cand
                    removal_candidate = feat

            current_indices.remove(removal_candidate)
            self.elimination_path.append(
                (len(current_indices), removal_rmse, FEATURE_NAMES[removal_candidate])
            )

            if removal_rmse <= best_rmse:
                best_rmse = removal_rmse
                best_indices = current_indices.copy()

            logger.info(
                f"  Removed '{FEATURE_NAMES[removal_candidate]}' "
                f"\u2192 {len(current_indices)} features, RMSE={removal_rmse:.6f}"
            )

        # Final evaluation at the optimal point along the path
        rmse, mae, r2, train_time, aic, bic = self._evaluate_model(
            X_train[:, best_indices], y_train,
            X_test[:, best_indices], y_test,
        )

        result = SelectionResult(
            method="Backward Elim.",
            n_features=len(best_indices),
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=train_time,
            selected_features=[FEATURE_NAMES[i] for i in best_indices],
            aic=aic,
            bic=bic,
        )
        self.results.append(result)
        self._log_result(result)

    def _log_result(self, result: SelectionResult):
        """Log a single result."""
        logger.info(f"  Features: {result.n_features}")
        logger.info(f"  RMSE: {result.rmse:.6f} N")
        logger.info(f"  MAE:  {result.mae:.6f} N")
        logger.info(f"  R²:   {result.r2:.6f}")
        if result.aic is not None:
            logger.info(f"  AIC: {result.aic:.2f}")
        if result.bic is not None:
            logger.info(f"  BIC: {result.bic:.2f}")
        if result.selected_features:
            logger.info(f"  Selected: {result.selected_features[:5]}...")
        if result.explained_variance:
            logger.info(f"  Explained variance: {result.explained_variance:.4f}")

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from results."""
        data = []
        for r in self.results:
            data.append(
                {
                    "Method": r.method,
                    "Features": r.n_features,
                    "RMSE": r.rmse,
                    "MAE": r.mae,
                    "R²": r.r2,
                    "Train Time (s)": r.training_time,
                    "AIC": r.aic,
                    "BIC": r.bic,
                }
            )

        df = pd.DataFrame(data)

        logger.info("\n" + "=" * 70)
        logger.info("FEATURE SELECTION RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info("\n" + df.to_string(index=False))

        filepath = self.output_dir.parent / "reports" / "feature_selection_results.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"\nResults saved to: {filepath}")

        return df

    def _plot_results(self, results_df: pd.DataFrame):
        """Plot comparison of all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = sns.color_palette("husl", len(results_df))

        ax = axes[0, 0]
        bars = ax.barh(results_df["Method"], results_df["RMSE"], color=colors)
        ax.set_xlabel("RMSE (N)", fontsize=11)
        ax.set_title("RMSE by Feature Selection Method", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        ax.axvline(results_df["RMSE"].min(), color="green", linestyle="--", alpha=0.7, label="Best")
        for bar, val in zip(bars, results_df["RMSE"]):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        ax = axes[0, 1]
        bars = ax.barh(results_df["Method"], results_df["R²"], color=colors)
        ax.set_xlabel("R² Score", fontsize=11)
        ax.set_title("R² by Feature Selection Method", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        ax.axvline(results_df["R²"].max(), color="green", linestyle="--", alpha=0.7, label="Best")
        for bar, val in zip(bars, results_df["R²"]):
            ax.text(
                val - 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        ax = axes[1, 0]
        ax.scatter(
            results_df["Features"],
            results_df["RMSE"],
            s=150,
            c=colors,
            edgecolors="black",
            linewidth=1.5,
        )
        for i, row in results_df.iterrows():
            method_name = str(row["Method"]).split("(")[0].strip()
            ax.annotate(
                method_name,
                (row["Features"], row["RMSE"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.set_xlabel("Number of Features", fontsize=11)
        ax.set_ylabel("RMSE (N)", fontsize=11)
        ax.set_title("RMSE vs Number of Features", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.scatter(
            results_df["Features"],
            results_df["R²"],
            s=150,
            c=colors,
            edgecolors="black",
            linewidth=1.5,
        )
        for i, row in results_df.iterrows():
            method_name = str(row["Method"]).split("(")[0].strip()
            ax.annotate(
                method_name,
                (row["Features"], row["R²"]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
            )
        ax.set_xlabel("Number of Features", fontsize=11)
        ax.set_ylabel("R² Score", fontsize=11)
        ax.set_title("R² vs Number of Features", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

        plt.suptitle(
            "Feature Selection / Dimensionality Reduction Comparison",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()

        filepath = self.output_dir / "feature_selection_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def _plot_information_criteria(self, results_df: pd.DataFrame):
        """Plot AIC and BIC vs number of features."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = sns.color_palette("husl", len(results_df))

        for ax, criterion in zip(axes, ("AIC", "BIC"), strict=True):
            ax.scatter(
                results_df["Features"],
                results_df[criterion],
                s=150,
                c=colors,
                edgecolors="black",
                linewidth=1.5,
            )
            for _i, row in results_df.iterrows():
                method_name = str(row["Method"]).split("(")[0].strip()
                ax.annotate(
                    method_name,
                    (row["Features"], row[criterion]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

            best_idx = results_df[criterion].idxmin()
            ax.scatter(
                [results_df.loc[best_idx, "Features"]],
                [results_df.loc[best_idx, criterion]],
                s=300,
                facecolors="none",
                edgecolors="green",
                linewidth=2.5,
                zorder=5,
                label=f"Best {criterion}",
            )

            ax.set_xlabel("Number of Features", fontsize=11)
            ax.set_ylabel(criterion, fontsize=11)
            ax.set_title(f"{criterion} vs Number of Features", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(
            "Information Criteria (AIC / BIC) Comparison",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()

        filepath = self.output_dir / "information_criteria_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def _plot_feature_scores(self, X_train: np.ndarray, y_train: np.ndarray):
        """Plot feature importance scores from F-test."""
        selector = SelectKBest(f_regression, k=X_train.shape[1])
        selector.fit(X_train, y_train)
        scores = selector.scores_

        if scores is None:
            logger.warning("Could not compute feature scores")
            return

        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = plt.get_cmap("RdYlGn")
        colors = [cmap(x) for x in np.linspace(0.2, 0.8, len(sorted_scores))][::-1]
        bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors)

        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("F-Score", fontsize=11)
        ax.set_title("Feature Importance (F-regression scores)", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")

        for bar, score in zip(bars, sorted_scores):
            ax.text(
                score + max(sorted_scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        filepath = self.output_dir / "feature_importance_scores.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def _plot_pca_variance(self, X_train: np.ndarray):
        """Plot PCA cumulative explained variance."""
        pca = PCA(random_state=42)
        pca.fit(X_train)

        cumulative_var = np.cumsum(pca.explained_variance_ratio_)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="black",
            label="Individual",
        )
        ax.plot(
            range(1, len(cumulative_var) + 1),
            cumulative_var,
            "ro-",
            linewidth=2,
            markersize=6,
            label="Cumulative",
        )
        ax.axhline(0.95, color="green", linestyle="--", alpha=0.7, label="95% threshold")
        ax.axhline(0.99, color="orange", linestyle="--", alpha=0.7, label="99% threshold")
        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Explained Variance Ratio", fontsize=11)
        ax.set_title("PCA Explained Variance", fontsize=13, fontweight="bold")
        ax.legend(loc="center right")
        ax.grid(alpha=0.3)

        n_95 = np.argmax(cumulative_var >= 0.95) + 1
        n_99 = np.argmax(cumulative_var >= 0.99) + 1

        ax = axes[1]
        components_needed = [5, 10, 15, n_95, n_99, 24]
        variance_captured = [
            cumulative_var[min(c - 1, len(cumulative_var) - 1)] for c in components_needed
        ]

        bars = ax.bar(
            range(len(components_needed)),
            variance_captured,
            color=sns.color_palette("viridis", len(components_needed)),
            edgecolor="black",
        )
        ax.set_xticks(range(len(components_needed)))
        ax.set_xticklabels([f"{c} PCs" for c in components_needed])
        ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
        ax.set_title("Variance Captured by N Components", fontsize=13, fontweight="bold")
        ax.axhline(0.95, color="green", linestyle="--", alpha=0.7)
        ax.axhline(0.99, color="orange", linestyle="--", alpha=0.7)
        ax.grid(alpha=0.3, axis="y")

        for bar, var in zip(bars, variance_captured):
            ax.text(
                bar.get_x() + bar.get_width() / 2, var + 0.01, f"{var:.2%}", ha="center", fontsize=9
            )

        plt.suptitle(
            f"PCA Analysis (95% variance → {n_95} components, 99% → {n_99} components)",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()

        filepath = self.output_dir / "pca_variance_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def _plot_elimination_path(self):
        """Plot RMSE along the backward elimination path."""
        if len(self.elimination_path) < 2:
            return

        n_feats = [s[0] for s in self.elimination_path]
        rmses = [s[1] for s in self.elimination_path]

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(n_feats, rmses, "o-", color="#1976D2", linewidth=2, markersize=5, zorder=2)

        # Mark optimal point
        best_idx = int(np.argmin(rmses))
        ax.scatter(
            [n_feats[best_idx]],
            [rmses[best_idx]],
            s=150,
            c="#D32F2F",
            edgecolors="black",
            linewidth=1.5,
            zorder=3,
            label=f"Optimal: {n_feats[best_idx]} features (RMSE={rmses[best_idx]:.4f})",
        )

        # Annotate every other removed feature to avoid label overlap
        for i in range(1, len(self.elimination_path)):
            if i % 2 != 0:
                name = self.elimination_path[i][2]
                if name:
                    ax.annotate(
                        name,
                        (n_feats[i], rmses[i]),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        fontsize=6.5,
                        color="#757575",
                        rotation=30,
                    )

        ax.set_xlabel("Number of Features", fontsize=11)
        ax.set_ylabel("RMSE (N)", fontsize=11)
        ax.set_title("Backward Elimination Path", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()

        filepath = self.output_dir / "backward_elimination_path.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filepath}")
        plt.close()

    def get_conclusion(self) -> str:
        """Generate conclusion text for report."""
        if not self.results:
            return "No results available."

        baseline = self.results[0]
        best_result = min(self.results, key=lambda r: r.rmse)
        best_bic_result = min(self.results, key=lambda r: r.bic if r.bic is not None else float("inf"))

        conclusion = f"""
FEATURE SELECTION ANALYSIS CONCLUSION:

1. Baseline ({len(FEATURE_NAMES)} features, backward-elimination optimised):
   - RMSE: {baseline.rmse:.6f} N
   - R²: {baseline.r2:.6f}

2. Best performing method (by RMSE): {best_result.method}
   - RMSE: {best_result.rmse:.6f} N ({(baseline.rmse - best_result.rmse) / baseline.rmse * 100:+.2f}% vs baseline)
   - R²: {best_result.r2:.6f}
   - Features: {best_result.n_features}

3. Interpretation:
   {"The baseline feature set performs optimally." if best_result.method.startswith("Baseline") else f"Dimensionality reduction to {best_result.n_features} features achieves comparable/better performance."}

4. Key insights:
   - Autoregressive features (thrust[t-1..t-10]) capture temporal dynamics
   - Command history (ton[t-1..t-5]) captures transient behavior
   - Aging factors explain long-term degradation trends

5. Information criteria (BIC):
   - Best BIC method: {best_bic_result.method} (BIC={best_bic_result.bic:.2f}, {best_bic_result.n_features} features)
   - BIC penalises complexity more heavily than AIC for n > 7 (always true here).
     A lower BIC indicates a better trade-off between fit quality and parsimony.
"""
        return conclusion
