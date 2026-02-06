from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats


class Visualizer:
    """
    Comprehensive visualization toolkit for STFT dataset exploratory analysis.
    Optimized for LaTeX A4 documents with professional, readable plots.
    """

    FIGURE_DPI = 300
    FIGURE_SIZE_SMALL = (7, 5)
    FIGURE_SIZE_MEDIUM = (8, 6)
    FIGURE_SIZE_LARGE = (10, 7)
    FIGURE_SIZE_WIDE = (10, 5)

    HISTOGRAM_BINS = 50
    HISTOGRAM_BINS_COMPARISON = 30
    HISTOGRAM_ALPHA = 0.7
    HISTOGRAM_ALPHA_COMPARISON = 0.6
    HISTOGRAM_EDGECOLOR = "black"

    SCATTER_ALPHA = 0.6
    SCATTER_SIZE = 30

    COLOR_PRIMARY = "#1f77b4"
    COLOR_SECONDARY = "#aec7e8"
    COLOR_ACCENT = "#ff7f0e"
    COLOR_TRAIN = "#1f77b4"
    COLOR_TEST = "#ff7f0e"
    COLOR_MEAN = "#d62728"
    COLOR_MEDIAN = "#2ca02c"
    COLOR_TREND = "#2ca02c"
    COLOR_BOUNDARY = "#d62728"

    GRID_ALPHA = 0.3
    LINE_WIDTH = 2
    LINE_WIDTH_THIN = 1.5
    MARKER_SIZE = 4

    TRAIN_MARKER = "o"
    TEST_MARKER = "s"
    TRAIN_ALPHA = 0.7
    TEST_ALPHA = 0.6
    TRAIN_LINE_STYLE = "-"
    TEST_LINE_STYLE = "--"

    LABEL_ROTATION = 45
    POLYNOMIAL_DEGREE = 1
    TREND_LINE_POINTS = 100

    EXCLUDE_COLS = ["uid", "test_id", "sn"]
    EXCLUDE_COLS_CORRELATION = ["uid", "test_id"]

    AGING_FACTORS = ["cumulated_throughput", "cumulated_on_time", "cumulated_pulses"]
    PERFORMANCE_METRICS = ["thrust_mean", "mfr_mean"]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-paper")
        sns.set_context("paper", font_scale=1.2)
        sns.set_palette("colorblind")

        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        plt.rcParams["legend.fontsize"] = 9
        plt.rcParams["figure.titlesize"] = 13

        logger.info(f"Visualizer initialized. Output directory: {output_dir}")

    def _format_label(self, text: str) -> str:
        return text.replace("_", " ").title()

    def _add_grid(self, ax, axis: str = "both") -> None:
        ax.grid(axis=axis, alpha=self.GRID_ALPHA, linestyle=":", linewidth=0.5)

    def _add_mean_median_lines(self, ax, data: pd.Series) -> None:
        ax.axvline(
            data.mean(),
            color=self.COLOR_MEAN,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Mean",
        )
        ax.axvline(
            data.median(),
            color=self.COLOR_MEDIAN,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Median",
        )

    def _style_boxplot(self, box_plot_dict: dict) -> None:
        for box in box_plot_dict["boxes"]:
            box.set_facecolor(self.COLOR_SECONDARY)
            box.set_edgecolor(self.COLOR_PRIMARY)
            box.set_linewidth(1.5)

    def _add_trend_line(self, ax, x_data: pd.Series, y_data: pd.Series) -> None:
        x_clean = x_data.dropna()
        y_clean = y_data.dropna()
        common_idx = x_clean.index.intersection(y_clean.index)

        if len(common_idx) < 2:
            return

        z = np.polyfit(x_clean[common_idx], y_clean[common_idx], self.POLYNOMIAL_DEGREE)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), self.TREND_LINE_POINTS)
        ax.plot(
            x_line,
            p(x_line),
            color=self.COLOR_TREND,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}",
            alpha=0.8,
        )

    def _add_train_test_scatter(self, ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        train_data = df[df["split"] == "train"]
        test_data = df[df["split"] == "test"]

        ax.scatter(
            train_data[x_col],
            train_data[y_col],
            alpha=self.SCATTER_ALPHA,
            s=self.SCATTER_SIZE,
            label="Train (SN01-12)",
            color=self.COLOR_TRAIN,
            edgecolors="black",
            linewidths=0.5,
        )

        ax.scatter(
            test_data[x_col],
            test_data[y_col],
            alpha=self.SCATTER_ALPHA,
            s=self.SCATTER_SIZE,
            label="Test (SN13-24)",
            color=self.COLOR_TEST,
            edgecolors="black",
            linewidths=0.5,
        )

    def create_all_visualizations(self, df: pd.DataFrame) -> None:
        logger.info("Creating all visualizations for Midterm 1")

        # Core visualizations for report
        self.plot_distributions_and_statistics(df)
        self.plot_boxplots_comprehensive(df)
        self.plot_correlation_analysis(df)
        self.plot_aging_effects(df)
        self.plot_pressure_vs_performance(df)
        self.plot_train_test_comparison(df)

        # Additional exploratory plots
        self.plot_qq_plots(df)
        self.plot_pairplot_key_features(df)
        self.plot_performance_degradation_by_sn(df)

        # Optional plots (if data available)
        self.plot_test_mode_analysis(df)
        self.plot_anomaly_analysis_improved(df)

        logger.info("All visualizations created successfully")

    def plot_distributions_and_statistics(self, df: pd.DataFrame) -> None:
        """Histograms with statistical markers - REQUIRED for report Section 2"""
        logger.info("Creating distribution plots")

        for performance_metric in self.PERFORMANCE_METRICS:
            if performance_metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_SMALL)

            data = df[performance_metric].dropna()
            ax.hist(
                data,
                bins=self.HISTOGRAM_BINS,
                alpha=self.HISTOGRAM_ALPHA,
                color=self.COLOR_PRIMARY,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )

            self._add_mean_median_lines(ax, data)

            skewness = data.skew()
            ax.text(
                0.95,
                0.95,
                f"Mean: {data.mean():.3f}\nMedian: {data.median():.3f}\nStd: {data.std():.3f}\nSkew: {skewness:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=8,
            )

            ax.set_xlabel(self._format_label(performance_metric))
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {self._format_label(performance_metric)}")
            ax.legend()
            self._add_grid(ax, axis="y")

            plt.tight_layout()
            self._save_figure(f"01_distribution_{performance_metric}")
            plt.close()

        logger.info("Distribution plots created")

    def plot_boxplots_comprehensive(self, df: pd.DataFrame) -> None:
        """
        Comprehensive box plots for all key variables.
        REQUIRED for report Section 2 - showing outliers and distributions.
        """
        logger.info("Creating comprehensive box plots")

        # Box plots by split (train vs test)
        key_vars = ["thrust_mean", "mfr_mean", "test_pressure"] + self.AGING_FACTORS
        available_vars = [v for v in key_vars if v in df.columns]

        if len(available_vars) > 0:
            n_vars = len(available_vars)
            n_cols = 3
            n_rows = (n_vars + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 3))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            for idx, var in enumerate(available_vars):
                ax = axes[idx]

                data_to_plot = [
                    df[df["split"] == "train"][var].dropna(),
                    df[df["split"] == "test"][var].dropna(),
                ]

                bp = ax.boxplot(
                    data_to_plot,
                    labels=["Train", "Test"],
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                )

                for patch, color in zip(bp["boxes"], [self.COLOR_TRAIN, self.COLOR_TEST]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

                ax.set_ylabel(self._format_label(var))
                ax.set_title(f"{self._format_label(var)}")
                self._add_grid(ax, axis="y")

            for idx in range(len(available_vars), len(axes)):
                fig.delaxes(axes[idx])

            plt.suptitle("Distribution Comparison: Train vs Test", fontsize=14, y=1.00)
            plt.tight_layout()
            self._save_figure("02_boxplots_train_test")
            plt.close()

        # Box plots by test pressure
        if "test_pressure" in df.columns and "thrust_mean" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_LARGE)

            for idx, metric in enumerate(["thrust_mean", "mfr_mean"]):
                if metric not in df.columns:
                    continue

                ax = axes[idx]

                pressures = sorted(df["test_pressure"].unique())
                data_by_pressure = [
                    df[df["test_pressure"] == p][metric].dropna() for p in pressures
                ]

                bp = ax.boxplot(
                    data_by_pressure,
                    labels=[f"{int(p)}" for p in pressures],
                    patch_artist=True,
                    showmeans=True,
                )

                self._style_boxplot(bp)

                ax.set_xlabel("Test Pressure (bars)")
                ax.set_ylabel(self._format_label(metric))
                ax.set_title(f"{self._format_label(metric)} by Pressure")
                self._add_grid(ax, axis="y")

            plt.tight_layout()
            self._save_figure("03_boxplots_by_pressure")
            plt.close()

        logger.info("Comprehensive box plots created")

    def plot_correlation_analysis(self, df: pd.DataFrame) -> None:
        """
        Correlation heatmap - REQUIRED for report Section 3.
        Shows relationships between predictors.
        """
        logger.info("Creating correlation analysis plots")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        numeric_cols = [col for col in numeric_cols if col not in self.EXCLUDE_COLS_CORRELATION]

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return

        corr_matrix = df[numeric_cols].corr()

        # Create readable heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            ax=ax,
            annot_kws={"fontsize": 7},
        )

        ax.set_title("Feature Correlation Matrix (Lower Triangle)")
        plt.tight_layout()
        self._save_figure("04_correlation_heatmap")
        plt.close()

        logger.info("Correlation analysis plots created")

    def plot_aging_effects(self, df: pd.DataFrame) -> None:
        """
        Scatter plots with trend lines - REQUIRED for report Section 3.
        Shows predictor relationships with target variables.
        """
        logger.info("Creating aging effects plots")

        for aging_factor in self.AGING_FACTORS:
            if aging_factor not in df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_LARGE)

            for idx, performance_metric in enumerate(self.PERFORMANCE_METRICS):
                if performance_metric not in df.columns:
                    continue

                ax = axes[idx]

                self._add_train_test_scatter(ax, df, aging_factor, performance_metric)
                self._add_trend_line(ax, df[aging_factor], df[performance_metric])

                # Calculate R²
                x_clean = df[aging_factor].dropna()
                y_clean = df[performance_metric].dropna()
                common_idx = x_clean.index.intersection(y_clean.index)
                if len(common_idx) > 1:
                    r_squared = np.corrcoef(x_clean[common_idx], y_clean[common_idx])[0, 1] ** 2
                    ax.text(
                        0.05,
                        0.95,
                        f"R² = {r_squared:.3f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                        fontsize=9,
                    )

                ax.set_xlabel(self._format_label(aging_factor))
                ax.set_ylabel(self._format_label(performance_metric))
                ax.set_title(f"{self._format_label(performance_metric)}")
                ax.legend(fontsize=8)
                self._add_grid(ax)

            plt.suptitle(f"Performance vs {self._format_label(aging_factor)}", fontsize=13)
            plt.tight_layout()
            self._save_figure(f"05_{aging_factor}_vs_performance")
            plt.close()

        logger.info("Aging effects plots created")

    def plot_pressure_vs_performance(self, df: pd.DataFrame) -> None:
        """Pressure effect analysis - important predictor."""
        logger.info("Creating pressure vs performance plots")

        if "test_pressure" not in df.columns:
            logger.warning("test_pressure column not found")
            return

        fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_LARGE)

        for idx, performance_metric in enumerate(self.PERFORMANCE_METRICS):
            if performance_metric not in df.columns:
                continue

            ax = axes[idx]

            scatter = ax.scatter(
                df["test_pressure"],
                df[performance_metric],
                c=df["cumulated_throughput"],
                cmap="viridis",
                alpha=self.SCATTER_ALPHA,
                s=self.SCATTER_SIZE,
                edgecolors="black",
                linewidths=0.5,
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Throughput (kg)", fontsize=9)

            ax.set_xlabel("Test Pressure (bars)")
            ax.set_ylabel(self._format_label(performance_metric))
            ax.set_title(f"{self._format_label(performance_metric)}")
            self._add_grid(ax)

        plt.suptitle("Pressure Effect on Performance (colored by aging)", fontsize=13)
        plt.tight_layout()
        self._save_figure("06_pressure_vs_performance")
        plt.close()

        logger.info("Pressure vs performance plots created")

    def plot_train_test_comparison(self, df: pd.DataFrame) -> None:
        """
        Train/test distribution comparison - REQUIRED for validation.
        Shows if test set is representative.
        """
        logger.info("Creating train/test comparison plots")

        if "split" not in df.columns:
            logger.warning("split column not found")
            return

        comparison_features = [
            "test_pressure",
            "cumulated_throughput",
            "thrust_mean",
            "mfr_mean",
        ]

        fig, axes = plt.subplots(2, 2, figsize=self.FIGURE_SIZE_LARGE)
        axes = axes.flatten()

        for idx, feature in enumerate(comparison_features):
            if feature not in df.columns:
                continue

            ax = axes[idx]

            train_data = df[df["split"] == "train"][feature].dropna()
            test_data = df[df["split"] == "test"][feature].dropna()

            ax.hist(
                train_data,
                bins=self.HISTOGRAM_BINS_COMPARISON,
                alpha=self.HISTOGRAM_ALPHA_COMPARISON,
                label="Train",
                color=self.COLOR_TRAIN,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )
            ax.hist(
                test_data,
                bins=self.HISTOGRAM_BINS_COMPARISON,
                alpha=self.HISTOGRAM_ALPHA_COMPARISON,
                label="Test",
                color=self.COLOR_TEST,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )

            ax.axvline(
                train_data.mean(),
                color=self.COLOR_TRAIN,
                linestyle="--",
                linewidth=self.LINE_WIDTH_THIN,
                alpha=0.8,
            )
            ax.axvline(
                test_data.mean(),
                color=self.COLOR_TEST,
                linestyle="--",
                linewidth=self.LINE_WIDTH_THIN,
                alpha=0.8,
            )

            ax.set_title(f"{self._format_label(feature)}", fontsize=11)
            ax.set_xlabel(self._format_label(feature), fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8)
            self._add_grid(ax)

        plt.suptitle("Train vs Test Set Comparison", fontsize=13)
        plt.tight_layout()
        self._save_figure("07_train_test_comparison")
        plt.close()

        logger.info("Train/test comparison plots created")

    def plot_qq_plots(self, df: pd.DataFrame) -> None:
        """QQ plots to check normality assumption - useful for exploration."""
        logger.info("Creating QQ plots")

        fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_LARGE)

        for idx, metric in enumerate(self.PERFORMANCE_METRICS):
            if metric not in df.columns:
                continue

            ax = axes[idx]
            data = df[metric].dropna()

            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {self._format_label(metric)}")
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=":", linewidth=0.5)

        plt.tight_layout()
        self._save_figure("08_qq_plots")
        plt.close()

        logger.info("QQ plots created")

    def plot_pairplot_key_features(self, df: pd.DataFrame) -> None:
        """Pairplot for key features - comprehensive relationship view."""
        logger.info("Creating pairplot")

        key_features = ["thrust_mean", "mfr_mean", "test_pressure", "cumulated_throughput"]
        available = [f for f in key_features if f in df.columns]

        if len(available) < 2 or "split" not in df.columns:
            logger.warning("Not enough features for pairplot")
            return

        sample_size = min(500, len(df))
        df_sample = df[available + ["split"]].sample(n=sample_size, random_state=42)

        g = sns.pairplot(
            df_sample,
            hue="split",
            palette={"train": self.COLOR_TRAIN, "test": self.COLOR_TEST},
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 20, "edgecolor": "black", "linewidth": 0.5},
            diag_kws={"alpha": 0.7, "edgecolor": "black"},
        )

        g.fig.suptitle("Pairwise Relationships of Key Features", y=1.02, fontsize=13)
        plt.tight_layout()
        self._save_figure("09_pairplot")
        plt.close()

        logger.info("Pairplot created")

    def plot_performance_degradation_by_sn(self, df: pd.DataFrame) -> None:
        """Performance degradation trends by serial number."""
        logger.info("Creating performance degradation plots")

        for performance_metric in self.PERFORMANCE_METRICS:
            if performance_metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_WIDE)

            train_sns = sorted(df[df["split"] == "train"]["sn"].unique())
            test_sns = sorted(df[df["split"] == "test"]["sn"].unique())

            colors_train = plt.cm.Blues(np.linspace(0.4, 0.9, len(train_sns)))
            colors_test = plt.cm.Oranges(np.linspace(0.4, 0.9, len(test_sns)))

            for idx, sn in enumerate(train_sns[:3]):  # Show only first 3 for readability
                sn_data = df[df["sn"] == sn].sort_values("cumulated_throughput")
                ax.plot(
                    sn_data["cumulated_throughput"],
                    sn_data[performance_metric],
                    marker="o",
                    markersize=4,
                    label=f"SN{sn:02d} (train)",
                    alpha=0.8,
                    linewidth=1.5,
                    color=colors_train[idx],
                )

            for idx, sn in enumerate(test_sns[:3]):  # Show only first 3 for readability
                sn_data = df[df["sn"] == sn].sort_values("cumulated_throughput")
                ax.plot(
                    sn_data["cumulated_throughput"],
                    sn_data[performance_metric],
                    marker="s",
                    markersize=4,
                    label=f"SN{sn:02d} (test)",
                    alpha=0.8,
                    linewidth=1.5,
                    linestyle="--",
                    color=colors_test[idx],
                )

            ax.set_xlabel("Cumulated Throughput (kg)")
            ax.set_ylabel(self._format_label(performance_metric))
            ax.set_title(f"Performance Degradation: {self._format_label(performance_metric)}")
            ax.legend(fontsize=8, ncol=2)
            self._add_grid(ax)

            plt.tight_layout()
            self._save_figure(f"10_degradation_{performance_metric}")
            plt.close()

        logger.info("Performance degradation plots created")

    def plot_test_mode_analysis(self, df: pd.DataFrame) -> None:
        """Test mode analysis if available."""
        logger.info("Creating test mode analysis plots")

        if "test_mode" not in df.columns:
            logger.info("test_mode column not found, skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_LARGE)

        for idx, performance_metric in enumerate(self.PERFORMANCE_METRICS):
            if performance_metric not in df.columns:
                continue

            ax = axes[idx]

            test_modes = sorted(df["test_mode"].unique())
            data_by_mode = [
                df[df["test_mode"] == mode][performance_metric].dropna() for mode in test_modes
            ]

            bp = ax.boxplot(data_by_mode, labels=test_modes, patch_artist=True, showmeans=True)
            self._style_boxplot(bp)

            ax.set_xlabel("Test Mode", fontsize=10)
            ax.set_ylabel(self._format_label(performance_metric), fontsize=10)
            ax.set_title(f"{self._format_label(performance_metric)}", fontsize=11)
            self._add_grid(ax, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

        plt.suptitle("Performance by Test Mode", fontsize=13)
        plt.tight_layout()
        self._save_figure("11_test_mode_analysis")
        plt.close()

        logger.info("Test mode analysis plots created")

    def plot_anomaly_analysis_improved(self, df: pd.DataFrame) -> None:
        """
        Improved anomaly analysis - READABLE version.
        Fixed: removed cluttered percentage labels.
        """
        logger.info("Creating improved anomaly analysis plots")

        if "anomalous" not in df.columns:
            logger.info("anomalous column not found, skipping")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.FIGURE_SIZE_LARGE)

        # Anomaly by split
        anomaly_split = pd.crosstab(df["split"], df["anomalous"])
        anomaly_split.plot(
            kind="bar",
            ax=axes[0, 0],
            color=[self.COLOR_PRIMARY, self.COLOR_ACCENT],
            edgecolor="black",
            linewidth=1,
        )
        axes[0, 0].set_title("Anomaly Distribution by Split")
        axes[0, 0].set_xlabel("Split")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend(title="Anomalous", labels=["Normal", "Anomalous"])
        self._add_grid(axes[0, 0], axis="y")
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=0)

        # Anomaly by SN
        anomaly_by_sn = df.groupby("sn")["anomalous"].sum()
        axes[0, 1].bar(
            anomaly_by_sn.index,
            anomaly_by_sn.values,
            color=self.COLOR_PRIMARY,
            edgecolor="black",
            linewidth=1,
        )
        axes[0, 1].set_title("Anomaly Count by Serial Number")
        axes[0, 1].set_xlabel("Serial Number (SN)")
        axes[0, 1].set_ylabel("Number of Anomalies")
        self._add_grid(axes[0, 1], axis="y")
        axes[0, 1].axvline(x=12.5, color=self.COLOR_BOUNDARY, linestyle="--", linewidth=2)

        # Thrust distribution comparison
        if "thrust_mean" in df.columns:
            anomalous_data = df[df["anomalous"]]["thrust_mean"].dropna()
            normal_data = df[~df["anomalous"]]["thrust_mean"].dropna()

            axes[1, 0].hist(
                normal_data,
                bins=30,
                alpha=0.6,
                label="Normal",
                color=self.COLOR_PRIMARY,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )
            axes[1, 0].hist(
                anomalous_data,
                bins=30,
                alpha=0.6,
                label="Anomalous",
                color=self.COLOR_ACCENT,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )

            axes[1, 0].set_title("Thrust: Normal vs Anomalous")
            axes[1, 0].set_xlabel("Thrust Mean (N)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            self._add_grid(axes[1, 0])

        # IMPROVED: Anomaly codes - showing only counts, NO percentage labels
        if "anomaly_code" in df.columns:
            anomaly_codes = df[df["anomaly_code"] > 0]["anomaly_code"]

            if len(anomaly_codes) > 0:
                code_counts = anomaly_codes.value_counts().sort_index()

                # Show only top 10 codes for readability
                if len(code_counts) > 10:
                    code_counts = code_counts.nlargest(10)

                bars = axes[1, 1].bar(
                    range(len(code_counts)),
                    code_counts.values,
                    color=self.COLOR_ACCENT,
                    edgecolor="black",
                    linewidth=1,
                )

                axes[1, 1].set_xticks(range(len(code_counts)))
                axes[1, 1].set_xticklabels(
                    [f"{int(code)}" for code in code_counts.index],
                    rotation=0,
                    fontsize=8,
                )

                axes[1, 1].set_title("Anomaly Code Distribution (Top 10)")
                axes[1, 1].set_xlabel("Anomaly Code")
                axes[1, 1].set_ylabel("Count")
                self._add_grid(axes[1, 1], axis="y")

                # Add count values on top of bars (NO percentages)
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No anomalies detected",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Anomaly Code Distribution")

        plt.suptitle("Anomaly Analysis", fontsize=13)
        plt.tight_layout()
        self._save_figure("12_anomaly_analysis")
        plt.close()

        logger.info("Improved anomaly analysis plots created")

    def _save_figure(self, filename: str) -> None:
        filepath = self.output_dir / f"{filename}.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches="tight")
        logger.debug(f"Saved figure: {filepath}")
