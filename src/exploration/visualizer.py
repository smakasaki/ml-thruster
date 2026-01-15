from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


class Visualizer:
    """
    Comprehensive visualization toolkit for STFT dataset exploratory analysis.
    Generates all plots required for Midterm 1 presentation.
    """

    FIGURE_DPI = 150
    FIGURE_SIZE_SMALL = (10, 6)
    FIGURE_SIZE_MEDIUM = (12, 8)
    FIGURE_SIZE_LARGE = (14, 10)
    FIGURE_SIZE_WIDE = (16, 6)

    HISTOGRAM_BINS = 50
    HISTOGRAM_BINS_COMPARISON = 30
    HISTOGRAM_ALPHA = 0.7
    HISTOGRAM_ALPHA_COMPARISON = 0.5
    HISTOGRAM_EDGECOLOR = "black"

    SCATTER_ALPHA = 0.6
    SCATTER_SIZE = 30

    COLOR_PRIMARY = "steelblue"
    COLOR_SECONDARY = "lightblue"
    COLOR_ACCENT = "salmon"
    COLOR_TRAIN = "blue"
    COLOR_TEST = "red"
    COLOR_MEAN = "red"
    COLOR_MEDIAN = "green"
    COLOR_TREND = "green"
    COLOR_BOUNDARY = "red"

    GRID_ALPHA = 0.3
    LINE_WIDTH = 2
    LINE_WIDTH_THIN = 1.5
    MARKER_SIZE = 4

    TRAIN_MARKER = "o"
    TEST_MARKER = "s"
    TRAIN_ALPHA = 0.7
    TEST_ALPHA = 0.5
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
        """
        Initialize Visualizer with output directory.

        Parameters
        ----------
        output_dir : Path
            Directory where plots will be saved
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        sns.set_palette("husl")

        logger.info(f"Visualizer initialized. Output directory: {output_dir}")

    def _format_label(self, text: str) -> str:
        """Format column name to readable label."""
        return text.replace("_", " ").title()

    def _add_grid(self, ax, axis: str = "both") -> None:
        """Add grid to axis with consistent styling."""
        ax.grid(axis=axis, alpha=self.GRID_ALPHA)

    def _add_mean_median_lines(self, ax, data: pd.Series) -> None:
        """Add vertical lines for mean and median to histogram."""
        ax.axvline(
            data.mean(),  # type: ignore[arg-type]
            color=self.COLOR_MEAN,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Mean",
        )
        ax.axvline(
            data.median(),  # type: ignore[arg-type]
            color=self.COLOR_MEDIAN,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Median",
        )

    def _style_boxplot(self, box_plot_dict: dict) -> None:
        """Apply consistent styling to boxplot."""
        for box in box_plot_dict["boxes"]:
            box.set_facecolor(self.COLOR_SECONDARY)

    def _add_trend_line(self, ax, x_data: pd.Series, y_data: pd.Series) -> None:
        """Add polynomial trend line to scatter plot."""
        x_clean = x_data.dropna()  # type: ignore[union-attr]
        y_clean = y_data.dropna()  # type: ignore[union-attr]
        common_idx = x_clean.index.intersection(y_clean.index)

        if len(common_idx) < 2:
            return

        z = np.polyfit(x_clean[common_idx], y_clean[common_idx], self.POLYNOMIAL_DEGREE)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), self.TREND_LINE_POINTS)  # type: ignore[arg-type]
        ax.plot(
            x_line,
            p(x_line),
            color=self.COLOR_TREND,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Trend line",
            alpha=0.8,
        )

    def _add_train_test_scatter(self, ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """Add scatter points for train and test data."""
        train_data = df[df["split"] == "train"]
        test_data = df[df["split"] == "test"]

        ax.scatter(
            train_data[x_col],
            train_data[y_col],
            alpha=self.SCATTER_ALPHA,
            s=self.SCATTER_SIZE,
            label="Train (SN01-12)",
            color=self.COLOR_TRAIN,
        )

        ax.scatter(
            test_data[x_col],
            test_data[y_col],
            alpha=self.SCATTER_ALPHA,
            s=self.SCATTER_SIZE,
            label="Test (SN13-24)",
            color=self.COLOR_TEST,
        )

    def _plot_sn_line(
        self,
        ax,
        df: pd.DataFrame,
        sn: int,
        x_col: str,
        y_col: str,
        marker: str,
        alpha: float,
        linestyle: str,
        linewidth: float,
        label_suffix: str = "",
    ) -> None:
        """Plot performance line for a single serial number."""
        sn_data = df[df["sn"] == sn].sort_values(x_col)  # type: ignore[call-overload]
        label = f"SN{sn:02d}{label_suffix}"
        ax.plot(
            sn_data[x_col],
            sn_data[y_col],
            marker=marker,
            markersize=self.MARKER_SIZE,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    def _add_bar_labels(self, ax, bars, total_count: int) -> None:
        """Add count and percentage labels above bars."""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}\n({height / total_count * 100:.1f}%)",
                ha="center",
                va="bottom",
            )

    def _plot_anomaly_by_split(self, ax, df: pd.DataFrame) -> None:
        """Plot anomaly distribution by train/test split."""
        anomaly_split = pd.crosstab(df["split"], df["anomalous"])
        anomaly_split.plot(kind="bar", ax=ax, color=[self.COLOR_SECONDARY, self.COLOR_ACCENT])
        ax.set_title("Anomaly Distribution by Split")
        ax.set_xlabel("Split")
        ax.set_ylabel("Count")
        ax.legend(title="Anomalous", labels=["Normal", "Anomalous"])
        self._add_grid(ax, axis="y")

    def _plot_anomaly_by_sn(self, ax, df: pd.DataFrame) -> None:
        """Plot anomaly count by serial number."""
        anomaly_by_sn = df.groupby("sn")["anomalous"].sum()
        ax.bar(
            anomaly_by_sn.index,
            anomaly_by_sn.values,  # type: ignore[arg-type]
            color=self.COLOR_PRIMARY,
            edgecolor=self.HISTOGRAM_EDGECOLOR,
        )
        ax.set_title("Anomaly Count by Serial Number")
        ax.set_xlabel("Serial Number (SN)")
        ax.set_ylabel("Number of Anomalies")
        self._add_grid(ax, axis="y")
        ax.axvline(
            x=12.5,
            color=self.COLOR_BOUNDARY,
            linestyle="--",
            linewidth=self.LINE_WIDTH,
            label="Train/Test",
        )
        ax.legend()

    def _plot_thrust_distribution_comparison(self, ax, df: pd.DataFrame) -> None:
        """Plot thrust distribution comparison between normal and anomalous data."""
        if "thrust_mean" not in df.columns:
            return

        anomalous_data = df[df["anomalous"]]
        normal_data = df[~df["anomalous"]]

        ax.hist(
            normal_data["thrust_mean"].dropna(),  # type: ignore[union-attr]
            bins=self.HISTOGRAM_BINS_COMPARISON,
            alpha=self.HISTOGRAM_ALPHA_COMPARISON,
            label="Normal",
            color=self.COLOR_TRAIN,
        )
        ax.hist(
            anomalous_data["thrust_mean"].dropna(),  # type: ignore[union-attr]
            bins=self.HISTOGRAM_BINS_COMPARISON,
            alpha=self.HISTOGRAM_ALPHA_COMPARISON,
            label="Anomalous",
            color=self.COLOR_TEST,
        )
        ax.set_title("Thrust Distribution: Normal vs Anomalous")
        ax.set_xlabel("Thrust Mean (N)")
        ax.set_ylabel("Frequency")
        ax.legend()
        self._add_grid(ax)

    def _plot_anomaly_codes(self, ax, df: pd.DataFrame) -> None:
        """Plot top anomaly codes distribution."""
        if "anomaly_code" not in df.columns or not df["anomaly_code"].notna().any():  # type: ignore[union-attr]
            ax.text(
                0.5,
                0.5,
                "No anomaly code data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return

        anomaly_codes = df[df["anomaly_code"] > 0]["anomaly_code"].value_counts().head(10)  # type: ignore[union-attr]
        ax.barh(
            range(len(anomaly_codes)),
            anomaly_codes.values,
            color=self.COLOR_ACCENT,
            edgecolor=self.HISTOGRAM_EDGECOLOR,
        )
        ax.set_yticks(range(len(anomaly_codes)))
        ax.set_yticklabels([f"Code {int(code)}" for code in anomaly_codes.index])
        ax.set_title("Top 10 Anomaly Codes")
        ax.set_xlabel("Frequency")
        self._add_grid(ax, axis="x")

    def create_all_visualizations(self, df: pd.DataFrame) -> None:
        """
        Generate all visualizations required for Midterm 1.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated dataset with metadata and time series statistics
        """
        logger.info("Starting comprehensive visualization generation")

        self.plot_univariate_distributions(df)
        self.plot_categorical_distributions(df)
        self.plot_correlation_heatmap(df)
        self.plot_aging_vs_performance(df)
        self.plot_pressure_vs_performance(df)
        self.plot_performance_degradation_by_sn(df)
        self.plot_test_mode_analysis(df)
        self.plot_anomaly_analysis(df)
        self.plot_train_test_comparison(df)

        logger.info("All visualizations completed successfully")

    def plot_univariate_distributions(self, df: pd.DataFrame) -> None:
        """
        Create histograms and box plots for continuous variables.

        Generates combined histogram + box plot for each numeric feature.
        """
        logger.info("Creating univariate distribution plots")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        plot_cols = [col for col in numeric_cols if col not in self.EXCLUDE_COLS]

        for col in plot_cols:
            fig, axes = plt.subplots(1, 2, figsize=self.FIGURE_SIZE_MEDIUM)

            data = df[col].dropna()

            axes[0].hist(
                data,
                bins=self.HISTOGRAM_BINS,
                alpha=self.HISTOGRAM_ALPHA,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
                color=self.COLOR_PRIMARY,
            )
            axes[0].set_title(f"{col} - Distribution")
            axes[0].set_xlabel(col)
            axes[0].set_ylabel("Frequency")
            self._add_grid(axes[0])

            self._add_mean_median_lines(axes[0], data)  # type: ignore[arg-type]
            axes[0].legend()

            box_plot = axes[1].boxplot(data, vert=True, patch_artist=True)
            self._style_boxplot(box_plot)
            axes[1].set_title(f"{col} - Box Plot")
            axes[1].set_ylabel(col)
            self._add_grid(axes[1])

            plt.tight_layout()
            self._save_figure(f"01_univariate_{col}")
            plt.close()

        logger.info(f"Created {len(plot_cols)} univariate distribution plots")

    def plot_categorical_distributions(self, df: pd.DataFrame) -> None:
        """
        Create count plots for categorical variables.

        Shows distribution of test_mode, anomalous flag, and split.
        """
        logger.info("Creating categorical distribution plots")

        categorical_cols = ["test_mode", "anomalous", "split"]

        for col in categorical_cols:
            if col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_SMALL)

            counts = df[col].value_counts()
            bars = ax.bar(
                range(len(counts)),
                counts.values,  # type: ignore[arg-type]
                color=self.COLOR_PRIMARY,
                edgecolor=self.HISTOGRAM_EDGECOLOR,
            )
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=self.LABEL_ROTATION, ha="right")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col}")
            self._add_grid(ax, axis="y")

            self._add_bar_labels(ax, bars, len(df))

            plt.tight_layout()
            self._save_figure(f"02_categorical_{col}")
            plt.close()

        logger.info(f"Created {len(categorical_cols)} categorical distribution plots")

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Create correlation heatmap for numeric features.

        Uses Pearson correlation coefficient with annotated values.
        """
        logger.info("Creating correlation heatmap")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        analysis_cols = [col for col in numeric_cols if col not in self.EXCLUDE_COLS_CORRELATION]

        corr_matrix = df[analysis_cols].corr()  # type: ignore[call-overload]

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_LARGE)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        heatmap_config = {
            "annot": True,
            "fmt": ".2f",
            "cmap": "coolwarm",
            "center": 0,
            "square": True,
            "linewidths": 0.5,
            "cbar_kws": {"shrink": 0.8},
            "vmin": -1,
            "vmax": 1,
        }

        sns.heatmap(corr_matrix, mask=mask, ax=ax, **heatmap_config)

        ax.set_title("Correlation Matrix (Pearson)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_figure("03_correlation_heatmap")
        plt.close()

        logger.info("Correlation heatmap created")

    def plot_aging_vs_performance(self, df: pd.DataFrame) -> None:
        """
        Create scatter plots showing performance degradation vs aging factors.

        Generates 6 plots: 3 aging factors × 2 performance metrics.
        """
        logger.info("Creating aging vs performance scatter plots")

        for aging_factor in self.AGING_FACTORS:
            if aging_factor not in df.columns:
                continue

            for performance_metric in self.PERFORMANCE_METRICS:
                if performance_metric not in df.columns:
                    continue

                fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_MEDIUM)

                self._add_train_test_scatter(ax, df, aging_factor, performance_metric)
                self._add_trend_line(ax, df[aging_factor], df[performance_metric])  # type: ignore[arg-type]

                ax.set_xlabel(self._format_label(aging_factor))
                ax.set_ylabel(self._format_label(performance_metric))
                ax.set_title(f"Performance Degradation: {performance_metric} vs {aging_factor}")
                ax.legend()
                self._add_grid(ax)

                plt.tight_layout()
                self._save_figure(f"04_aging_{aging_factor}_vs_{performance_metric}")
                plt.close()

        logger.info("Aging vs performance plots created")

    def plot_pressure_vs_performance(self, df: pd.DataFrame) -> None:
        """
        Create scatter plots showing performance vs test pressure.

        Shows how thrust and MFR vary with inlet pressure colored by thruster age.
        """
        logger.info("Creating pressure vs performance plots")

        if "test_pressure" not in df.columns:
            logger.warning("test_pressure column not found, skipping pressure plots")
            return

        for performance_metric in self.PERFORMANCE_METRICS:
            if performance_metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_MEDIUM)

            scatter = ax.scatter(
                df["test_pressure"],
                df[performance_metric],
                c=df["cumulated_throughput"],
                cmap="viridis",
                alpha=self.SCATTER_ALPHA,
                s=self.SCATTER_SIZE,
                edgecolors=self.HISTOGRAM_EDGECOLOR,
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cumulated Throughput (kg)")

            ax.set_xlabel("Test Pressure (bars)")
            ax.set_ylabel(self._format_label(performance_metric))
            ax.set_title(f"Pressure Effect on {performance_metric}")
            self._add_grid(ax)

            plt.tight_layout()
            self._save_figure(f"05_pressure_vs_{performance_metric}")
            plt.close()

        logger.info("Pressure vs performance plots created")

    def plot_performance_degradation_by_sn(self, df: pd.DataFrame) -> None:
        """
        Create line plots showing performance evolution for each thruster.

        Shows degradation trend across all serial numbers.
        """
        logger.info("Creating performance degradation by SN plots")

        for performance_metric in self.PERFORMANCE_METRICS:
            if performance_metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_WIDE)

            train_sns = df[df["split"] == "train"]["sn"].unique()  # type: ignore[union-attr]
            test_sns = df[df["split"] == "test"]["sn"].unique()  # type: ignore[union-attr]

            for sn in sorted(train_sns):
                self._plot_sn_line(
                    ax,
                    df,
                    sn,
                    "cumulated_throughput",
                    performance_metric,
                    self.TRAIN_MARKER,
                    self.TRAIN_ALPHA,
                    self.TRAIN_LINE_STYLE,
                    self.LINE_WIDTH,
                )

            for sn in sorted(test_sns):
                self._plot_sn_line(
                    ax,
                    df,
                    sn,
                    "cumulated_throughput",
                    performance_metric,
                    self.TEST_MARKER,
                    self.TEST_ALPHA,
                    self.TEST_LINE_STYLE,
                    self.LINE_WIDTH_THIN,
                    " (test)",
                )

            ax.set_xlabel("Cumulated Throughput (kg)")
            ax.set_ylabel(self._format_label(performance_metric))
            ax.set_title(f"Thruster Performance Degradation: {performance_metric} over Life")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
            self._add_grid(ax)

            ax.axvline(
                x=df[df["split"] == "train"]["cumulated_throughput"].max(),  # type: ignore[arg-type]
                color=self.COLOR_BOUNDARY,
                linestyle=":",
                linewidth=self.LINE_WIDTH,
            )

            plt.tight_layout()
            self._save_figure(f"06_degradation_by_sn_{performance_metric}")
            plt.close()

        logger.info("Performance degradation by SN plots created")

    def plot_test_mode_analysis(self, df: pd.DataFrame) -> None:
        """
        Create box plots comparing performance across test modes.

        Shows how different operational modes affect thrust and MFR.
        """
        logger.info("Creating test mode analysis plots")

        if "test_mode" not in df.columns:
            logger.warning("test_mode column not found, skipping test mode plots")
            return

        for performance_metric in self.PERFORMANCE_METRICS:
            if performance_metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=self.FIGURE_SIZE_MEDIUM)

            test_modes = df["test_mode"].unique()
            data_by_mode = [
                df[df["test_mode"] == mode][performance_metric].dropna()  # type: ignore[union-attr]
                for mode in test_modes
            ]

            box_plot = ax.boxplot(data_by_mode, patch_artist=True)
            self._style_boxplot(box_plot)
            ax.set_xticks(range(1, len(test_modes) + 1))
            ax.set_xticklabels(test_modes)

            ax.set_xlabel("Test Mode")
            ax.set_ylabel(self._format_label(performance_metric))
            ax.set_title(f"Performance Comparison Across Test Modes: {performance_metric}")
            self._add_grid(ax, axis="y")
            plt.xticks(rotation=self.LABEL_ROTATION, ha="right")

            plt.tight_layout()
            self._save_figure(f"07_test_mode_{performance_metric}")
            plt.close()

        logger.info("Test mode analysis plots created")

    def plot_anomaly_analysis(self, df: pd.DataFrame) -> None:
        """
        Create visualizations analyzing anomaly distribution and characteristics.

        Includes: anomaly frequency by phase, anomaly code distribution.
        """
        logger.info("Creating anomaly analysis plots")

        if "anomalous" not in df.columns:
            logger.warning("anomalous column not found, skipping anomaly plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.FIGURE_SIZE_LARGE)

        self._plot_anomaly_by_split(axes[0, 0], df)
        self._plot_anomaly_by_sn(axes[0, 1], df)
        self._plot_thrust_distribution_comparison(axes[1, 0], df)
        self._plot_anomaly_codes(axes[1, 1], df)

        plt.tight_layout()
        self._save_figure("08_anomaly_analysis")
        plt.close()

        logger.info("Anomaly analysis plots created")

    def plot_train_test_comparison(self, df: pd.DataFrame) -> None:
        """
        Create visualizations comparing train and test set distributions.

        Critical for validating that test set is representative of train set.
        """
        logger.info("Creating train/test comparison plots")

        if "split" not in df.columns:
            logger.warning("split column not found, skipping train/test comparison")
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

            train_data = df[df["split"] == "train"][feature].dropna()  # type: ignore[union-attr]
            test_data = df[df["split"] == "test"][feature].dropna()  # type: ignore[union-attr]

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
                linewidth=self.LINE_WIDTH,
                label="Train mean",
            )
            ax.axvline(
                test_data.mean(),
                color=self.COLOR_TEST,
                linestyle="--",
                linewidth=self.LINE_WIDTH,
                label="Test mean",
            )

            ax.set_title(f"Train vs Test: {self._format_label(feature)}")
            ax.set_xlabel(self._format_label(feature))
            ax.set_ylabel("Frequency")
            ax.legend()
            self._add_grid(ax)

        plt.tight_layout()
        self._save_figure("09_train_test_comparison")
        plt.close()

        logger.info("Train/test comparison plots created")

    def _save_figure(self, filename: str) -> None:
        """
        Save current figure to output directory with consistent naming and format.

        Parameters
        ----------
        filename : str
            Base filename without extension
        """
        filepath = self.output_dir / f"{filename}.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches="tight")
        logger.debug(f"Saved figure: {filepath}")
