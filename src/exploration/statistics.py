from pathlib import Path

import pandas as pd
from loguru import logger


class StatisticsAnalyzer:
    EXCLUDED_COLUMNS = ["uid", "filename"]
    SEPARATOR_WIDTH = 80
    CORRELATION_METHOD = "pearson"

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        return [col for col in numeric_cols if col not in self.EXCLUDED_COLUMNS]

    def compute_summary_statistics(
        self, df: pd.DataFrame, split_name: str | None = None
    ) -> pd.DataFrame:
        if split_name:
            df = df[df["split"] == split_name].copy()

        numeric_cols = self._get_numeric_columns(df)

        return pd.DataFrame(
            {
                "mean": df[numeric_cols].mean(),
                "median": df[numeric_cols].median(),
                "std": df[numeric_cols].std(),
                "min": df[numeric_cols].min(),
                "max": df[numeric_cols].max(),
                "q1": df[numeric_cols].quantile(0.25),
                "q3": df[numeric_cols].quantile(0.75),
                "iqr": df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
                "skewness": df[numeric_cols].skew(),
                "count": df[numeric_cols].count(),
            }
        )

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing correlation matrix")
        numeric_cols = self._get_numeric_columns(df)
        numeric_df = df[numeric_cols]
        return numeric_df.corr(method=self.CORRELATION_METHOD)

    def compute_target_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing target correlations")

        targets = ["thrust_mean", "mfr_mean"]
        existing_targets = [t for t in targets if t in df.columns]

        if len(existing_targets) == 0:
            logger.warning("No target columns found")
            return pd.DataFrame()

        numeric_cols = self._get_numeric_columns(df)
        feature_cols = [col for col in numeric_cols if col not in targets]

        if len(feature_cols) == 0:
            logger.warning("No feature columns found")
            return pd.DataFrame()

        correlations = {}
        for target in existing_targets:
            target_corrs = []
            for feature in feature_cols:
                corr_value = df[feature].corr(df[target])
                target_corrs.append(corr_value)
            correlations[target] = target_corrs

        corr_df = pd.DataFrame(correlations, index=feature_cols)
        corr_df = corr_df.reindex(
            corr_df[existing_targets[0]].abs().sort_values(ascending=False).index
        )

        return corr_df

    def analyze_data_quality(self, df: pd.DataFrame) -> dict:
        logger.info("Analyzing data quality")

        quality_report = {
            "total_records": len(df),
            "missing_values": df.isna().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }

        if "thrust_mean" in df.columns:
            quality_report["negative_thrust_count"] = (df["thrust_mean"] < 0).sum()

        if "mfr_mean" in df.columns:
            quality_report["negative_mfr_count"] = (df["mfr_mean"] < 0).sum()

        return quality_report

    def _format_statistics_report(self, df: pd.DataFrame) -> str:
        separator = "=" * self.SEPARATOR_WIDTH
        subseparator = "-" * self.SEPARATOR_WIDTH

        report = []
        report.append(separator)
        report.append("EXPLORATORY DATA ANALYSIS - SUMMARY STATISTICS")
        report.append(separator)
        report.append("")

        report.append("Dataset Overview:")
        report.append(f"Total records: {len(df)}")
        report.append(f"Train records: {(df['split'] == 'train').sum()}")
        report.append(f"Test records: {(df['split'] == 'test').sum()}")
        report.append("")

        quality = self.analyze_data_quality(df)
        report.append("Data Quality:")
        report.append(f"Duplicate rows: {quality['duplicate_rows']}")
        if "negative_thrust_count" in quality:
            report.append(f"Records with negative thrust: {quality['negative_thrust_count']}")
        if "negative_mfr_count" in quality:
            report.append(f"Records with negative MFR: {quality['negative_mfr_count']}")
        report.append("")

        for split_name in ["train", "test"]:
            stats = self.compute_summary_statistics(df, split_name)
            report.append(f"\n{split_name.upper()} SPLIT STATISTICS:")
            report.append(subseparator)
            report.append(stats.to_string())
            report.append("")

        target_corrs = self.compute_target_correlations(df)
        if not target_corrs.empty:
            report.append("\nTOP CORRELATIONS WITH TARGETS:")
            report.append(subseparator)
            report.append(target_corrs.head(15).to_string())
            report.append("")

        return "\n".join(report)

    def generate_statistics_report(self, df: pd.DataFrame, output_dir: Path) -> None:
        logger.info("Generating statistics report")
        report_text = self._format_statistics_report(df)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "exploration_report.txt"
        report_path.write_text(report_text)
        logger.info(f"Statistics report saved to {report_path}")

    def save_statistics_tables(self, df: pd.DataFrame, output_dir: Path) -> None:
        logger.info("Saving statistics tables")
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "test", None]:
            stats = self.compute_summary_statistics(df, split_name)
            filename = f"summary_stats_{split_name if split_name else 'all'}.csv"
            stats.to_csv(output_dir / filename)
            logger.debug(f"Saved {filename}")

        corr = self.compute_correlation_matrix(df)
        corr.to_csv(output_dir / "correlation_matrix.csv")

        target_corrs = self.compute_target_correlations(df)
        if not target_corrs.empty:
            target_corrs.to_csv(output_dir / "target_correlations.csv")

        logger.info("Saved statistics tables")
