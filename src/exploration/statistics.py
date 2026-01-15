from pathlib import Path

import pandas as pd
from loguru import logger


class StatisticsAnalyzer:
    EXCLUDED_COLUMNS = ["uid"]
    SEPARATOR_WIDTH = 80
    CORRELATION_METHOD = "pearson"

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        return [col for col in numeric_cols if col not in self.EXCLUDED_COLUMNS]

    def compute_summary_statistics(
        self, df: pd.DataFrame, split_name: str | None = None
    ) -> pd.DataFrame:
        if split_name:
            df = df[df["split"] == split_name].copy()  # type: ignore[assignment]

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
        return numeric_df.corr(method=self.CORRELATION_METHOD)  # type: ignore[call-arg]

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

        for split_name in ["train", "test"]:
            stats = self.compute_summary_statistics(df, split_name)
            report.append(f"\n{split_name.upper()} SPLIT STATISTICS:")
            report.append(subseparator)
            report.append(stats.to_string())
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
        logger.info("Saved 3 statistics tables and correlation matrix")
