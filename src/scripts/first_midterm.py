import argparse
import sys
import time

import pandas as pd
from loguru import logger

from ..common import config
from ..common.logger import setup_logger
from ..common.time import format_elapsed_time
from ..data.aggregator import DataAggregator
from ..data.cleaner import DataCleaner
from ..data.loader import DataLoader
from ..exploration.statistics import StatisticsAnalyzer
from ..exploration.visualizer import Visualizer


class Pipeline:
    PROGRESS_LOG_INTERVAL = 100

    def __init__(self, save_individual_cleaned_files: bool = False):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.aggregator = DataAggregator()
        self.stats_analyzer = StatisticsAnalyzer()
        self.visualizer = Visualizer(config.FIGURES_DIR)
        self.save_individual_cleaned_files = save_individual_cleaned_files

        self._cached_metadata = None
        self._cached_aggregated_df = None

    def run_full_pipeline(self) -> None:
        logger.info("Starting full pipeline")
        pipeline_start_time = time.time()

        try:
            metadata = self._load_and_clean_metadata()
            aggregates_list, successful_indices = self._process_time_series_files(metadata)
            filtered_metadata = metadata.loc[successful_indices].reset_index(drop=True)
            aggregated_df = self._create_aggregated_dataset(filtered_metadata, aggregates_list)
            self._save_processed_data(filtered_metadata, aggregated_df)
            self._generate_statistics(aggregated_df)
            self.visualizer.create_all_visualizations(aggregated_df)
            self._save_cleaning_report()

            self._cached_metadata = filtered_metadata
            self._cached_aggregated_df = aggregated_df

            total_time = time.time() - pipeline_start_time
            time_formatted = format_elapsed_time(total_time)
            logger.info(f"Pipeline completed successfully in {time_formatted}")
        except Exception as e:
            total_time = time.time() - pipeline_start_time
            time_formatted = format_elapsed_time(total_time)
            logger.error(f"Pipeline failed after {time_formatted}: {e}")
            raise

    def run_data_processing(self) -> None:
        logger.info("Starting data loading and cleaning")
        start_time = time.time()

        try:
            metadata = self._load_and_clean_metadata()
            aggregates_list, successful_indices = self._process_time_series_files(metadata)
            filtered_metadata = metadata.loc[successful_indices].reset_index(drop=True)
            aggregated_df = self._create_aggregated_dataset(filtered_metadata, aggregates_list)
            self._save_processed_data(filtered_metadata, aggregated_df)
            self._save_cleaning_report()

            self._cached_metadata = filtered_metadata
            self._cached_aggregated_df = aggregated_df

            elapsed = format_elapsed_time(time.time() - start_time)
            logger.info(f"Data processing completed in {elapsed}")
        except Exception as e:
            elapsed = format_elapsed_time(time.time() - start_time)
            logger.error(f"Data processing failed after {elapsed}: {e}")
            raise

    def run_statistics(self) -> None:
        logger.info("Generating statistics")
        start_time = time.time()

        try:
            aggregated_df = self._load_aggregated_data()
            self._generate_statistics(aggregated_df)

            elapsed = format_elapsed_time(time.time() - start_time)
            logger.info(f"Statistics generated in {elapsed}")
        except Exception as e:
            elapsed = format_elapsed_time(time.time() - start_time)
            logger.error(f"Statistics generation failed after {elapsed}: {e}")
            raise

    def run_visualizations(self) -> None:
        logger.info("Creating visualizations")
        start_time = time.time()

        try:
            aggregated_df = self._load_aggregated_data()
            self.visualizer.create_all_visualizations(aggregated_df)

            elapsed = format_elapsed_time(time.time() - start_time)
            logger.info(f"Visualizations created in {elapsed}")
        except Exception as e:
            elapsed = format_elapsed_time(time.time() - start_time)
            logger.error(f"Visualization failed after {elapsed}: {e}")
            raise

    def _load_aggregated_data(self) -> pd.DataFrame:
        if self._cached_aggregated_df is not None:
            logger.debug("Using cached aggregated data")
            return self._cached_aggregated_df

        if config.AGGREGATED_DATA_PATH.exists():
            logger.info(f"Loading aggregated data from {config.AGGREGATED_DATA_PATH}")
            df = pd.read_csv(config.AGGREGATED_DATA_PATH)
            self._cached_aggregated_df = df
            return df

        logger.error(
            f"Aggregated data not found at {config.AGGREGATED_DATA_PATH}. "
            "Please run data processing first."
        )
        raise FileNotFoundError("Aggregated data not found. Run 'Load and Clean Data' step first.")

    def _load_and_clean_metadata(self) -> pd.DataFrame:
        logger.info("Loading and cleaning metadata")
        metadata = self.loader.load_metadata()
        return self.cleaner.clean_metadata(metadata)

    def _process_single_time_series_file(self, filename: str, sn: int) -> tuple[dict, bool]:
        data_dir = config.TRAIN_DIR if sn in config.TRAIN_SNS else config.TEST_DIR

        try:
            ts_df = self.loader.load_time_series(filename, data_dir)
            ts_df = self.cleaner.clean_time_series(ts_df)
            aggregates = self.aggregator.aggregate_time_series(ts_df)

            if self.save_individual_cleaned_files:
                output_path = config.TIME_SERIES_DIR / filename
                ts_df.to_csv(output_path, index=False)

            return aggregates, True

        except FileNotFoundError:
            logger.warning(f"File not found: {filename}")
            return {}, False
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return {}, False

    def _process_time_series_files(self, metadata: pd.DataFrame) -> tuple[list[dict], list[int]]:
        logger.info("Processing time series files")
        aggregates_list = []
        successful_indices = []

        total_files = len(metadata)
        processed_count = 0
        failed_count = 0
        start_time = time.time()

        for file_index, row in metadata.iterrows():
            index_value = int(file_index) if isinstance(file_index, (int, float)) else 0

            filename = str(row["filename"])
            sn = int(row["sn"])

            aggregates, success = self._process_single_time_series_file(filename, sn)

            if success:
                aggregates_list.append(aggregates)
                successful_indices.append(file_index)
                processed_count += 1
            else:
                failed_count += 1

            if index_value % self.PROGRESS_LOG_INTERVAL == 0 and index_value > 0:
                elapsed = time.time() - start_time
                speed = index_value / elapsed if elapsed > 0 else 0
                percent = (index_value / total_files) * 100
                logger.info(
                    f"Progress: {percent:.1f}% ({index_value}/{total_files}) | "
                    f"Speed: {speed:.1f} files/s | Errors: {failed_count}"
                )

        logger.info(f"Processed {processed_count} files successfully, {failed_count} failed")
        return aggregates_list, successful_indices

    def _create_aggregated_dataset(
        self, metadata: pd.DataFrame, aggregates_list: list[dict]
    ) -> pd.DataFrame:
        logger.info("Creating aggregated dataset")
        return self.aggregator.create_aggregated_dataset(metadata, aggregates_list)

    def _save_processed_data(self, metadata: pd.DataFrame, aggregated_df: pd.DataFrame) -> None:
        logger.info("Saving processed data")
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        try:
            metadata.to_csv(config.METADATA_CLEANED_PATH, index=False)
            logger.info(f"Saved cleaned metadata ({len(metadata)} records)")

            aggregated_df.to_csv(config.AGGREGATED_DATA_PATH, index=False)
            logger.info(f"Saved aggregated data ({len(aggregated_df)} records)")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def _generate_statistics(self, df: pd.DataFrame) -> None:
        logger.info("Generating statistics")
        config.STATISTICS_DIR.mkdir(parents=True, exist_ok=True)

        self.stats_analyzer.generate_statistics_report(df, config.REPORTS_DIR)
        self.stats_analyzer.save_statistics_tables(df, config.STATISTICS_DIR)

    def _save_cleaning_report(self) -> None:
        logger.info("Saving cleaning report")
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        report = self.cleaner.get_cleaning_report()
        report_path = config.REPORTS_DIR / "cleaning_report.txt"
        report_path.write_text(report)
        logger.info(f"Cleaning report saved to {report_path}")


def run_interactive_menu(pipeline: Pipeline) -> None:
    while True:
        print("\n" + "=" * 60)
        print("ML THRUSTER DATA PROCESSING PIPELINE")
        print("=" * 60)
        print("\n[1] Run Full Pipeline")
        print("[2] Load and Clean Data (metadata + time series)")
        print("[3] Generate Statistics")
        print("[4] Create Visualizations")
        print("[5] Run Statistics + Visualizations")
        print("[0] Exit")
        print()

        try:
            choice = input("Select option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

        print()

        if choice == "1":
            pipeline.run_full_pipeline()
        elif choice == "2":
            pipeline.run_data_processing()
        elif choice == "3":
            pipeline.run_statistics()
        elif choice == "4":
            pipeline.run_visualizations()
        elif choice == "5":
            pipeline.run_statistics()
            pipeline.run_visualizations()
        elif choice == "0":
            logger.info("Exiting pipeline")
            break
        else:
            logger.warning(f"Invalid choice: {choice}")


def main():
    parser = argparse.ArgumentParser(description="ML Thruster Data Processing Pipeline")
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="Enable logging to file",
    )
    parser.add_argument(
        "--save-cleaned-csv",
        action="store_true",
        dest="save_individual_files",
        help="Save cleaned individual CSV files",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive menu mode",
    )

    args = parser.parse_args()

    enable_file_logging = args.log_file
    log_path = config.OUTPUTS_DIR / "pipeline.log" if enable_file_logging else None
    setup_logger(enable_file_logging=enable_file_logging, log_path=log_path)

    pipeline = Pipeline(save_individual_cleaned_files=args.save_individual_files)

    if args.interactive:
        run_interactive_menu(pipeline)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
