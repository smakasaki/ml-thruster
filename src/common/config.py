from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

METADATA_PATH = DATA_DIR / "dataset" / "metadata.csv"
TRAIN_DIR = DATA_DIR / "dataset" / "train"
TEST_DIR = DATA_DIR / "dataset" / "test"

PROCESSED_DIR = DATA_DIR / "processed"
METADATA_CLEANED_PATH = PROCESSED_DIR / "metadata_cleaned.csv"
AGGREGATED_DATA_PATH = PROCESSED_DIR / "aggregated_data.csv"
TIME_SERIES_DIR = PROCESSED_DIR / "time_series"

REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
STATISTICS_DIR = OUTPUTS_DIR / "statistics"

# Training SNs: 1-12, Test SNs: 13-24 based on dataset split
TRAIN_SNS = list(range(1, 13))
TEST_SNS = list(range(13, 25))

METADATA_COLUMNS = [
    "uid",
    "filename",
    "test_id",
    "sn",
    "test_pressure",
    "test_mode",
    "vl1",
    "vl2",
    "vl3",
    "anomalous",
    "anomaly_code",
    "cumulated_throughput",
    "cumulated_on_time",
    "cumulated_pulses",
]

TIME_SERIES_COLUMNS = ["timestamp", "ton", "thrust", "mfr", "vl", "anomaly_code"]

SAMPLING_RATE_HZ = 100
