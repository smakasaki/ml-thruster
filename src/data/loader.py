from pathlib import Path

import pandas as pd
from loguru import logger

from ..common import config


class DataLoader:
    def load_metadata(self) -> pd.DataFrame:
        logger.info(f"Loading metadata from {config.METADATA_PATH}")
        try:
            df = pd.read_csv(config.METADATA_PATH)
            logger.info(f"Loaded {len(df)} records from metadata")
            return df
        except FileNotFoundError as e:
            logger.error(f"Metadata file not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse metadata CSV: {e}")
            raise
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read metadata file: {e}")
            raise

    def load_time_series(self, csv_filename: str, data_dir: Path) -> pd.DataFrame:
        csv_filepath = data_dir / csv_filename
        if not csv_filepath.exists():
            msg = f"File not found: {csv_filepath}"
            raise FileNotFoundError(msg)

        try:
            return pd.read_csv(csv_filepath)
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse time series CSV {csv_filename}: {e}")
            raise
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read time series file {csv_filename}: {e}")
            raise

    def list_csv_files(self, data_dir: Path) -> list[Path]:
        return sorted(data_dir.glob("*.csv"))
