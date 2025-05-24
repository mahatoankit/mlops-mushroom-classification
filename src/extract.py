"""
Extract component of the ETL pipeline for Mushroom Classification.
Responsible for reading data from the source files.
"""

import os
import pandas as pd
import logging

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/extract.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_data(data_path):
    """
    Extract data from source files or directory.

    Args:
        data_path (str): Path to the data file or directory containing data files.

    Returns:
        pd.DataFrame: Extracted data.
    """
    try:
        logger.info(f"Extracting data from {data_path}")

        # Check if path exists
        if not os.path.exists(data_path):
            logger.error(f"Path {data_path} does not exist")
            raise FileNotFoundError(f"Path {data_path} does not exist")

        # Handle both file and directory paths
        if os.path.isfile(data_path):
            # Direct file path
            file_path = data_path
            logger.info(f"Using file: {file_path}")
        elif os.path.isdir(data_path):
            # Directory path - find CSV files
            csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
            if not csv_files:
                logger.error(f"No CSV files found in {data_path}")
                raise FileNotFoundError(f"No CSV files found in {data_path}")

            # Use the first CSV file found
            file_path = os.path.join(data_path, csv_files[0])
            logger.info(f"Using file: {file_path}")
        else:
            raise ValueError(f"Path {data_path} is neither a file nor a directory")

        # Try different delimiters and encodings
        delimiters = [";", ",", "\t"]
        encodings = ["utf-8", "latin-1", "cp1252"]

        df = None
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                    if not df.empty and len(df.columns) > 1:
                        logger.info(
                            f"Successfully read with delimiter='{delimiter}' and encoding='{encoding}'"
                        )
                        break
                except Exception as e:
                    logger.debug(
                        f"Failed with delimiter='{delimiter}' and encoding='{encoding}': {e}"
                    )
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            # Fallback: try with default pandas settings
            try:
                df = pd.read_csv(file_path)
                logger.info("Successfully read with default pandas settings")
            except Exception as e:
                logger.error(f"Could not read file with any method: {e}")
                raise

        # Clean column names - replace hyphens with underscores and strip whitespace
        df.columns = df.columns.str.replace("-", "_").str.strip().str.lower()

        # Basic data validation
        if df.empty:
            raise ValueError("Extracted DataFrame is empty")

        if len(df.columns) == 0:
            raise ValueError("Extracted DataFrame has no columns")

        logger.info(f"Successfully extracted {len(df)} records from {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
