"""
Extract component of the ETL pipeline for Mushroom Classification.
Responsible for reading data from the source files.
"""

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/extract.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_data(data_path):
    """
    Extract data from source files in the provided directory.

    Args:
        data_path (str): Path to the directory containing data files.

    Returns:
        pd.DataFrame: Extracted data.
    """
    try:
        logger.info(f"Extracting data from {data_path}")

        # Check if directory exists
        if not os.path.exists(data_path):
            logger.error(f"Directory {data_path} does not exist")
            raise FileNotFoundError(f"Directory {data_path} does not exist")

        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if not csv_files:
            logger.error(f"No CSV files found in {data_path}")
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        # Use the first CSV file found
        file_path = os.path.join(data_path, csv_files[0])
        logger.info(f"Using file: {file_path}")

        # Read the CSV file with semicolon delimiter
        df = pd.read_csv(file_path, delimiter=";")

        # Clean column names - replace hyphens with underscores and strip whitespace
        df.columns = df.columns.str.replace("-", "_").str.strip()

        logger.info(f"Successfully extracted {len(df)} records from {file_path}")
        logger.info(f"Data shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
