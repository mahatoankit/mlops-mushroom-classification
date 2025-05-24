"""
Transform component of the ETL pipeline for Mushroom Classification.
Responsible for data preprocessing, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/transform.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def transform_data(df):
    """Transform data exactly like in the notebook"""
    logger.info("Starting data transformation")

    # Drop columns with too many missing values (exactly like notebook)
    columns_to_drop = [
        "gill_spacing",
        "stem_surface",
        "stem_root",
        "spore_print_color",
        "veil_type",
        "veil_color",
    ]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df.drop(columns=existing_columns, inplace=True)
        logger.info(f"Dropped columns: {existing_columns}")

    # Encode and impute specific columns (like notebook)
    label_encoder = LabelEncoder()

    def encode_and_impute(column):
        # Convert to string and handle NaN values properly
        col_data = df[column].astype(str)

        # Replace string 'nan' with a placeholder before encoding
        col_data = col_data.replace("nan", "MISSING_VALUE")

        # Fit and transform the data
        encoded = label_encoder.fit_transform(col_data)
        encoded = pd.Series(encoded, index=df.index)

        # Find the encoded value for 'MISSING_VALUE' and set to NaN
        try:
            missing_value_encoded = label_encoder.transform(["MISSING_VALUE"])[0]
            encoded[encoded == missing_value_encoded] = np.nan
        except ValueError:
            # If 'MISSING_VALUE' wasn't in the data, no NaN values to handle
            pass

        # Get non-NaN values for imputation
        non_nan_values = encoded.dropna().values

        # Sample randomly from these values for imputation
        if encoded.isna().sum() > 0 and len(non_nan_values) > 0:
            sampled_values = np.random.choice(
                non_nan_values, size=encoded.isna().sum(), replace=True
            )
            # Create a copy and set the NaN values to the sampled values
            filled = encoded.copy()
            filled[filled.isna()] = sampled_values
        else:
            filled = encoded.copy()

        # Convert back to original labels
        return label_encoder.inverse_transform(filled.astype(int))

    # Apply to specific columns (like notebook)
    for col in ["cap_surface", "gill_attachment", "ring_type"]:
        if col in df.columns:
            df[col] = encode_and_impute(col)
            logger.info(f"Encoded and imputed column: {col}")

    # Encode target and binary variables (exactly like notebook)
    df["class_encoded"] = label_encoder.fit_transform(df["class"])
    df["does_bruise_or_bleed_encoded"] = label_encoder.fit_transform(
        df["does_bruise_or_bleed"]
    )
    df["has_ring_encoded"] = label_encoder.fit_transform(df["has_ring"])
    logger.info("Encoded target and binary variables")

    # Handle rare categories (like notebook)
    for col in [
        "habitat",
        "stem_color",
        "gill_color",
        "cap_color",
        "cap_shape",
        "cap_surface",
        "ring_type",
    ]:
        if col in df.columns:
            rare_vals = df[col].value_counts()[df[col].value_counts() < 1000].index
            if len(rare_vals) > 0:
                df[col] = df[col].replace(rare_vals, "Other")
                logger.info(f"Handled rare categories in column: {col}")

    # Drop original categorical columns (like notebook)
    df.drop(columns=["class", "does_bruise_or_bleed", "has_ring"], inplace=True)
    logger.info(
        "Dropped original categorical columns: ['class', 'does_bruise_or_bleed', 'has_ring']"
    )

    # Remove outliers using z-score (like notebook)
    outlier_mask = pd.Series(True, index=df.index)
    for col in ["cap_diameter", "stem_height", "stem_width"]:
        if col in df.columns:
            col_mask = zscore(df[col]) < 2.5
            outlier_mask = outlier_mask & col_mask
            logger.info(f"Applied outlier filter for column: {col}")

    # Apply combined outlier mask
    df = df[outlier_mask]
    logger.info(f"Removed outliers. New shape: {df.shape}")

    df.reset_index(drop=True, inplace=True)

    # One-hot encode categorical variables (exactly like notebook)
    df = pd.get_dummies(df, drop_first=True)
    logger.info("One-hot encoded categorical variables")

    logger.info(f"Data transformation complete. Shape: {df.shape}")
    return df
