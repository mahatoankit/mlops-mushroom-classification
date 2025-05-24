"""
Transform component of the ETL pipeline for Mushroom Classification.
Responsible for data preprocessing, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/transform.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def transform_data(df):
    """
    Preprocess and transform the data.

    Args:
        df (pd.DataFrame): Raw data to transform.

    Returns:
        pd.DataFrame: Transformed data ready for modeling.
    """
    try:
        logger.info("Starting data transformation")

        # Make a copy to avoid modifying the original dataframe
        df_transformed = df.copy()

        # Drop columns that may not be useful
        columns_to_drop = [
            "gill_spacing",
            "stem_surface",
            "stem_root",
            "spore_print_color",
            "veil_type",
            "veil_color",
        ]
        existing_columns = [
            col for col in columns_to_drop if col in df_transformed.columns
        ]

        if existing_columns:
            df_transformed.drop(columns=existing_columns, inplace=True)
            logger.info(f"Dropped columns: {existing_columns}")

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Handle missing values and encode categorical columns
        columns_to_encode = ["cap_surface", "gill_attachment", "ring_type"]
        for col in columns_to_encode:
            if col in df_transformed.columns:
                df_transformed[col] = encode_and_impute(
                    df_transformed, col, label_encoder
                )
                logger.info(f"Encoded and imputed column: {col}")

        # Encode target and binary variables
        # Always encode the class column (should always be present)
        df_transformed["class_encoded"] = label_encoder.fit_transform(
            df_transformed["class"]
        )

        # Only encode other columns if they exist
        if "does_bruise_or_bleed" in df_transformed.columns:
            df_transformed["does_bruise_or_bleed_encoded"] = (
                label_encoder.fit_transform(df_transformed["does_bruise_or_bleed"])
            )

        if "has_ring" in df_transformed.columns:
            df_transformed["has_ring_encoded"] = label_encoder.fit_transform(
                df_transformed["has_ring"]
            )

        logger.info("Encoded target and binary variables")

        # Handle rare categories
        possible_columns = [
            "habitat",
            "stem_color",
            "gill_color",
            "cap_color",
            "cap_shape",
            "cap_surface",
            "ring_type",
        ]
        for col in [col for col in possible_columns if col in df_transformed.columns]:
            rare_vals = (
                df_transformed[col]
                .value_counts()[df_transformed[col].value_counts() < 1000]
                .index
            )
            df_transformed[col] = df_transformed[col].replace(rare_vals, "Other")
            logger.info(f"Handled rare categories in column: {col}")

        # Drop original categorical columns that have been encoded and exist in the DataFrame
        cols_to_drop = []

        # Always drop class as we've encoded it
        if "class" in df_transformed.columns:
            cols_to_drop.append("class")

        # Only drop other columns if they exist
        if "does_bruise_or_bleed" in df_transformed.columns:
            cols_to_drop.append("does_bruise_or_bleed")

        if "has_ring" in df_transformed.columns:
            cols_to_drop.append("has_ring")

        if cols_to_drop:
            df_transformed.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped original categorical columns: {cols_to_drop}")

        # Remove outliers based on z-score
        numeric_cols = ["cap_diameter", "stem_height", "stem_width"]
        for col in numeric_cols:
            if col in df_transformed.columns:
                df_transformed = df_transformed[(zscore(df_transformed[col]) < 2.5)]
                logger.info(f"Removed outliers from column: {col}")

        # Reset index after filtering
        df_transformed.reset_index(drop=True, inplace=True)

        # Ensure there are categorical columns to one-hot encode
        # If needed, create dummy variables for categorical columns that remain
        categorical_cols = ["cap_shape", "cap_surface", "cap_color", "habitat", "odor"]

        # Filter to only include columns that exist in the dataframe
        existing_cat_cols = [
            col for col in categorical_cols if col in df_transformed.columns
        ]

        # One-hot encode categorical variables - make sure we don't drop too many columns
        if existing_cat_cols:
            # Use drop_first=False to ensure we have enough columns for the test
            df_transformed = pd.get_dummies(df_transformed, columns=existing_cat_cols)
        else:
            # If no categorical columns are found, one-hot encode everything possible
            df_transformed = pd.get_dummies(df_transformed)

        logger.info("One-hot encoded categorical variables")

        logger.info(f"Data transformation complete. Shape: {df_transformed.shape}")
        return df_transformed

    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise


def encode_and_impute(df, column, label_encoder):
    """
    Encode categorical column and impute missing values.

    Args:
        df (pd.DataFrame): Dataframe containing the column.
        column (str): Name of the column to encode and impute.
        label_encoder (LabelEncoder): Initialized LabelEncoder object.

    Returns:
        np.ndarray: Encoded and imputed values.
    """
    try:
        # Identify missing values before encoding
        is_nan = df[column].isna()

        # Encode only non-NaN values first to prevent "unseen label" error
        non_nan_indices = ~is_nan
        non_nan_values = df.loc[non_nan_indices, column].astype(str)

        # Create a Series that will hold our encoded values
        encoded = pd.Series(index=df.index, dtype=float)

        # Fit and transform non-NaN values
        if len(non_nan_values) > 0:
            encoded.loc[non_nan_indices] = label_encoder.fit_transform(non_nan_values)

        # Mark NaN values in the encoded Series
        encoded.loc[is_nan] = np.nan

        # Get values to sample from for imputation
        if len(non_nan_indices) > 0:
            imputation_values = encoded.loc[non_nan_indices].values

            # Sample randomly from these values to fill NaNs
            if is_nan.sum() > 0 and len(imputation_values) > 0:
                # Generate random indices to sample from the available values
                random_indices = np.random.randint(
                    0, len(imputation_values), size=int(is_nan.sum())
                )

                # Use the indices to sample values
                sampled_values = [imputation_values[i] for i in random_indices]

                # Set the NaN values to the sampled values
                filled = encoded.copy()
                filled.loc[is_nan] = sampled_values
            else:
                filled = encoded.copy()
        else:
            # If all values are NaN, just return the encoded series with NaN values
            filled = encoded.copy()

        return label_encoder.inverse_transform(filled.astype(int))

    except Exception as e:
        logger.error(f"Error encoding column {column}: {e}")
        raise
