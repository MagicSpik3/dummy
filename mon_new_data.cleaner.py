"""
survey-assist-utils/src/survey_assist_utils/data_cleaning/data_cleaner.py


This module defines the DataCleaner class, which encapsulates all logic related to
cleaning and preparing raw evaluation data for downstream analysis or modeling.
"""

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from survey_assist_utils.configs.column_config import (
    ColumnConfig,
)

_SIC_CODE_PADDING = 5


class DataCleaner:
    """Handles the cleaning and validation of the raw evaluation DataFrame."""

    _MISSING_VALUE_FORMATS: ClassVar[list[str]] = [
        "", " ", "nan", "None", "Null", "<NA>",
    ]

    def __init__(self, column_config: ColumnConfig):
        """Initializes the data processing class with a column configuration.

        Args:
            column_config (ColumnConfig): Configuration object specifying column-related
                metadata such as which columns to clean, filter, or validate.
        """
        self.config = column_config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full data cleaning and preparation pipeline on a given DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame.
        """
        # REFACTOR: A copy is made to avoid modifying the original input DataFrame.
        # This copy is then passed through the pipeline of cleaning functions.
        working_df = df.copy()
        
        self._validate_inputs(working_df)
        working_df = self._filter_unambiguous(working_df)
        working_df = self._clean_dataframe(working_df)
        
        return working_df

    # REFACTOR: This method now accepts a DataFrame to validate against.
    def _validate_inputs(self, df: pd.DataFrame):
        """Validates that all required inputs are present in the DataFrame."""
        required_cols = [
            self.config.id_col,
            *self.config.model_label_cols,
            *self.config.model_score_cols,
            *self.config.clerical_label_cols,
        ]
        if self.config.filter_unambiguous:
            required_cols.append("Unambiguous")

        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.config.model_label_cols) != len(self.config.model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

    # REFACTOR: This method now accepts a DataFrame, filters it, and returns the result.
    def _filter_unambiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters the DataFrame to retain only unambiguous records, if configured."""
        if self.config.filter_unambiguous:
            if "Unambiguous" not in df.columns:
                return df
            if df["Unambiguous"].dtype != bool:
                df["Unambiguous"] = (
                    df["Unambiguous"]
                    .str.lower()
                    .map({"true": True, "false": False})
                )
            return df[df["Unambiguous"]]
        return df

    @staticmethod
    def _safe_zfill(value: Any) -> Any:
        """Safely pads a numeric value with leading zeros."""
        if pd.isna(value) or value in ["4+", "-9"]:
            return value
        try:
            return str(int(float(value))).zfill(_SIC_CODE_PADDING)
        except (ValueError, TypeError):
            return value

    # REFACTOR: This method now accepts a DataFrame, cleans it, and returns the result.
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by standardizing label columns."""
        label_cols = self.config.model_label_cols + self.config.clerical_label_cols
        df[label_cols] = df[label_cols].astype(str)
        df[label_cols] = df[label_cols].replace(
            self._MISSING_VALUE_FORMATS, np.nan
        )

        for col in label_cols:
            df[col] = df[col].apply(self._safe_zfill)
        
        return df
