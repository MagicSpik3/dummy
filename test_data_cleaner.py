"""Unit tests for the DataCleaner class.

These tests verify the correctness of the data cleaning pipeline, including:
- Input validation
- Filtering of ambiguous records
- Standardization of label formats
- Handling of missing values
"""

# REFACTOR: Removed 'import unittest' as it's not needed for pytest.
import pandas as pd
import pytest

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.data_cleaning.data_cleaner import DataCleaner


# NOTE: The 'raw_data_and_config' fixture is automatically discovered
# by pytest from the 'conftest.py' file.

# --- Tests for DataCleaner ---
class TestDataCleaner:
    """Tests the DataCleaner's ability to validate and clean the DataFrame."""

    def test_cleaning_process(
        self, raw_data_and_config: tuple[pd.DataFrame, ColumnConfig]
    ):
        """Tests that the main process method correctly cleans data."""
        df, config = raw_data_and_config
        cleaner = DataCleaner(df, config)
        clean_df = cleaner.process()

        # Check zfill padding
        assert clean_df.loc[1, "clerical_label_1"] == "01234"
        # Check special value preservation
        assert clean_df.loc[2, "clerical_label_1"] == "-9"
        # Check that string 'nan' and empty strings become real NaNs
        assert pd.isna(clean_df.loc[3, "clerical_label_1"])
        assert pd.isna(clean_df.loc[3, "clerical_label_2"])

    def test_unambiguous_filter(
        self, raw_data_and_config: tuple[pd.DataFrame, ColumnConfig]
    ):
        """Tests that the unambiguous filter is applied correctly."""
        df, config = raw_data_and_config
        config.filter_unambiguous = True
        cleaner = DataCleaner(df, config)
        filtered_df = cleaner.process()
        # Row C has Unambiguous=False, so it should be removed.
        assert len(filtered_df) == 4
        assert "C" not in filtered_df["unique_id"].values

    def test_validation_raises_errors(
        self, raw_data_and_config: tuple[pd.DataFrame, ColumnConfig]
    ):
        """Tests that validation raises ValueErrors for bad configuration."""
        df, _ = raw_data_and_config
        bad_config = ColumnConfig(
            model_label_cols=["model_label_1", "missing_col"],
            model_score_cols=["model_score_1"],
            clerical_label_cols=["clerical_label_1"],
            id_col="unique_id",
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            DataCleaner(df, bad_config).process()

# REFACTOR: Removed the unnecessary 'if __name__ == "__main__":' block.
# Pytest discovers and runs tests automatically.
