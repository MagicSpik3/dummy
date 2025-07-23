"""Unit tests for the JsonPreprocessor class.

This module contains a suite of unit tests for the `JsonPreprocessor` class,
which is responsible for processing raw JSON files from Google Cloud Storage.
The tests use the `pytest` framework and `unittest.mock` to isolate the
class from external dependencies, and do not require a live GCS connection.

Key areas tested include:
- Correct initialisation of the class with a configuration object.
- Verification of the GCS file discovery and filtering logic.
- Accurate counting of records within mock JSON data.
- Deduplication of records based on a unique id.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor

# --- Test Fixtures ---


@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for tests."""
    return {
        "paths": {
            "gcs_bucket_name": "test-bucket",
            "gcs_json_dir": "test/prefix/",
            "processed_csv_output": "gs://test-bucket/data/eval.csv",
        },
        "parameters": {"date_since": "20230101"},
        "json_keys": {"unique_id": "unique_id"},
    }


@pytest.fixture
def mock_storage_client():
    """Mocks the GCS storage client to avoid real API calls."""
    with patch("google.cloud.storage.Client") as mock_client:
        yield mock_client


# --- Test Cases ---


def test_initialization(
    mock_config, mock_storage_client
):  # pylint: disable=redefined-outer-name
    """Test that the class initializes correctly with a config."""
    preprocessor = JsonPreprocessor(mock_config)
    assert preprocessor.config == mock_config
    # Check that the storage client was instantiated
    mock_storage_client.assert_called_once()


def test_get_gcs_filepaths(
    mock_config, mock_storage_client
):  # pylint: disable=redefined-outer-name
    """Test the logic for listing and filtering files from GCS."""
    # 1. Setup mock blobs that the client will return
    mock_blob_1 = MagicMock()
    mock_blob_1.name = "test/prefix/20230105_output.json"

    mock_blob_2 = MagicMock()
    # Should be filtered out by date
    mock_blob_2.name = "test/prefix/20221231_output.json"

    mock_blob_3 = MagicMock()
    # Should be filtered out (in sub-directory)
    mock_blob_3.name = "test/prefix/subfolder/20230106_output.json"

    mock_blob_4 = MagicMock()
    mock_blob_4.name = (
        "test/prefix/20230201_another.json"  # Not ending with _output.json
    )

    mock_blob_5 = MagicMock()
    mock_blob_5.name = "test/prefix/20230301_output.json"  # A valid file

    # Configure the mock client's list_blobs method to return our mock blobs
    mock_instance = mock_storage_client.return_value
    mock_instance.list_blobs.return_value = [
        mock_blob_1,
        mock_blob_2,
        mock_blob_3,
        mock_blob_4,
        mock_blob_5,
    ]

    # 2. Instantiate the class and call the method
    preprocessor = JsonPreprocessor(mock_config)
    filepaths = preprocessor.get_gcs_filepaths()

    # 3. Assert the results
    expected_paths = [
        "gs://test-bucket/test/prefix/20230105_output.json",
        "gs://test-bucket/test/prefix/20230301_output.json",
    ]

    assert len(filepaths) == 2  # noqa: PLR2004
    assert sorted(filepaths) == sorted(expected_paths)
    mock_instance.list_blobs.assert_called_with("test-bucket", prefix="test/prefix/")


# FIX: Added 'mock_storage_client' to the function signature to activate the mock.
def test_record_count(
    mock_config, mock_storage_client
):  # pylint: disable=redefined-outer-name
    """Test counting records from a mock JSON response."""
    preprocessor = JsonPreprocessor(mock_config)

    # Mock the _get_json_data method to return controlled data
    with patch.object(preprocessor, "_get_json_data") as mock_get_data:
        # Case 1: List of records
        mock_get_data.return_value = [{"id": 1}, {"id": 2}]
        assert preprocessor.record_count("any/path") == 2  # noqa: PLR2004

        # Case 2: Single record in a dict
        mock_get_data.return_value = {"id": 1}
        assert preprocessor.record_count("any/path") == 1

        # Case 3: Empty list
        mock_get_data.return_value = []
        assert preprocessor.record_count("any/path") == 0

        # Case 4: Invalid data
        mock_get_data.return_value = None
        assert preprocessor.record_count("any/path") == 0


# FIX: Added 'mock_storage_client' to the function signature to activate the mock.
def test_process_files_deduplication(
    mock_config, mock_storage_client
):  # pylint: disable=redefined-outer-name
    """Test that process_files correctly handles and removes duplicates."""
    preprocessor = JsonPreprocessor(mock_config)

    # Create two dataframes, one with a duplicate unique_id
    df1_data = {"unique_id": ["A", "B"], "candidate_1_sic_code": ["1111", "2222"]}
    df2_data = {
        "unique_id": ["A", "C"],  # "A" is a duplicate
        "candidate_1_sic_code": ["1111_new", "3333"],
    }
    df1 = pd.DataFrame(df1_data)
    df2 = pd.DataFrame(df2_data)

    # Mock the methods that lead up to the final processing
    with patch.object(
        preprocessor, "get_gcs_filepaths"
    ) as mock_get_paths, patch.object(
        preprocessor, "flatten_llm_json_to_dataframe"
    ) as mock_flatten:

        mock_get_paths.return_value = ["path1", "path2"]
        # Have flatten return our predefined dataframes
        mock_flatten.side_effect = [df1, df2]

        result_df = preprocessor.process_files()

        # After concat, there are 4 rows. After drop_duplicates, there should be 3.
        assert len(result_df) == 3  # noqa: PLR2004
        assert sorted(result_df["unique_id"].tolist()) == ["A", "B", "C"]
        # Check that the first instance of 'A' was kept
        assert (
            result_df[result_df["unique_id"] == "A"]["candidate_1_sic_code"].iloc[0]
            == "1111"
        )
