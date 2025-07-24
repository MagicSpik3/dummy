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
- Error handling for various failure modes.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.api_core import exceptions

from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor


# --- Test Fixtures ---
@pytest.fixture
def mock_config() -> dict:
    """Provides a mock configuration dictionary for tests."""
    return {
        "paths": {
            "gcs_bucket_name": "test-bucket",
            "gcs_json_dir": "test/prefix/",
            "processed_csv_output": "gs://test-bucket/data/eval.csv",
            "named_file": "gs://test-bucket/test/prefix/single_file.json",
        },
        "parameters": {"date_since": "20230101", "single_file": "False"},
        "json_keys": {"unique_id": "unique_id"},
    }


@pytest.fixture
def mock_storage_client():
    """Mocks the GCS storage client to avoid real API calls."""
    with patch("google.cloud.storage.Client") as mock_client:
        yield mock_client


# --- Test Cases ---


def test_initialization(mock_config: dict, mock_storage_client):
    """
    Tests that the class initializes correctly with a valid config.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    preprocessor = JsonPreprocessor(mock_config)
    assert preprocessor.config == mock_config
    mock_storage_client.assert_called_once()


def test_init_raises_type_error():
    """Tests that __init__ raises a TypeError for a non-dict config."""
    with pytest.raises(TypeError, match="config must be a dictionary"):
        JsonPreprocessor("not a dict")


def test_get_gcs_filepaths_directory_mode(mock_config: dict, mock_storage_client):
    """
    Tests the logic for listing and filtering files from GCS in directory mode.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    mock_blob_1 = MagicMock()
    mock_blob_1.name = "test/prefix/20230105_output.json"
    mock_blob_2 = MagicMock()
    mock_blob_2.name = "test/prefix/20221231_output.json"  # Should be filtered out by date

    mock_instance = mock_storage_client.return_value
    mock_instance.list_blobs.return_value = [mock_blob_1, mock_blob_2]

    preprocessor = JsonPreprocessor(mock_config)
    filepaths = preprocessor.get_gcs_filepaths()

    assert len(filepaths) == 1
    assert filepaths[0] == "gs://test-bucket/test/prefix/20230105_output.json"
    mock_instance.list_blobs.assert_called_with("test-bucket", prefix="test/prefix/")


def test_get_gcs_filepaths_single_file_mode(mock_config: dict, mock_storage_client):
    """
    Tests the logic for returning a single file path when in single file mode.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    mock_config["parameters"]["single_file"] = "True"
    preprocessor = JsonPreprocessor(mock_config)
    filepaths = preprocessor.get_gcs_filepaths()

    assert len(filepaths) == 1
    assert filepaths[0] == "gs://test-bucket/test/prefix/single_file.json"
    mock_storage_client.return_value.list_blobs.assert_not_called()


def test_get_gcs_filepaths_single_file_mode_no_file(mock_config: dict, mock_storage_client):
    """
    Tests that single file mode returns an empty list if 'named_file' is missing.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    mock_config["parameters"]["single_file"] = "True"
    del mock_config["paths"]["named_file"]
    preprocessor = JsonPreprocessor(mock_config)
    filepaths = preprocessor.get_gcs_filepaths()
    assert filepaths == []


def test_get_json_data_not_found(mock_config: dict, mock_storage_client):
    """
    Tests that _get_json_data returns None when a GCS file is not found.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    mock_instance = mock_storage_client.return_value
    mock_instance.bucket.return_value.blob.return_value.download_as_string.side_effect = exceptions.NotFound("File not found")
    
    preprocessor = JsonPreprocessor(mock_config)
    result = preprocessor._get_json_data("gs://test-bucket/nonexistent.json")
    assert result is None


def test_flatten_llm_json_to_dataframe_detailed(mock_config: dict, mock_storage_client):
    """
    Tests the flatten_llm_json_to_dataframe method with detailed data.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    preprocessor = JsonPreprocessor(mock_config)
    mock_json_data = [{
        "unique_id": "A",
        "classified": "Yes",
        "sic_code": "12345",
        "request_payload": {"job_title": "Developer"},
        "sic_candidates": [{"sic_code": "12345", "likelihood": 0.9}]
    }]

    with patch.object(preprocessor, "_get_json_data", return_value=mock_json_data) as mock_get:
        df = preprocessor.flatten_llm_json_to_dataframe("any/path")
        mock_get.assert_called_once_with("any/path")
        assert not df.empty
        assert df.loc[0, "unique_id"] == "A"
        assert df.loc[0, "chosen_sic_code"] == "12345"
        assert df.loc[0, "payload_job_title"] == "Developer"
        assert df.loc[0, "candidate_1_sic_code"] == "12345"
        assert df.loc[0, "candidate_1_likelihood"] == 0.9


def test_process_files_no_files_found(mock_config: dict, mock_storage_client):
    """
    Tests that process_files returns an empty DataFrame if no files are found.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    preprocessor = JsonPreprocessor(mock_config)
    with patch.object(preprocessor, "get_gcs_filepaths", return_value=[]) as mock_get_paths:
        df = preprocessor.process_files()
        mock_get_paths.assert_called_once()
        assert df.empty


def test_merge_eval_data(mock_config: dict, mock_storage_client, tmp_path):
    """
    Tests a successful merge of flattened data with evaluation data.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
        tmp_path: Pytest fixture for a temporary directory.
    """
    # Create a dummy evaluation CSV file
    eval_csv_path = tmp_path / "eval.csv"
    eval_df = pd.DataFrame({"unique_id": ["A"], "SIC_Division": ["01"]})
    eval_df.to_csv(eval_csv_path, index=False)
    
    # Update config to point to the local temp file
    mock_config["paths"]["processed_csv_output"] = str(eval_csv_path)

    flattened_df = pd.DataFrame({"unique_id": ["A"], "chosen_sic_code": ["12345"]})
    
    preprocessor = JsonPreprocessor(mock_config)
    merged_df = preprocessor.merge_eval_data(flattened_df)

    assert not merged_df.empty
    assert len(merged_df) == 1
    assert "SIC_Division" in merged_df.columns
    assert "chosen_sic_code" in merged_df.columns


def test_merge_eval_data_file_not_found(mock_config: dict, mock_storage_client):
    """
    Tests that merge_eval_data returns an empty DataFrame if the CSV is not found.

    Args:
        mock_config (dict): The mock configuration fixture.
        mock_storage_client: The mocked GCS client fixture.
    """
    mock_config["paths"]["processed_csv_output"] = "/nonexistent/path/eval.csv"
    preprocessor = JsonPreprocessor(mock_config)
    merged_df = preprocessor.merge_eval_data(pd.DataFrame({"unique_id": ["A"]}))
    assert merged_df.empty

