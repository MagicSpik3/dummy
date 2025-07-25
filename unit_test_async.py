# In a test file, e.g., tests/test_batch_processor.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the function you want to test
from scripts.process_tlfs_evaluation_data import process_row_minibatch

@pytest.fixture
def mock_app_config():
    """Provides a mock configuration for the tests."""
    return {
        "column_names": {
            "payload_unique_id": "unique_id",
            "payload_job_title": "job_title",
            "payload_job_description": "job_desc",
            "payload_industry_description": "ind_desc",
        }
    }

def test_process_row_minibatch(mock_app_config):
    """
    Tests the asynchronous minibatch processing logic.
    """
    # 1. Create mock data for a minibatch of two rows
    rows = [
        pd.Series({"unique_id": "A", "job_title": "Dev", "job_desc": "Code", "ind_desc": "Tech"}),
        pd.Series({"unique_id": "B", "job_title": "PM", "job_desc": "Manage", "ind_desc": "Business"}),
    ]

    # 2. Create mock responses that grequests.map will return
    # Mock a successful response for the first request
    mock_success_response = MagicMock()
    mock_success_response.json.return_value = {"sic_code": "12345"}
    
    # Mock a failed response for the second request
    mock_failure_response = MagicMock()
    # Make raise_for_status throw an exception
    mock_failure_response.raise_for_status.side_effect = Exception("503 Server Error")

    # 3. Patch grequests.map to return our mock responses
    with patch('grequests.map', return_value=[mock_success_response, mock_failure_response]) as mock_map:
        
        # 4. Call the function with the mock data
        results = process_row_minibatch(rows, "fake_token", mock_app_config)

        # 5. Assert the results
        # Check that we got two results back
        assert len(results) == 2

        # Check the successful result
        assert results[0]["unique_id"] == "A"
        assert results[0]["sic_code"] == "12345"
        assert "error" not in results[0]

        # Check the failed result
        assert results[1]["unique_id"] == "B"
        assert "error" in results[1]
        assert "503 Server Error" in results[1]["error"]
