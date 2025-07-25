"""This module contains pytest configuration and shared fixtures.

Fixtures defined in this file are automatically discovered by pytest and can be
used in any test file within this directory without needing to be imported.

Fixtures:
    raw_data_and_config: Provides a consistent, raw DataFrame and ColumnConfig
                         object for use across multiple test suites.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to Python path to ensure modules can be found
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# REFACTOR: Import the config class to be used in the shared fixture.
from survey_assist_utils.configs.column_config import ColumnConfig

# Configure a global logger for the test suite
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# REFACTOR: The sample_data_and_config fixture has been moved here from the
# test_coder_alignment.py file so it can be shared across all test files.
# It is now renamed to 'raw_data_and_config' for clarity.
@pytest.fixture
def raw_data_and_config() -> tuple[pd.DataFrame, ColumnConfig]:
    """A pytest fixture to create a standard set of RAW test data and config."""
    test_data = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
            "Unambiguous": [True, True, False, True, True],
        }
    )

    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config


def pytest_configure(config):  # pylint: disable=unused-argument
    """Hook function for pytest global configuration."""
    logger.info("=== Global Test Configuration Applied ===")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Logs the start of a test session."""
    logger.info("=== Test Session Started ===")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Logs the end of a test session."""
    logger.info("=== Test Session Finished ===")

