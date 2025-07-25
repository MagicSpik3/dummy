"""Unit tests for the ColumnConfig data class.

These tests verify that default values are correctly applied and that custom
values are properly assigned when the configuration object is instantiated.
"""

# REFACTOR: Removed 'import unittest' as it's not needed for pytest.
from survey_assist_utils.configs.column_config import ColumnConfig


# REFACTOR: Converted from a class-based test to simple, standalone functions,
# which is the standard pytest style.
def test_defaults():
    """Test that default values are correctly set when optional arguments are omitted."""
    config = ColumnConfig(
        model_label_cols=["label1"],
        model_score_cols=["score1"],
        clerical_label_cols=["clerical1"],
    )
    # REFACTOR: Replaced unittest's 'self.assertEqual' with a standard 'assert'.
    assert config.id_col == "id"
    # REFACTOR: Replaced unittest's 'self.assertFalse' with a standard 'assert'.
    assert not config.filter_unambiguous


def test_custom_values():
    """Test that custom values are correctly assigned to all fields."""
    config = ColumnConfig(
        model_label_cols=["label1", "label2"],
        model_score_cols=["score1", "score2"],
        clerical_label_cols=["clerical1"],
        id_col="custom_id",
        filter_unambiguous=True,
    )
    assert config.id_col == "custom_id"
    # REFACTOR: Replaced unittest's 'self.assertTrue' with a standard 'assert'.
    assert config.filter_unambiguous

