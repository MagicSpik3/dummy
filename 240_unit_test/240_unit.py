"""Unit tests for the refactored coder_alignment module.

This test suite verifies the functionality of the refactored classes, ensuring
that each component correctly performs its single responsibility.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Assuming the classes are in a file named coder_alignment.py
from coder_alignment import (
    ColumnConfig,
    DataCleaner,
    MetricCalculator,
    Visualizer,
    LabelAccuracy,
    PlotConfig,
    ConfusionMatrixConfig
)

# --- Main Fixture ---
@pytest.fixture
def raw_data_and_config() -> tuple[pd.DataFrame, ColumnConfig]:
    """A pytest fixture to create a standard set of RAW test data and config."""
    test_data = pd.DataFrame({
        "unique_id": ["A", "B", "C", "D", "E"],
        "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
        "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
        "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
        "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
        "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
        "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
        "Unambiguous": [True, True, False, True, True],
    })
    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config

# --- Tests for DataCleaner ---
class TestDataCleaner:
    """Tests the DataCleaner's ability to validate and clean the DataFrame."""

    def test_cleaning_process(self, raw_data_and_config: tuple):
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

    def test_unambiguous_filter(self, raw_data_and_config: tuple):
        """Tests that the unambiguous filter is applied correctly."""
        df, config = raw_data_and_config
        config.filter_unambiguous = True
        cleaner = DataCleaner(df, config)
        filtered_df = cleaner.process()
        # Row C has Unambiguous=False, so it should be removed.
        assert len(filtered_df) == 4
        assert "C" not in filtered_df["unique_id"].values

    def test_validation_raises_errors(self, raw_data_and_config: tuple):
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


# --- Tests for MetricCalculator ---
class TestMetricCalculator:
    """Tests the MetricCalculator's ability to compute metrics on clean data."""

    @pytest.fixture
    def setup_calculator(self, raw_data_and_config: tuple) -> MetricCalculator:
        """Helper to create a MetricCalculator instance with clean data."""
        df, config = raw_data_and_config
        clean_df = DataCleaner(df, config).process()
        return MetricCalculator(clean_df, config)

    def test_add_derived_columns(self, setup_calculator: MetricCalculator):
        """Tests that derived columns are created with correct values."""
        analyzer = setup_calculator
        # Full matches: A, B. is_correct = [T, T, F, F, F]
        assert analyzer.df["is_correct"].tolist() == [True, True, False, False, False]
        # 2-digit matches: A, B, E. is_correct_2_digit = [T, T, F, F, T]
        assert analyzer.df["is_correct_2_digit"].tolist() == [True, True, False, False, True]
        assert analyzer.df.loc[0, "max_score"] == 0.9

    def test_get_accuracy(self, setup_calculator: MetricCalculator):
        """Tests the get_accuracy method."""
        analyzer = setup_calculator
        # 2 full matches out of 5 = 40%
        assert analyzer.get_accuracy(match_type="full") == pytest.approx(40.0)
        # 3 2-digit matches out of 5 = 60%
        assert analyzer.get_accuracy(match_type="2-digit") == pytest.approx(60.0)

    def test_get_jaccard_similarity(self, setup_calculator: MetricCalculator):
        """Tests the Jaccard similarity calculation."""
        analyzer = setup_calculator
        # A: int=1, uni=3 -> 0.333
        # B: int=1, uni=3 -> 0.333
        # C: int=0, uni=2 -> 0
        # D: int=1, uni=3 -> 0.333
        # E: int=0, uni=3 -> 0
        # Mean = (0.333 + 0.333 + 0 + 0.333 + 0) / 5 = 1.0 / 5 = 0.2
        assert analyzer.get_jaccard_similarity() == pytest.approx(0.2, abs=0.01)


# --- Tests for Visualizer ---
class TestVisualizer:
    """Tests that the Visualizer can call plotting functions without error."""

    def test_plotting_runs_without_error(self, raw_data_and_config: tuple, monkeypatch, tmp_path):
        """Tests that plotting functions run and can save files."""
        monkeypatch.setattr(plt, "show", lambda: None)
        df, config = raw_data_and_config
        
        clean_df = DataCleaner(df, config).process()
        calculator = MetricCalculator(clean_df, config)
        visualizer = Visualizer(calculator.df, calculator)

        # Mock get_threshold_stats as it's not implemented in the provided refactor
        calculator.get_threshold_stats = lambda: pd.DataFrame({
            'threshold': [0.0, 0.5, 1.0],
            'accuracy': [40.0, 50.0, 0.0],
            'coverage': [100.0, 80.0, 0.0]
        })

        plot_conf = PlotConfig(save=True, filename=str(tmp_path / "test.png"))
        matrix_conf = ConfusionMatrixConfig(human_code_col="clerical_label_1", llm_code_col="model_label_1")

        visualizer.plot_threshold_curves(plot_config=plot_conf)
        visualizer.plot_confusion_heatmap(matrix_config=matrix_conf, plot_config=plot_conf)
        
        assert os.path.isfile(plot_conf.filename)


# --- Tests for LabelAccuracy Facade ---
class TestLabelAccuracyFacade:
    """Tests that the main LabelAccuracy class orchestrates the helpers correctly."""

    def test_facade_initialization(self, raw_data_and_config: tuple):
        """Tests that the facade initializes its components."""
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df, config)
        assert isinstance(analyzer.calculator, MetricCalculator)
        assert isinstance(analyzer.visualizer, Visualizer)
        assert len(analyzer.df) == 5 # Check that the final df is available

    def test_facade_delegates_calls(self, raw_data_and_config: tuple):
        """Tests that public methods delegate calls to the correct helper."""
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df, config)
        
        # This result should come from the MetricCalculator via the facade
        accuracy = analyzer.get_accuracy()
        assert accuracy == pytest.approx(40.0)

