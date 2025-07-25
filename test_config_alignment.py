"""Unit tests for the refactored coder_alignment module.

This test suite verifies the functionality of the refactored classes, ensuring
that each component correctly performs its single responsibility.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import pytest

# REFACTOR: Import the new helper classes alongside the main facade.
from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    ConfusionMatrixConfig,
    DataCleaner,
    LabelAccuracy,
    MetricCalculator,
    PlotConfig,
    Visualizer,
)

# NOTE: The 'raw_data_and_config' fixture is automatically discovered
# by pytest from the 'conftest.py' file and provides the raw data.


# --- Tests for MetricCalculator ---
class TestMetricCalculator:
    """Tests the MetricCalculator's ability to compute metrics on clean data."""

    @pytest.fixture
    def setup_calculator(self, raw_data_and_config: tuple) -> MetricCalculator:
        """
        Helper fixture to create a MetricCalculator instance with clean data.

        This fixture first runs the DataCleaner on the raw data, then uses the
        resulting clean DataFrame to initialize the MetricCalculator.

        Args:
            raw_data_and_config (tuple): The pytest fixture providing raw data and config.

        Returns:
            MetricCalculator: An instance ready for testing.
        """
        df, config = raw_data_and_config
        clean_df = DataCleaner(df, config).process()
        return MetricCalculator(clean_df, config)

    def test_add_derived_columns(self, setup_calculator: MetricCalculator):
        """Tests that derived columns are created with correct values."""
        analyzer = setup_calculator
        # Full matches: A, B. is_correct = [True, True, False, False, False]
        assert analyzer.df["is_correct"].tolist() == [True, True, False, False, False]
        # 2-digit matches: A, B, E. is_correct_2_digit = [True, True, False, False, True]
        assert analyzer.df["is_correct_2_digit"].tolist() == [
            True,
            True,
            False,
            False,
            True,
        ]
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
        # C: int=0, uni=1 -> 0
        # D: int=1, uni=2 -> 0.5
        # E: int=1, uni=3 -> 0.333
        # Mean = (0.333 + 0.333 + 0 + 0.5 + 0.333) / 5 = 1.5 / 5 = 0.3
        assert analyzer.get_jaccard_similarity() == pytest.approx(0.3, abs=0.01)


# --- Tests for Visualizer ---
class TestVisualizer:
    """Tests that the Visualizer can call plotting functions without error."""

    def test_plotting_runs_without_error(
        self, raw_data_and_config: tuple, monkeypatch, tmp_path
    ):
        """Tests that plotting functions run and can save files."""
        monkeypatch.setattr(plt, "show", lambda: None)
        df, config = raw_data_and_config

        clean_df = DataCleaner(df, config).process()
        calculator = MetricCalculator(clean_df, config)
        visualizer = Visualizer(calculator.df, calculator)

        # Mock get_threshold_stats as it's not implemented in the provided refactor
        def mock_get_threshold_stats(thresholds=None):
            return pd.DataFrame(
                {
                    "threshold": [0.0, 0.5, 1.0],
                    "accuracy": [40.0, 50.0, 0.0],
                    "coverage": [100.0, 80.0, 0.0],
                }
            )

        calculator.get_threshold_stats = mock_get_threshold_stats

        plot_conf = PlotConfig(save=True, filename=str(tmp_path / "test.png"))
        matrix_conf = ConfusionMatrixConfig(
            human_code_col="clerical_label_1", llm_code_col="model_label_1"
        )

        visualizer.plot_threshold_curves(plot_config=plot_conf)
        visualizer.plot_confusion_heatmap(
            matrix_config=matrix_conf, plot_config=plot_conf
        )

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
        assert len(analyzer.df) == 5  # Check that the final df is available

    def test_facade_delegates_calls(self, raw_data_and_config: tuple):
        """Tests that public methods delegate calls to the correct helper."""
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df, config)

        # This result should come from the MetricCalculator via the facade
        accuracy = analyzer.get_accuracy()
        assert accuracy == pytest.approx(40.0)
