"""Unit tests for the refactored coder_alignment module.

This test suite verifies the functionality of the refactored classes, ensuring
that each component correctly performs its single responsibility.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import pytest

# Import all the refactored classes
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
# by pytest from the 'conftest.py' file.


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
        calc = setup_calculator
        # Full matches: A, B, D. is_correct = [True, True, False, True, False]
        assert calc.df["is_correct"].tolist() == [True, True, False, True, False]
        # 2-digit matches: A, B, D, E. is_correct_2_digit = [True, True, False, True, True]
        assert calc.df["is_correct_2_digit"].tolist() == [
            True,
            True,
            False,
            True,
            True,
        ]
        assert calc.df.loc[0, "max_score"] == 0.9

    def test_get_accuracy(self, setup_calculator: MetricCalculator):
        """Tests the get_accuracy method."""
        calc = setup_calculator
        # 3 full matches out of 5 = 60%
        assert calc.get_accuracy(match_type="full") == pytest.approx(60.0)
        # 4 2-digit matches out of 5 = 80%
        assert calc.get_accuracy(match_type="2-digit") == pytest.approx(80.0)

    def test_get_jaccard_similarity(self, setup_calculator: MetricCalculator):
        """Tests the Jaccard similarity calculation."""
        calc = setup_calculator
        # A: int=1, uni=4 -> 0.25
        # B: int=1, uni=3 -> 0.333
        # C: int=0, uni=2 -> 0
        # D: int=1, uni=3 -> 0.333
        # E: int=0, uni=3 -> 0
        # Mean = (0.25 + 0.333 + 0 + 0.333 + 0) / 5 = 0.916 / 5 = 0.183
        assert calc.get_jaccard_similarity() == pytest.approx(0.183, abs=0.01)

    def test_get_coverage(self, setup_calculator: MetricCalculator):
        """Tests the get_coverage method."""
        calc = setup_calculator
        # 4 of 5 scores are >= 0.8, so coverage is 80%
        assert calc.get_coverage(threshold=0.8) == pytest.approx(80.0)
        # All scores are >= 0.1, so coverage is 100%
        assert calc.get_coverage(threshold=0.1) == pytest.approx(100.0)

    def test_get_summary_stats(self, setup_calculator: MetricCalculator):
        """Tests that get_summary_stats returns a dictionary with correct values."""
        calc = setup_calculator
        stats = calc.get_summary_stats()
        assert isinstance(stats, dict)
        assert stats["total_samples"] == 5
        assert stats["overall_accuracy"] == pytest.approx(60.0)


# --- Tests for Visualizer ---
class TestVisualizer:
    """Tests that the Visualizer can call plotting functions without error."""

    def test_plotting_runs_without_error(
        self, raw_data_and_config: tuple, monkeypatch, tmp_path
    ):
        """
        Tests that plotting functions run and can save files.

        Args:
            raw_data_and_config (tuple): The pytest fixture providing raw data and config.
            monkeypatch: Pytest fixture for modifying classes/functions.
            tmp_path: Pytest fixture for a temporary directory.
        """
        monkeypatch.setattr(plt, "show", lambda: None)
        df, config = raw_data_and_config
        
        clean_df = DataCleaner(df, config).process()
        calculator = MetricCalculator(clean_df, config)
        visualizer = Visualizer(calculator.df, calculator)

        # FIX: The mock function's signature must match the original function's signature.
        # The original get_threshold_stats accepts an optional 'thresholds' argument.
        def mock_get_threshold_stats(thresholds=None):
            return pd.DataFrame(
                {
                    "threshold": [0.0, 0.5, 1.0],
                    "accuracy": [60.0, 75.0, 0.0],
                    "coverage": [100.0, 80.0, 0.0],
                }
            )

        # Use monkeypatch to replace the method for the duration of this test.
        monkeypatch.setattr(calculator, "get_threshold_stats", mock_get_threshold_stats)

        plot_conf = PlotConfig(save=True, filename=str(tmp_path / "test.png"))
        matrix_conf = ConfusionMatrixConfig(human_code_col="clerical_label_1", ll_code_col="model_label_1")

        visualizer.plot_threshold_curves(plot_config=plot_conf)
        visualizer.plot_confusion_heatmap(matrix_config=matrix_conf, plot_config=plot_conf)
        
        assert os.path.isfile(plot_conf.filename)


# --- Tests for LabelAccuracy Facade ---
class TestLabelAccuracyFacade:
    """Tests that the main LabelAccuracy class orchestrates the helpers correctly."""

    def test_facade_initialization(self, raw_data_and_config: tuple):
        """
        Tests that the facade initializes its components.

        Args:
            raw_data_and_config (tuple): The pytest fixture providing raw data and config.
        """
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df, config)
        assert isinstance(analyzer.calculator, MetricCalculator)
        assert isinstance(analyzer.visualizer, Visualizer)
        assert len(analyzer.df) == 5

    def test_facade_delegates_calls(self, raw_data_and_config: tuple):
        """
        Tests that public methods delegate calls to the correct helper.

        Args:
            raw_data_and_config (tuple): The pytest fixture providing raw data and config.
        """
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df, config)
        
        # This result should come from the MetricCalculator via the facade
        accuracy = analyzer.get_accuracy()
        assert accuracy == pytest.approx(60.0)

        # ADDED: Test for the newly added get_threshold_stats passthrough method.
        stats = analyzer.get_threshold_stats(thresholds=[0.5])
        assert isinstance(stats, pd.DataFrame)
        assert "accuracy" in stats.columns
