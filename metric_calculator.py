"""
This module defines the MetricCalculator class, which is responsible for
computing all numerical evaluation metrics from a clean DataFrame.
"""
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig


class MetricCalculator:
    """Calculates all numerical evaluation metrics from a clean DataFrame."""

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """
        Initialises the MetricCalculator.

        Args:
            df (pd.DataFrame): A clean DataFrame, post-DataCleaner.
            column_config (ColumnConfig): The configuration for the analysis.
        """
        self.df = df.copy()
        self.config = column_config
        self._add_derived_columns()

    def _melt_and_clean(self, value_vars: list[str], value_name: str) -> pd.DataFrame:
        """
        Helper to reshape data from wide to long and drop NaNs.

        Args:
            value_vars (list[str]): A list of columns to melt.
            value_name (str): The name for the new value column.

        Returns:
            pd.DataFrame: A cleaned, long-format DataFrame.
        """
        melted_df = self.df.melt(
            id_vars=[self.config.id_col], value_vars=value_vars, value_name=value_name
        )
        return melted_df.dropna(subset=[value_name])

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches."""
        model_melted = self._melt_and_clean(self.config.model_label_cols, "model_label")
        clerical_melted = self._melt_and_clean(
            self.config.clerical_label_cols, "clerical_label"
        )

        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.config.id_col, "model_label"],
            right_on=[self.config.id_col, "clerical_label"],
        )
        full_match_ids = full_matches[self.config.id_col].unique()

        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted[
            "clerical_label"
        ].str[:2]

        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.config.id_col, "model_label_2_digit"],
            right_on=[self.config.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.config.id_col].unique()

        self.df["is_correct"] = self.df[self.config.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.config.id_col].isin(
            partial_match_ids
        )

        for col in self.config.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["max_score"] = self.df[self.config.model_score_cols].max(axis=1)

    def get_jaccard_similarity(self) -> float:
        """Calculates the average Jaccard Similarity Index."""
        def calculate_jaccard_for_row(row):
            model_set = set(row[self.config.model_label_cols].dropna())
            clerical_set = set(row[self.config.clerical_label_cols].dropna())
            if not model_set and not clerical_set:
                return 1.0
            intersection = len(model_set.intersection(clerical_set))
            union = len(model_set.union(clerical_set))
            return intersection / union if union > 0 else 0.0

        return self.df.apply(calculate_jaccard_for_row, axis=1).mean()

    def get_accuracy(
        self, threshold: float = 0.0, match_type: str = "full", extended: bool = False
    ) -> Union[float, dict[str, Any]]:
        """
        Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold. Defaults to 0.0.
            match_type (str): 'full' or '2-digit'. Defaults to "full".
            extended (bool): If True, returns a detailed dictionary. Defaults to False.

        Returns:
            Union[float, dict[str, Any]]: The accuracy percentage or a detailed dictionary.
        """
        correct_col = "is_correct_2_digit" if match_type == "2-digit" else "is_correct"
        filtered_df = self.df[self.df["max_score"] >= threshold]
        total = len(filtered_df)
        if total == 0:
            return {"accuracy_percent": 0.0, "matches": 0, "total_considered": 0} if extended else 0.0
        
        matches = filtered_df[correct_col].sum()
        accuracy = 100 * matches / total
        
        if extended:
            return {"accuracy_percent": round(accuracy, 1), "matches": int(matches), "non_matches": int(total - matches), "total_considered": total}
        return accuracy

    def get_coverage(self, threshold: float = 0.0) -> float:
        """
        Calculate the percentage of predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold. Defaults to 0.0.

        Returns:
            float: Coverage as a percentage.
        """
        return 100 * (self.df["max_score"] >= threshold).mean()

    def get_threshold_stats(self, thresholds: Optional[list[float]] = None) -> pd.DataFrame:
        """
        Calculate accuracy and coverage across multiple thresholds.

        Args:
            thresholds (Optional[list[float]]): A list of thresholds to evaluate.
                                                Defaults to a range from 0 to 1.

        Returns:
            pd.DataFrame: A DataFrame with columns for threshold, accuracy, and coverage.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21).tolist()
        stats = [{"threshold": t, "accuracy": self.get_accuracy(t), "coverage": self.get_coverage(t)} for t in thresholds]
        return pd.DataFrame(stats)

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get a dictionary of summary statistics for the classification results.

        Returns:
            dict[str, Any]: A dictionary containing various summary statistics.
        """
        return {
            "total_samples": len(self.df),
            "overall_accuracy": self.get_accuracy(),
            "accuracy_above_0.80": self.get_accuracy(0.8),
            "coverage_above_0.80": self.get_coverage(0.8),
        }
