"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

This module has been refactored to separate responsibilities into distinct classes:
- DataCleaner: For all data preprocessing and validation.
- MetricCalculator: For all numerical metric computations.
- Visualizer: For all plotting and visual output.
- LabelAccuracy: The main entry point that orchestrates the other classes.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import the moved classes from their new locations.
from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.data_cleaning.data_cleaner import DataCleaner


@dataclass
class PlotConfig:
    """Configuration for plotting functions."""

    figsize: tuple[int, int] = (12, 10)
    save: bool = False
    filename: Optional[str] = None


@dataclass
class ConfusionMatrixConfig:
    """Configuration for the confusion matrix plot."""

    human_code_col: str
    llm_code_col: str
    top_n: int = 10


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
        """Helper to reshape data from wide to long and drop NaNs."""
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


class Visualizer:
    """Handles all visualization tasks."""

    def __init__(self, df: pd.DataFrame, calculator: MetricCalculator):
        """
        Initialises the Visualizer.

        Args:
            df (pd.DataFrame): The processed DataFrame with derived columns.
            calculator (MetricCalculator): An instance of the calculator to generate stats.
        """
        self.df = df
        self.calculator = calculator

    def plot_threshold_curves(self, plot_config: Optional[PlotConfig] = None):
        """
        Plots accuracy and coverage curves against confidence thresholds.

        Args:
            plot_config (Optional[PlotConfig]): Configuration for saving and styling.
        """
        if plot_config is None:
            plot_config = PlotConfig(figsize=(10, 6))
        
        stats_df = self.calculator.get_threshold_stats()
        plt.figure(figsize=plot_config.figsize)
        plt.plot(stats_df["threshold"], stats_df["coverage"], label="Coverage")
        plt.plot(stats_df["threshold"], stats_df["accuracy"], label="Accuracy")
        plt.xlabel("Confidence threshold")
        plt.ylabel("Percentage")
        plt.grid(True)
        plt.legend()
        plt.title("Coverage and Accuracy vs Confidence Threshold")
        plt.tight_layout()
        
        if plot_config.save:
            if not plot_config.filename:
                raise ValueError("Filename must be provided in PlotConfig when save=True.")
            plt.savefig(plot_config.filename)
            plt.close()
        else:
            plt.show()

    def plot_confusion_heatmap(
        self, matrix_config: ConfusionMatrixConfig, plot_config: Optional[PlotConfig] = None
    ):
        """
        Generates a confusion matrix heatmap.

        Args:
            matrix_config (ConfusionMatrixConfig): Config specifying columns and top_n.
            plot_config (Optional[PlotConfig]): Config for saving and styling.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        temp_df = self.df[[matrix_config.human_code_col, matrix_config.llm_code_col]].dropna()
        top_human = temp_df[matrix_config.human_code_col].value_counts().nlargest(matrix_config.top_n).index
        top_llm = temp_df[matrix_config.llm_code_col].value_counts().nlargest(matrix_config.top_n).index
        
        filtered_df = temp_df[temp_df[matrix_config.human_code_col].isin(top_human) & temp_df[matrix_config.llm_code_col].isin(top_llm)]
        
        if filtered_df.empty:
            print("No overlapping data for top codes.")
            return
        
        matrix = pd.crosstab(filtered_df[matrix_config.human_code_col], filtered_df[matrix_config.llm_code_col])
        plt.figure(figsize=plot_config.figsize)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"Confusion Matrix: Top {matrix_config.top_n} Codes")
        plt.ylabel(f"Human Coder ({matrix_config.human_code_col})")
        plt.xlabel(f"LLM Prediction ({matrix_config.llm_code_col})")
        plt.tight_layout()

        if plot_config.save:
            if not plot_config.filename:
                raise ValueError("Filename must be provided in PlotConfig when save=True.")
            plt.savefig(plot_config.filename)
            plt.close()
        else:
            plt.show()


class LabelAccuracy:
    """Orchestrates data cleaning, metric calculation, and visualization."""

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """
        Initialises the full analysis pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.
            column_config (ColumnConfig): The configuration for the analysis.
        """
        cleaner = DataCleaner(df, column_config)
        clean_df = cleaner.process()
        self.calculator = MetricCalculator(clean_df, column_config)
        self.df = self.calculator.df
        self.visualizer = Visualizer(self.df, self.calculator)

    def get_accuracy(self, **kwargs) -> Union[float, dict[str, Any]]:
        """Delegates call to MetricCalculator.get_accuracy."""
        return self.calculator.get_accuracy(**kwargs)

    def get_jaccard_similarity(self, **kwargs) -> float:
        """Delegates call to MetricCalculator.get_jaccard_similarity."""
        return self.calculator.get_jaccard_similarity(**kwargs)

    def get_coverage(self, **kwargs) -> float:
        """Delegates call to MetricCalculator.get_coverage."""
        return self.calculator.get_coverage(**kwargs)

    def get_summary_stats(self, **kwargs) -> dict[str, Any]:
        """Delegates call to MetricCalculator.get_summary_stats."""
        return self.calculator.get_summary_stats(**kwargs)

    def plot_confusion_heatmap(self, **kwargs):
        """Delegates call to Visualizer.plot_confusion_heatmap."""
        return self.visualizer.plot_confusion_heatmap(**kwargs)

    def plot_threshold_curves(self, **kwargs):
        """Delegates call to Visualizer.plot_threshold_curves."""
        return self.visualizer.plot_threshold_curves(**kwargs)

    @staticmethod
    def save_output(metadata: dict, eval_result: dict, save_path: str = "data/") -> str:
        """
        Save evaluation results to files.

        Args:
            metadata (dict): Dictionary of metadata parameters.
            eval_result (dict): Dictionary containing evaluation metrics.
            save_path (str): The folder where results should be saved.

        Returns:
            str: The folder path where results were stored.
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(save_path, f"outputs/{dt_str}_{metadata.get('evaluation_type', 'unnamed')}")
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        with open(os.path.join(folder_name, "evaluation_result.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4)
        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name
