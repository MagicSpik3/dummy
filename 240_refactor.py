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
from typing import Any, ClassVar, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The standard character length for a fully-padded SIC code.
_SIC_CODE_PADDING = 5


@dataclass
class ColumnConfig:
    """A data structure to hold the name configurations for the analysis."""
    model_label_cols: list[str]
    model_score_cols: list[str]
    clerical_label_cols: list[str]
    id_col: str = "id"
    filter_unambiguous: bool = False

# REFACTOR: A new dataclass to handle plotting arguments cleanly.
@dataclass
class PlotConfig:
    """Configuration for plotting functions."""
    figsize: tuple[int, int] = (12, 10)
    save: bool = False
    filename: Optional[str] = None

# REFACTOR: A new dataclass for the confusion matrix to keep the main
# function signature clean and within linting limits.
@dataclass
class ConfusionMatrixConfig:
    """Configuration for the confusion matrix plot."""
    human_code_col: str
    llm_code_col: str
    top_n: int = 10


# REFACTOR: This new class handles all data cleaning and preparation.
# Its only responsibility is to take a raw DataFrame and return a clean one.
class DataCleaner:
    """Handles the cleaning and validation of the raw evaluation DataFrame."""

    _MISSING_VALUE_FORMATS: ClassVar[list[str]] = [
        "", " ", "nan", "None", "Null", "<NA>",
    ]

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        self.df = df.copy()
        self.config = column_config

    def process(self) -> pd.DataFrame:
        """Main method to run the entire cleaning and preparation pipeline."""
        self._validate_inputs()
        self._filter_unambiguous()
        self._clean_dataframe()
        return self.df

    def _validate_inputs(self):
        """Centralised method for all input validations."""
        required_cols = [
            self.config.id_col,
            *self.config.model_label_cols,
            *self.config.model_score_cols,
            *self.config.clerical_label_cols,
        ]
        if self.config.filter_unambiguous:
            required_cols.append("Unambiguous")

        if missing_cols := [col for col in required_cols if col not in self.df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.config.model_label_cols) != len(self.config.model_score_cols):
            raise ValueError("Number of model label columns must match number of score columns")

    def _filter_unambiguous(self):
        """Filters the DataFrame for unambiguous records if configured to do so."""
        if self.config.filter_unambiguous:
            if "Unambiguous" not in self.df.columns:
                return
            if self.df["Unambiguous"].dtype != bool:
                self.df["Unambiguous"] = self.df["Unambiguous"].str.lower().map({"true": True, "false": False})
            self.df = self.df[self.df["Unambiguous"]]

    @staticmethod
    def _safe_zfill(value: Any) -> Any:
        """Safely pads a value with leading zeros to 5 digits."""
        if pd.isna(value) or value in ["4+", "-9"]:
            return value
        try:
            return str(int(float(value))).zfill(_SIC_CODE_PADDING)
        except (ValueError, TypeError):
            return value

    def _clean_dataframe(self):
        """Cleans the DataFrame by handling data types and missing values."""
        label_cols = self.config.model_label_cols + self.config.clerical_label_cols
        self.df[label_cols] = self.df[label_cols].astype(str)
        self.df[label_cols] = self.df[label_cols].replace(self._MISSING_VALUE_FORMATS, np.nan)

        for col in label_cols:
            self.df[col] = self.df[col].apply(self._safe_zfill)


# REFACTOR: This new class handles all numerical metric calculations.
# It takes a CLEAN DataFrame and focuses only on computation.
class MetricCalculator:
    """Calculates all numerical evaluation metrics."""

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
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
        clerical_melted = self._melt_and_clean(self.config.clerical_label_cols, "clerical_label")

        full_matches = pd.merge(
            model_melted, clerical_melted,
            left_on=[self.config.id_col, "model_label"],
            right_on=[self.config.id_col, "clerical_label"],
        )
        full_match_ids = full_matches[self.config.id_col].unique()

        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted["clerical_label"].str[:2]

        partial_matches = pd.merge(
            model_melted, clerical_melted,
            left_on=[self.config.id_col, "model_label_2_digit"],
            right_on=[self.config.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.config.id_col].unique()

        self.df["is_correct"] = self.df[self.config.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.config.id_col].isin(partial_match_ids)

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

    def get_accuracy(self, threshold: float = 0.0, match_type: str = "full", extended: bool = False):
        """Calculates accuracy for predictions above a confidence threshold."""
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

    # ... other metric methods like get_coverage, get_summary_stats etc. would go here ...


# REFACTOR: This new class handles all plotting.
# It takes a DataFrame that has already been processed by the MetricCalculator.
class Visualizer:
    """Handles all visualization tasks."""

    def __init__(self, df: pd.DataFrame, calculator: MetricCalculator):
        self.df = df
        self.calculator = calculator

    def plot_threshold_curves(self, plot_config: Optional[PlotConfig] = None):
        """Plots accuracy and coverage curves."""
        if plot_config is None:
            plot_config = PlotConfig(figsize=(10, 6))
        
        stats_df = self.calculator.get_threshold_stats() # Assumes get_threshold_stats exists on calculator
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

    def plot_confusion_heatmap(self, matrix_config: ConfusionMatrixConfig, plot_config: Optional[PlotConfig] = None):
        """Generates a confusion matrix heatmap."""
        if plot_config is None:
            plot_config = PlotConfig()

        temp_df = self.df[[matrix_config.human_code_col, matrix_config.llm_code_col]].dropna()
        top_human = temp_df[matrix_config.human_code_col].value_counts().nlargest(matrix_config.top_n).index
        top_llm = temp_df[matrix_config.llm_code_col].value_counts().nlargest(matrix_config.top_n).index
        
        filtered_df = temp_df[temp_df[matrix_config.human_code_col].isin(top_human) & temp_df[matrix_config.llm_code_col].isin(top_llm)]
        
        if filtered_df.empty:
            print("No overlapping data for top codes.")
            return None
        
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


# REFACTOR: The main LabelAccuracy class is now a "facade".
# It coordinates the other classes but keeps the public API the same,
# so your existing scripts will not break.
class LabelAccuracy:
    """
    Orchestrates the data cleaning, metric calculation, and visualization
    for coder alignment analysis.
    """
    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """
        Initialises the full analysis pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.
            column_config (ColumnConfig): The configuration for the analysis.
        """
        # Step 1: Clean and prepare the data
        cleaner = DataCleaner(df, column_config)
        clean_df = cleaner.process()

        # Step 2: Initialize the calculator with the clean data
        self.calculator = MetricCalculator(clean_df, column_config)
        
        # The final, processed DataFrame is stored here
        self.df = self.calculator.df

        # Step 3: Initialize the visualizer
        self.visualizer = Visualizer(self.df, self.calculator)

    # REFACTOR: Public methods now delegate their calls to the appropriate helper class.
    # This makes the LabelAccuracy class very simple and easy to read.
    def get_accuracy(self, **kwargs):
        """Calculate accuracy for predictions above a confidence threshold."""
        return self.calculator.get_accuracy(**kwargs)

    def get_jaccard_similarity(self, **kwargs):
        """Calculates the average Jaccard Similarity Index."""
        return self.calculator.get_jaccard_similarity(**kwargs)

    def plot_confusion_heatmap(self, **kwargs):
        """Generates and displays a confusion matrix heatmap."""
        return self.visualizer.plot_confusion_heatmap(**kwargs)

    # ... you would add similar passthrough methods for all other public functions ...
    # e.g., get_coverage, get_summary_stats, plot_threshold_curves etc.

    @staticmethod
    def save_output(metadata: dict, eval_result: dict, save_path: str = "data/") -> str:
        """Save evaluation results to files."""
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

