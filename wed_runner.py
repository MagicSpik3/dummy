"""Script to calculate and display evaluation metrics from a processed data file.

This script takes a file that has been processed and merged (containing both model
predictions and ground truth labels) and uses the MetricCalculator module to
compute and display various performance metrics.

It performs the following steps:
1. Loads the processed and merged data from a CSV file.
2. Defines the column configuration needed for the analysis.
3. Instantiates the MetricCalculator with the data and configuration.
4. Calls various metric calculation methods (accuracy, coverage, Jaccard, etc.).
5. Prints the results to the console in a formatted summary.

Example usage:
    python scripts/calculate_metrics.py data/final_processed_output.csv

Arguments:
    processed_file: Path to the processed CSV file containing merged data.
"""

import argparse
from collections.abc import Hashable

import pandas as pd

# This assumes the script is run from a location where survey_assist_utils is importable.
from survey_assist_utils.evaluation.metric_calculator import MetricCalculator
from survey_assist_utils.configs.column_config import ColumnConfig


def main(processed_data_path: str):
    """Main function to orchestrate the metric calculation and display.

    Args:
        processed_data_path (str): Path to the processed CSV file.
    """
    # --- Step 1: Load the processed data ---
    print(f"Loading processed data from: {processed_data_path}")

    # --- Step 2: Define column configuration ---
    # This configuration must match the columns in the input CSV file.
    config = ColumnConfig(
        model_label_cols=[f"candidate_{i}_sic_code" for i in range(1, 6)],
        model_score_cols=[f"candidate_{i}_likelihood" for i in range(1, 6)],
        clerical_label_cols=["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"],
        id_col="unique_id",
    )

    # Ensure code columns (which can have leading zeros) are read as strings.
    string_type_columns = config.model_label_cols + config.clerical_label_cols
    dtype_dict: dict[Hashable, str] = dict.fromkeys(string_type_columns, "string")

    try:
        processed_df = pd.read_csv(processed_data_path, dtype=dtype_dict)
        print(f"Successfully loaded data. Shape: {processed_df.shape}\n")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    # --- Step 3: Instantiate the MetricCalculator ---
    calculator = MetricCalculator(df=processed_df, column_config=config)
    print("MetricCalculator initialised. Starting calculations...")
    print("=" * 60)

    # --- Step 4 & 5: Calculate and print all metrics ---

    # General Summary
    print("\n--- 📊 Overall Summary ---")
    summary = calculator.get_summary_stats()
    jaccard = calculator.get_jaccard_similarity()
    print(f"Total Records Analyzed:     {summary['total_samples']}")
    print(f"Jaccard Similarity Index:   {jaccard:.4f}")
    print(f"Overall Accuracy (Full):    {summary['overall_accuracy']:.2f}%")
    print("-" * 30)

    # Detailed Accuracy
    print("\n--- 🎯 Detailed Accuracy Breakdown (at 0.75 threshold) ---")
    full_match_details = calculator.get_accuracy(
        threshold=0.75, match_type="full", extended=True
    )
    partial_match_details = calculator.get_accuracy(
        threshold=0.75, match_type="2-digit", extended=True
    )

    print("Full Match Details:")
    print(f"  - Accuracy:           {full_match_details['accuracy_percent']}%")
    print(f"  - Matched Records:    {full_match_details['matches']}")
    print(f"  - Total Considered:   {full_match_details['total_considered']}")

    print("\n2-Digit Match Details:")
    print(f"  - Accuracy:           {partial_match_details['accuracy_percent']}%")
    print(f"  - Matched Records:    {partial_match_details['matches']}")
    print(f"  - Total Considered:   {partial_match_details['total_considered']}")
    print("-" * 30)

    # Threshold Table
    print("\n--- 📈 Accuracy & Coverage vs. Confidence Threshold ---")
    custom_thresholds = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    threshold_stats_df = calculator.get_threshold_stats(thresholds=custom_thresholds)
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(threshold_stats_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Metric calculation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and display evaluation metrics from a processed data file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "processed_file",
        type=str,
        help="Path to the processed CSV file containing merged model and ground truth data.",
    )

    args = parser.parse_args()
    main(args.processed_file)
