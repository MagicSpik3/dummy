"""
Runner script to process a local JSON output file and merge it with cleaned
ground truth data.
"""
import argparse

import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.evaluation.coder_alignment import MetricCalculator
from survey_assist_utils.processing.json_processor import JsonProcessor


def main(json_file_path: str, cleaned_data_path: str, output_path: str):
    """
    Main function to orchestrate the data processing pipeline.

    Args:
        json_file_path (str): Path to the raw JSON output file from the model.
        cleaned_data_path (str): Path to the pre-cleaned ground truth CSV file.
        output_path (str): Path to save the final merged and processed CSV file.
    """
    print(f"Starting processing for: {json_file_path}")

    # --- Step 1: Process the local JSON file ---
    json_processor = JsonProcessor()
    llm_df = json_processor.flatten_json_file(json_file_path)
    if llm_df.empty:
        print("Processing failed: Could not create DataFrame from JSON.")
        return
    print(f"Successfully flattened JSON. Shape: {llm_df.shape}")

    # --- Step 2: Load the cleaned ground truth data ---
    try:
        ground_truth_df = pd.read_csv(cleaned_data_path, dtype=str)
        print(f"Successfully loaded cleaned ground truth data. Shape: {ground_truth_df.shape}")
    except FileNotFoundError:
        print(f"Error: Cleaned data file not found at {cleaned_data_path}")
        return

    # --- Step 3: Merge the two DataFrames ---
    merged_df = pd.merge(
        ground_truth_df,
        llm_df,
        on="unique_id",
        how="inner",
    )
    print(f"Successfully merged data. Shape: {merged_df.shape}")

    # --- Step 4: Add the helper/derived columns for analysis ---
    # Configure the MetricCalculator with all possible columns
    config = ColumnConfig(
        model_label_cols=[f"candidate_{i}_sic_code" for i in range(1, 6)],
        model_score_cols=[f"candidate_{i}_likelihood" for i in range(1, 6)],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"], # Adjust as needed
        id_col="unique_id",
    )
    # The MetricCalculator's __init__ method automatically adds the derived columns
    calculator = MetricCalculator(merged_df, config)
    final_df = calculator.df
    print("Successfully added derived columns for analysis (e.g., 'is_correct').")

    # --- Step 5: Save the final output ---
    final_df.to_csv(output_path, index=False)
    print(f"Processing complete. Final data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a model's JSON output and merge it with cleaned data."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="The path to the raw JSON output file from the model.",
    )
    parser.add_argument(
        "--cleaned_data",
        type=str,
        default="data/cleaned_ground_truth.csv",
        help="The path to the pre-cleaned ground truth CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/final_processed_output.csv",
        help="The path to save the final output CSV file.",
    )
    args = parser.parse_args()

    main(args.json_file, args.cleaned_data, args.output)
