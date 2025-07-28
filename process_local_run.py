"""
Runner script to process a local JSON output file and merge it with cleaned
and flagged ground truth data.
"""
import argparse

import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.evaluation.coder_alignment import MetricCalculator
from survey_assist_utils.processing.flag_generator import FlagGenerator
from survey_assist_utils.processing.json_processor import JsonProcessor


def main(json_file_path: str, raw_data_path: str, output_path: str):
    """
    Main function to orchestrate the data processing pipeline.

    Args:
        json_file_path (str): Path to the raw JSON output file from the model.
        raw_data_path (str): Path to the raw ground truth CSV file.
        output_path (str): Path to save the final merged and processed CSV file.
    """
    print(f"Starting processing for: {json_file_path}")

    # --- Step 1: Load the raw ground truth data ---
    try:
        raw_ground_truth_df = pd.read_csv(raw_data_path, dtype=str)
        print(f"Successfully loaded raw ground truth data. Shape: {raw_ground_truth_df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # --- Step 2: Add data quality flags to the ground truth data ---
    flag_generator = FlagGenerator()
    flagged_ground_truth_df = flag_generator.add_flags(raw_ground_truth_df)

    # --- Step 3: Process the local JSON file ---
    json_processor = JsonProcessor()
    llm_df = json_processor.flatten_json_file(json_file_path)
    if llm_df.empty:
        print("Processing failed: Could not create DataFrame from JSON.")
        return
    print(f"Successfully flattened JSON. Shape: {llm_df.shape}")

    # --- Step 4: Merge the two DataFrames ---
    merged_df = pd.merge(
        flagged_ground_truth_df,
        llm_df,
        on="unique_id",
        how="inner",
    )
    print(f"Successfully merged data. Shape: {merged_df.shape}")

    # --- Step 5: Add the final helper/derived columns for analysis ---
    config = ColumnConfig(
        model_label_cols=[f"candidate_{i}_sic_code" for i in range(1, 6)],
        model_score_cols=[f"candidate_{i}_likelihood" for i in range(1, 6)],
        clerical_label_cols=["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"],
        id_col="unique_id",
    )
    calculator = MetricCalculator(merged_df, config)
    final_df = calculator.df
    print("Successfully added derived columns for analysis (e.g., 'is_correct').")

    # --- Step 6: Save the final output ---
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
        "--raw_data",
        type=str,
        default="data/raw_ground_truth.csv",
        help="The path to the raw ground truth CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/final_processed_output.csv",
        help="The path to save the final output CSV file.",
    )
    args = parser.parse_args()

    main(args.json_file, args.raw_data, args.output)
