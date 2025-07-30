"""Script to calculate and display evaluation metrics based on a TOML config file.

This script takes a processed/merged data file and a TOML configuration file
that defines multiple evaluation scenarios. For each scenario, it computes and
displays the relevant performance metrics.

It performs the following steps:
1. Loads the evaluation scenarios from a .toml configuration file.
2. Loads the processed and merged data from a CSV file.
3. For each evaluation case in the config file:
    a. Filters the data based on the 'Unambiguous' flag, if required.
    b. Creates a dynamic ColumnConfig based on the number of coder and model columns.
    c. Instantiates the MetricCalculator with the correctly filtered data and config.
    d. Calculates and prints the relevant metrics (Jaccard or Accuracy).

Example usage:
    python scripts/run_evaluation.py data/final_processed_output.csv configs/evaluation_config.toml

Arguments:
    processed_file: Path to the processed CSV file containing merged data.
    config_file: Path to the .toml file defining the evaluation cases.
"""

import argparse
from collections.abc import Hashable

import pandas as pd
import toml

# This assumes the script is run from a location where survey_assist_utils is importable.
from survey_assist_utils.evaluation.metric_calculator import MetricCalculator
from survey_assist_utils.configs.column_config import ColumnConfig


def main(processed_data_path: str, config_path: str):
    """Main function to orchestrate the metric calculation and display.

    Args:
        processed_data_path (str): Path to the processed CSV file.
        config_path (str): Path to the TOML configuration file.
    """
    # --- Step 1: Load configuration and data ---
    print(f"Loading evaluation config from: {config_path}")
    try:
        eval_config = toml.load(config_path)
        evaluation_cases = eval_config.get("evaluation_cases", [])
        if not evaluation_cases:
            print("Error: No 'evaluation_cases' found in the config file.")
            return
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except toml.TomlDecodeError as e:
        print(f"Error parsing TOML file: {e}")
        return

    print(f"Loading processed data from: {processed_data_path}")
    # Define the master list of all possible columns to slice from
    master_model_labels = [f"candidate_{i}_sic_code" for i in range(1, 6)]
    master_clerical_labels = ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]
    string_type_columns = master_model_labels + master_clerical_labels
    dtype_dict: dict[Hashable, str] = dict.fromkeys(string_type_columns, "string")

    try:
        processed_df = pd.read_csv(processed_data_path, dtype=dtype_dict)
        print(f"Successfully loaded data. Shape: {processed_df.shape}")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    # --- Step 2: Loop through each evaluation case ---
    print("\n" + "=" * 70)
    for i, case in enumerate(evaluation_cases):
        title = case.get("Title", f"Unnamed Case {i+1}")
        print(f"▶️  Running Case {i+1}: {title}")

        # Filter data based on the 'Unambiguous' flag from the TOML file
        # NOTE: This assumes a boolean column 'is_unambiguous' exists in the data,
        # likely created by the FlagGenerator script.
        df_case = processed_df
        if case.get("Unambiguous", False):
            if "is_unambiguous" in processed_df.columns:
                df_case = processed_df[processed_df["is_unambiguous"]].copy()
                print(f"  - Filter applied: Unambiguous records only ({len(df_case)} rows)")
            else:
                print("  - WARNING: 'Unambiguous' flag set, but 'is_unambiguous' column not found. Using all data.")
        else:
             print(f"  - Filter applied: None ({len(df_case)} rows)")


        if df_case.empty:
            print("  - RESULT: No data remains after filtering. Skipping case.")
            print("-" * 70)
            continue

        # Create a dynamic ColumnConfig for this specific case
        num_llm = case.get("LLMs", [5])[0]
        num_cc = case.get("CCs", [3])[0]
        case_config = ColumnConfig(
            model_label_cols=master_model_labels[:num_llm],
            model_score_cols=[f"candidate_{i}_likelihood" for i in range(1, num_llm + 1)],
            clerical_label_cols=master_clerical_labels[:num_cc],
            id_col="unique_id",
        )
        print(f"  - Comparing top {num_cc} clerical code(s) vs. top {num_llm} model code(s).")

        # Instantiate calculator and get results
        calculator = MetricCalculator(df=df_case, column_config=case_config)

        print("  - RESULTS:")
        # Display Jaccard score if specified, otherwise show accuracy
        if "jaccard" in title.lower():
            jaccard_score = calculator.get_jaccard_similarity()
            print(f"    📊 Jaccard Similarity: {jaccard_score:.4f}")
        else:
            full_acc = calculator.get_accuracy(match_type="full", extended=True)
            print(f"    🎯 Full Match Accuracy: {full_acc['accuracy_percent']:.2f}% ({full_acc['matches']}/{full_acc['total_considered']})")

            partial_acc = calculator.get_accuracy(match_type="2-digit", extended=True)
            print(f"    🎯 2-Digit Accuracy:    {partial_acc['accuracy_percent']:.2f}% ({partial_acc['matches']}/{partial_acc['total_considered']})")

        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics based on a TOML config file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "processed_file",
        type=str,
        help="Path to the processed CSV file with merged model and ground truth data.",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the .toml file that defines the evaluation cases.",
    )

    args = parser.parse_args()
    main(args.processed_file, args.config_file)
