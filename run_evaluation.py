"""
This script runs the coder alignment evaluation on a pre-prepared dataset.

It takes a path to a cleaned and flagged CSV file and an evaluation
configuration file, then runs a series of tests as defined in the config.
"""
import argparse
from typing import TypedDict, List

import pandas as pd
import toml

from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    LabelAccuracy,
)


class EvaluationCase(TypedDict):
    """Defines the structure for a single evaluation case in the config."""
    Title: str
    CCs: list[int]
    LLMs: list[int]
    Unambiguous: bool


def run_all_evaluations(df: pd.DataFrame, evaluation_cases: List[EvaluationCase]):
    """
    Executes a full evaluation pipeline on a DataFrame using a series of
    predefined evaluation cases.

    Args:
        df (pd.DataFrame): The input DataFrame containing pre-prepared data.
        evaluation_cases (List[EvaluationCase]): A list of evaluation scenarios.
    """
    # Set up standard column names that the evaluation expects
    model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
    model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
    clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

    results_data = []

    for case in evaluation_cases:
        print(f"\n--- Running Case: {case['Title']} ---")

        config = ColumnConfig(
            model_label_cols=model_label_cols[: case["LLMs"][0]],
            model_score_cols=model_score_cols[: case["LLMs"][0]],
            clerical_label_cols=clerical_label_cols[: case["CCs"][0]],
            id_col="unique_id",
            filter_unambiguous=case["Unambiguous"],
        )

        analyzer = LabelAccuracy(df=df, column_config=config)

        # Calculate metrics
        acc_5_digit = analyzer.get_accuracy(match_type="full")
        acc_2_digit = analyzer.get_accuracy(match_type="2-digit")
        jaccard_5_digit = analyzer.get_jaccard_similarity()
        # jaccard_2_digit = analyzer.get_jaccard_similarity(match_type="2-digit") # Assuming you add this

        results_data.append({
            "Title": case['Title'],
            "5-Digit Accuracy (%)": acc_5_digit,
            "2-Digit Accuracy (%)": acc_2_digit,
            "5-Digit Jaccard": jaccard_5_digit,
            # "2-Digit Jaccard": jaccard_2_digit,
        })

    results_df = pd.DataFrame(results_data)
    print("\n--- Evaluation Summary ---")
    print(results_df.round(2).to_string(index=False))


def main(data_path: str, config_path: str):
    """
    Main function to load data and config, then run the evaluation.

    Args:
        data_path (str): Path to the pre-prepared input CSV file.
        config_path (str): Path to the TOML file defining evaluation cases.
    """
    print(f"Loading data from: {data_path}")
    try:
        input_df = pd.read_csv(data_path, dtype=str)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {data_path}")
        return

    print(f"Loading evaluation cases from: {config_path}")
    try:
        eval_config = toml.load(config_path)
        evaluation_cases = eval_config.get("evaluation_cases", [])
    except FileNotFoundError:
        print(f"Error: Evaluation config file not found at {config_path}")
        return

    run_all_evaluations(input_df, evaluation_cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coder alignment evaluation on a pre-prepared dataset."
    )
    parser.add_argument(
        "data_file",
        type=str,
        help="The path to the pre-prepared input CSV file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation_config.toml",
        help="The path to the TOML file defining the evaluation cases.",
    )
    args = parser.parse_args()

    main(args.data_file, args.config)
