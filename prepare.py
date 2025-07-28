"""
This script serves as a command-line utility to load raw evaluation data,
add a series of data quality flags using the FlagGenerator class, and save
the enriched DataFrame to a new CSV file.

It is controlled by a .toml configuration file.
"""

import logging
import os

import pandas as pd
import toml

# REFACTOR: Import the new, centralized FlagGenerator class.
from survey_assist_utils.processing.flag_generator import FlagGenerator


def load_config(config_path: str) -> dict:
    """Loads configuration settings from a .toml file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return toml.load(file)


# REFACTOR: All the duplicated helper functions (_calculate_num_answers,
# _create_sic_match_flags, add_data_quality_flags, etc.) have been removed
# from this file. The logic is now contained within the FlagGenerator class.


def main():
    """Main function to run the data preparation pipeline."""
    # --- 1. Load Configuration and Set Up Logging ---
    config = load_config("config.toml")
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- 2. Get File Paths from Config ---
    try:
        analysis_filepath = config["paths"]["batch_filepath"]
        analysis_csv = config["paths"]["analysis_csv"]
    except KeyError as e:
        logging.error("Missing required path in config.toml: %s", e)
        return

    # --- 3. Load Raw Data ---
    try:
        sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)
        logging.info("Successfully loaded raw data from: %s", analysis_filepath)
    except FileNotFoundError:
        logging.error("Raw data file not found at: %s", analysis_filepath)
        return

    # --- 4. Use FlagGenerator to Add Quality Flags ---
    # REFACTOR: Instantiate the new FlagGenerator class and call its process method.
    # This replaces the old, local add_data_quality_flags function.
    flag_generator = FlagGenerator()
    sic_dataframe_with_flags = flag_generator.add_flags(sic_dataframe)

    # --- 5. Save the Enriched DataFrame ---
    output_dir = os.path.dirname(analysis_csv)
    os.makedirs(output_dir, exist_ok=True)
    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)
    logging.info("Successfully saved data with flags to: %s", analysis_csv)

    print("\n--- Analysis Complete ---")
    print(f"Processed {len(sic_dataframe)} rows.")
    print(f"Output saved to {analysis_csv}")


if __name__ == "__main__":
    main()
