"""
This script serves as a command-line utility to load raw evaluation data,
add a series of data quality flags using the FlagGenerator class, and save
the enriched DataFrame to a new CSV file.

It is controlled by a .toml configuration file.
"""

import os

import pandas as pd
import toml

# REFACTOR: Import the custom logger instead of the standard one.
from survey_assist_utils.logging import get_logger
from survey_assist_utils.processing.flag_generator import FlagGenerator

# REFACTOR: Initialize the custom logger for the module.
logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Loads configuration settings from a .toml file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return toml.load(file)


def main():
    """Main function to run the data preparation pipeline."""
    # REFACTOR: The manual logging setup has been removed. The custom logger
    # handles its own configuration automatically.
    
    # --- 1. Load Configuration ---
    try:
        config = load_config("config.toml")
    except FileNotFoundError:
        logger.error("Configuration file 'config.toml' not found.")
        return

    # --- 2. Get File Paths from Config ---
    try:
        analysis_filepath = config["paths"]["batch_filepath"]
        analysis_csv = config["paths"]["analysis_csv"]
    except KeyError as e:
        logger.error("Missing required path in config.toml", missing_key=str(e))
        return

    # --- 3. Load Raw Data ---
    try:
        sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)
        logger.info("Successfully loaded raw data.", path=analysis_filepath)
    except FileNotFoundError:
        logger.error("Raw data file not found.", path=analysis_filepath)
        return

    # --- 4. Use FlagGenerator to Add Quality Flags ---
    flag_generator = FlagGenerator()
    sic_dataframe_with_flags = flag_generator.add_flags(sic_dataframe)

    # --- 5. Save the Enriched DataFrame ---
    output_dir = os.path.dirname(analysis_csv)
    os.makedirs(output_dir, exist_ok=True)
    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)
    logger.info("Successfully saved data with flags.", path=analysis_csv)

    print("\n--- Analysis Complete ---")
    print(f"Processed {len(sic_dataframe)} rows.")
    print(f"Output saved to {analysis_csv}")


if __name__ == "__main__":
    main()
