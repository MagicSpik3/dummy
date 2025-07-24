"""
Module for orchestrating and managing SIC classification experiments.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import toml

# Assume these functions exist to call your model
# from survey_assist_utils.api import call_survey_assist_api
# from survey_assist_utils.local import call_local_model


@dataclass
class Prompt:
    """A data structure for a single prompt to be tested."""
    name: str
    template: str


class ExperimentOrchestrator:
    """
    Manages the end-to-end process of running a classification experiment.

    This class reads a configuration, loads data, iterates through different
    prompts, calls the appropriate classification service (API or local),
    and structures the results into a DataFrame ready for analysis and
    human evaluation.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialises the orchestrator with a configuration dictionary.

        Args:
            config (dict[str, Any]): The loaded experiment configuration.
        """
        self.config = config
        self.metadata = config.get("metadata", {})
        self.paths = config.get("paths", {})
        self.settings = config.get("settings", {})
        self.prompts = [Prompt(**p) for p in config.get("prompts", [])]
        self.input_df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Loads the initial dataset from the path specified in the config."""
        input_path = self.paths.get("input_data_csv")
        if not input_path:
            raise ValueError("Input data path is not specified in the config.")
        print(f"Loading data from: {input_path}")
        # Assuming the input has these columns, adjust as necessary
        return pd.read_csv(input_path, dtype=str)

    def _run_single_case(self, row: pd.Series, prompt: Prompt) -> dict:
        """
        Processes a single row of data with a specific prompt.

        Args:
            row (pd.Series): A single row from the input DataFrame.
            prompt (Prompt): The prompt configuration to use.

        Returns:
            dict: A structured dictionary containing the results for this case.
        """
        # 1. Construct the payload from the input row
        payload = {
            "job_title": row.get("job_title"),
            "job_description": row.get("job_description"),
            "industry_descr": row.get("industry_descr"),
            "prompt": prompt.template,  # Pass the specific prompt template
        }

        # 2. Call the appropriate service based on the execution mode
        if self.settings.get("execution_mode") == "api":
            # response = call_survey_assist_api(payload) # Placeholder
            response = {"sic_candidates": [{"sic_code": "12345"}], "follow_up_question": "Is this wholesale?"} # Mock response
        else:
            # response = call_local_model(payload) # Placeholder
            response = {"sic_candidates": [{"sic_code": "54321"}], "follow_up_question": "Is this retail?"} # Mock response

        # 3. Structure the final result dictionary
        result = {
            "unique_id": row.get("unique_id"),
            "experiment_label": self.metadata.get("experiment_label"),
            "prompt_name": prompt.name,
            "initial_llm_payload": payload,
            "initial_human_codes": [row.get("sic_ind_occ1"), row.get("sic_ind_occ2")],
            "ai_response": response,
            "suggested_followup_question": response.get("follow_up_question"),
        }
        return result

    def run_experiment(self) -> pd.DataFrame:
        """
        Executes the full experiment across all prompts and data.

        Returns:
            pd.DataFrame: A DataFrame containing the structured results of the
                          entire experiment run.
        """
        results = []
        total_runs = len(self.input_df) * len(self.prompts)
        current_run = 0

        print(f"Starting experiment: {self.metadata.get('experiment_label')}")
        print(f"Total runs to execute: {total_runs}")

        for prompt in self.prompts:
            print(f"--- Processing with prompt: '{prompt.name}' ---")
            for _, row in self.input_df.iterrows():
                current_run += 1
                result = self._run_single_case(row, prompt)
                results.append(result)
                print(f"  > Completed run {current_run}/{total_runs}")

        results_df = pd.DataFrame(results)

        # Add placeholder columns for the human evaluation phase
        results_df["human_followup_answer"] = None
        results_df["human_final_sic"] = None
        results_df["human_question_score"] = None
        results_df["human_outcome_score"] = None

        print("Experiment finished.")
        return results_df

    def save_results(self, results_df: pd.DataFrame):
        """
        Saves the results DataFrame and a copy of the config.

        Args:
            results_df (pd.DataFrame): The DataFrame to save.
        """
        output_dir = self.paths.get("output_directory")
        if not output_dir:
            raise ValueError("Output directory is not specified in the config.")
        
        # Create a unique filename for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.metadata.get('experiment_label', 'experiment')}_{timestamp}"
        
        # Save results to Parquet
        parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
        results_df.to_parquet(parquet_path)
        print(f"Results saved to: {parquet_path}")

        # Save the config file used for this run for traceability
        config_path = os.path.join(output_dir, f"{base_filename}_config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)
        print(f"Configuration saved to: {config_path}")

