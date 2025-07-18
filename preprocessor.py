"""Module contains functionality to preprocess json output from Survey Assist.
Class:
    JsonPreprocessor
        Handles the processing of raw LLM JSON files into a clean DataFrame.

Methods:
    get_gcs_filepaths
        Gets a list of GCS filepaths to process based on config. Can operate
        in single-file or directory mode.
    record_count
        Counts records in a JSON file.
    flatten_llm_json_to_dataframe
        Flattens a JSON file into a pandas DataFrame.
    count_all_records
        Counts total records across all specified files.
    process_files
        Processes all specified JSON files into a single DataFrame.
    merge_eval_data
        Merges the processed data with external evaluation data.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Union

import pandas as pd
from google.api_core import exceptions
from google.cloud import storage

logger = logging.getLogger(__name__)


class JsonPreprocessor:
    """Handles the processing of raw LLM JSON files into a clean DataFrame."""

    def __init__(self, config: dict[str, Any]):
        """Initializes the preprocessor with a configuration dictionary and a GCS client.

        Args:
            config (dict[str, Any]): A dictionary containing configuration
                                     settings, typically loaded from a TOML
                                     file by the calling script.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")
        self.config = config
        self.storage_client = storage.Client()

    def get_gcs_filepaths(self) -> list[str]:
        """
        Gets a list of GCS filepaths to process based on config.

        Operates in one of two modes based on the 'single_file' parameter:
        1. Single File Mode: If 'single_file' is "True", it processes the
           single file specified in 'paths.named_file'.
        2. Directory Mode: Otherwise, it finds all files in 'paths.gcs_json_dir'
           created on or after 'parameters.date_since'.
        """
        # Check if single_file mode is enabled in the config
        is_single_file_mode = (
            self.config.get("parameters", {}).get("single_file", "False").lower()
            == "true"
        )

        if is_single_file_mode:
            named_file = self.config.get("paths", {}).get("named_file")
            if not named_file:
                logging.error(
                    "Single file mode is enabled, but 'named_file' path is missing from config."
                )
                return []
            logging.info("Single file mode enabled. Processing: %s", named_file)
            return [named_file]

        # --- Fallback to original directory processing mode ---
        bucket_name = self.config["paths"]["gcs_bucket_name"]
        prefix = self.config["paths"]["gcs_json_dir"]
        date_str = self.config["parameters"]["date_since"]
        given_date = datetime.strptime(date_str, "%Y%m%d")

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        later_files = []
        logging.info(
            "Directory mode: Searching for files in gs://%s/%s on or after %s",
            bucket_name,
            prefix,
            date_str,
        )
        try:
            blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix)
            for blob in blobs:
                # Skip "folders" and files in sub-directories
                if blob.name.endswith("/") or os.path.dirname(
                    blob.name
                ) != prefix.strip("/"):
                    continue

                filename = os.path.basename(blob.name)
                if filename.endswith("_output.json"):
                    try:
                        file_date = datetime.strptime(filename[:8], "%Y%m%d")
                        if file_date >= given_date:
                            later_files.append(f"gs://{bucket_name}/{blob.name}")
                    except ValueError:
                        # Ignore files that don't start with a valid date
                        continue
            logging.info("Found %d files to process.", len(later_files))
            return later_files
        except exceptions.NotFound:
            logging.error(
                "Source bucket or prefix not found: gs://%s/%s", bucket_name, prefix
            )
            return []

    def _get_json_data(self, file_path: str) -> Union[dict, list, None]:
        """Reads JSON data from either a local file or a GCS path.

        Args:
            file_path (str): The full local path or gs:// URI of the file.

        Returns:
            Union[dict, list, None]: The parsed JSON data, or None on error.
        """
        try:
            if file_path.startswith("gs://"):
                bucket_name, blob_name = file_path.replace("gs://", "").split("/", 1)
                blob = self.storage_client.bucket(bucket_name).blob(blob_name)
                content = blob.download_as_string()
                return json.loads(content)

            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, exceptions.NotFound):
            logging.error("File not found at %s", file_path)
            return None
        except json.JSONDecodeError as e:
            logging.error("Error decoding JSON from %s: %s", file_path, e)
            return None

    def record_count(self, file_path: str) -> int:
        """Reads an LLM response JSON file and counts the number of records.

        Args:
            file_path (str): The path to the JSON file (local or GCS).

        Returns:
            int: The number of records in the file.
        """
        json_data = self._get_json_data(file_path)
        if json_data is None:
            return 0

        if isinstance(json_data, list):
            return len(json_data)
        if isinstance(json_data, dict):
            return 1

        logging.warning("JSON content in %s is not a list or a dictionary.", file_path)
        return 0

    def flatten_llm_json_to_dataframe(
        self, file_path: str, max_candidates: int = 5
    ) -> pd.DataFrame:
        """Reads LLM response JSON data from a file (local or GCS), flattens it.

        Args:
            file_path (str): The path to the JSON file (local or GCS).
            max_candidates (int): The maximum number of SIC candidates to flatten.

        Returns:
            pd.DataFrame: A DataFrame with the flattened JSON data.
        """
        json_data = self._get_json_data(file_path)
        if json_data is None:
            return pd.DataFrame()

        records_to_process = []
        if isinstance(json_data, dict):
            records_to_process = [json_data]
        elif isinstance(json_data, list):
            records_to_process = json_data
        else:
            logging.error(
                "JSON content in %s is not a list or a dictionary.", file_path
            )
            return pd.DataFrame()

        all_flat_records = []
        logging.info(
            "Found %d records to process in %s", len(records_to_process), file_path
        )
        for record in records_to_process:
            flat_record = {}
            flat_record["unique_id"] = record.get("unique_id")
            flat_record["classified"] = record.get("classified")
            flat_record["followup"] = record.get("followup")
            flat_record["chosen_sic_code"] = record.get("sic_code")
            flat_record["chosen_sic_description"] = record.get("sic_description")
            flat_record["reasoning"] = record.get("reasoning")

            payload = record.get("request_payload", {})
            flat_record["payload_llm"] = payload.get("llm")
            flat_record["payload_type"] = payload.get("type")
            flat_record["payload_job_title"] = payload.get("job_title")
            flat_record["payload_job_description"] = payload.get("job_description")
            flat_record["payload_industry_descr"] = payload.get("industry_descr")

            candidates = record.get("sic_candidates", [])
            for i in range(max_candidates):
                if i < len(candidates) and isinstance(candidates[i], dict):
                    candidate = candidates[i]
                    flat_record[f"candidate_{i+1}_sic_code"] = candidate.get("sic_code")
                    flat_record[f"candidate_{i+1}_sic_descriptive"] = candidate.get(
                        "sic_descriptive"
                    )
                    flat_record[f"candidate_{i+1}_likelihood"] = candidate.get(
                        "likelihood"
                    )
                else:
                    flat_record[f"candidate_{i+1}_sic_code"] = None
                    flat_record[f"candidate_{i+1}_sic_descriptive"] = None
                    flat_record[f"candidate_{i+1}_likelihood"] = None
            all_flat_records.append(flat_record)

        return pd.DataFrame(all_flat_records)

    def count_all_records(self) -> int:
        """Counts the total number of records across all specified JSON files."""
        filepaths = self.get_gcs_filepaths()
        if not filepaths:
            logging.warning("No files found to count.")
            return 0
        return sum(self.record_count(path) for path in filepaths)

    def process_files(self) -> pd.DataFrame:
        """Main method to load, flatten, and combine all specified JSON files."""
        filepaths = self.get_gcs_filepaths()
        if not filepaths:
            logging.warning("No files found to process. Returning empty DataFrame.")
            return pd.DataFrame()

        all_dfs = [self.flatten_llm_json_to_dataframe(path) for path in filepaths]
        all_dfs = [df for df in all_dfs if not df.empty]

        if not all_dfs:
            logging.warning("All files were empty or failed to process.")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(
            "Combined all files into a DataFrame with shape %s", combined_df.shape
        )

        combined_df = combined_df.sort_values(
            by="candidate_1_sic_code", na_position="last"
        )
        unique_id_col = self.config["json_keys"]["unique_id"]
        combined_df.drop_duplicates(subset=[unique_id_col], inplace=True, keep="first")
        logging.info("Shape after dropping duplicates: %s", combined_df.shape)

        return combined_df

    def merge_eval_data(self, flattened_json_df: pd.DataFrame) -> pd.DataFrame:
        """Merges evaluation data from a CSV file with a flattened JSON DataFrame.
        Note: pandas can read from GCS if 'gcsfs' is installed.

        Args:
            flattened_json_df (pd.DataFrame): DataFrame with flattened JSON data.

        Returns:
            pd.DataFrame: A merged DataFrame.
        """
        processed_csv_output = self.config["paths"]["processed_csv_output"]
        try:
            eval_data = pd.read_csv(processed_csv_output, dtype={"SIC_Division": str})
            eval_data = eval_data.drop_duplicates(subset="unique_id")

            merged_df = pd.merge(
                eval_data,
                flattened_json_df,
                on="unique_id",
                how="inner",
                suffixes=("_eval", "_llm"),
            )
            logging.info("Merged DataFrame shape: %s", merged_df.shape)
            return merged_df
        except FileNotFoundError:
            logging.error("Evaluation data file not found: %s", processed_csv_output)
            return pd.DataFrame()
