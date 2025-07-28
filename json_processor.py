"""
This module defines the JsonProcessor class, which is responsible for
processing raw, local JSON output files from Survey Assist.
"""
import json
from typing import Any

import pandas as pd


class JsonProcessor:
    """Handles the processing of a raw LLM JSON file into a clean DataFrame."""

    def flatten_json_file(self, file_path: str, max_candidates: int = 5) -> pd.DataFrame:
        """
        Reads LLM response JSON data from a local file and flattens it.

        Args:
            file_path (str): The local path to the JSON file.
            max_candidates (int): The maximum number of SIC candidates to flatten.

        Returns:
            pd.DataFrame: A DataFrame with the flattened JSON data.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return pd.DataFrame()

        records_to_process = json_data if isinstance(json_data, list) else [json_data]
        all_flat_records = []

        for record in records_to_process:
            flat_record = {
                "unique_id": record.get("unique_id"),
                "classified": record.get("classified"),
                "chosen_sic_code": record.get("sic_code"),
            }
            # Flatten candidates
            candidates = record.get("sic_candidates", [])
            for i in range(max_candidates):
                if i < len(candidates) and isinstance(candidates[i], dict):
                    candidate = candidates[i]
                    flat_record[f"candidate_{i+1}_sic_code"] = candidate.get("sic_code")
                    flat_record[f"candidate_{i+1}_likelihood"] = candidate.get("likelihood")
                else:
                    flat_record[f"candidate_{i+1}_sic_code"] = None
                    flat_record[f"candidate_{i+1}_likelihood"] = None
            all_flat_records.append(flat_record)

        return pd.DataFrame(all_flat_records)
