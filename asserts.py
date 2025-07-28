# In survey_assist_utils/data_cleaning/data_cleaner.py

def _validate_inputs(self):
    """Validates that all required inputs and configurations are present and consistent."""
    # ... (existing column presence checks remain here) ...

    # --- NEW: Add data type validation ---
    # Check that score columns are numeric or can be coerced
    for col in self.config.model_score_cols:
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            # Attempt a dry-run of coercion to see if it's possible
            if not pd.to_numeric(self.df[col], errors='coerce').notna().all():
                # This warning flags columns with non-numeric values that will be lost
                print(f"Warning: Column '{col}' is not numeric and contains values that cannot be converted.")

    # Check that label columns are strings or objects that can be treated as strings
    for col in self.config.model_label_cols + self.config.clerical_label_cols:
        if not pd.api.types.is_string_dtype(self.df[col]) and not pd.api.types.is_object_dtype(self.df[col]):
            print(f"Warning: Label column '{col}' is not of type string or object.")

      
