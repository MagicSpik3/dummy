# In coder_alignment.py

def get_jaccard_similarity(self, match_type: str = "full") -> float:
    """Calculates the average Jaccard Similarity Index across all rows."""
    
    # Determine the length to slice the codes based on match_type
    match_len = 2 if match_type == "2-digit" else 5

    def calculate_jaccard_for_row(row):
        # The change is here: slice each code to the desired length
        model_set = {
            str(val)[:match_len] for val in row[self.model_label_cols].dropna() 
            if val not in self._MISSING_VALUE_FORMATS
        }
        clerical_set = {
            str(val)[:match_len] for val in row[self.clerical_label_cols].dropna() 
            if val not in self._MISSING_VALUE_FORMATS
        }

        if not model_set and not clerical_set:
            return 1.0
        
        intersection = len(model_set.intersection(clerical_set))
        union = len(model_set.union(clerical_set))
        return intersection / union if union > 0 else 0.0

    jaccard_scores = self.df.apply(calculate_jaccard_for_row, axis=1)
    return jaccard_scores.mean()

# You would then call it like this:
# jaccard_2_digit_score = analyzer.get_jaccard_similarity(match_type="2-digit")
