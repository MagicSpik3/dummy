"""
Module for high-level assessment of pre-processed evaluation data.
"""
import pandas as pd

class Assessor:
    """
    Provides high-level summary statistics on an evaluation DataFrame that has
    already had quality flags added.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Assessor with a pre-processed DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame, which must contain the
                               necessary data quality flag columns (e.g.,
                               'num_answers', 'Unambiguous', 'Match_5_digits').
        """
        self.df = df

    def get_group_membership_summary(self) -> pd.DataFrame:
        """
        Calculates and returns a summary of group memberships in the data.

        This method quantifies the distribution of records based on their
        codability characteristics.

        Returns:
            pd.DataFrame: A DataFrame summarizing the counts and percentages
                          of each codability group.
        """
        n = len(self.df)
        if n == 0:
            return pd.DataFrame(columns=["Category", "Count", "Percentage"])

        # Calculate counts for each category from the pre-existing flag columns
        count_uncodeable = (self.df["num_answers"] == 0).sum()
        count_2_digits_only = self.df["Match_2_digits"].sum()
        count_5_digits_unambiguous = self.df["Unambiguous"].sum()
        count_5_digits_ambiguous = (self.df["Match_5_digits"] & ~self.df["Unambiguous"]).sum()
        count_3_digits_only = self.df["Match_3_digits"].sum()
        count_four_plus = (self.df["num_answers"] == 4).sum()

        group_counts = [
            count_uncodeable, count_four_plus, count_2_digits_only,
            count_3_digits_only, count_5_digits_unambiguous, count_5_digits_ambiguous
        ]

        group_labels = [
            "Uncodeable", "4+ Codes", "Codeable to 2 digits only",
            "Codeable to 3 digits only", "Codeable unambiguously to 5 digits",
            "Codeable ambiguously to 5 digits"
        ]

        total_categorised = sum(group_counts)
        count_other = n - total_categorised

        if count_other > 0:
            group_counts.append(count_other)
            group_labels.append("Other")

        percentages = [(count / n) * 100 for count in group_counts]
        summary_df = pd.DataFrame({
            "Category": group_labels,
            "Count": group_counts,
            "Percentage": percentages
        })
        return summary_df
