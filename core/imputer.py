import numpy as np
import pandas as pd


class DataImputer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def audit_missing_values(self) -> pd.DataFrame:
        """Calculates missing value counts and percentages per column."""
        total_missing = self.df.isnull().sum()
        percent_missing = (total_missing / len(self.df)) * 100

        audit_df = pd.DataFrame(
            {"missing_count": total_missing, "missing_percent": percent_missing}
        )

        return audit_df[audit_df["missing_count"] > 0].sort_values(
            by="missing_count", ascending=False
        )

    def auto_clean_and_impute(
        self, missing_threshold: float = 0.5
    ) -> tuple[pd.DataFrame, dict]:
        """Automated sequential data cleaning pipeline.

        - Drops columns exceeding missing threshold (default > 50% missing).
        - Imputes numerical missing values with median.
        - Imputes categorical/text missing values with mode or 'Unknown'.
        - Imputes datetime missing values with forward fill.
        """
        initial_missing = int(self.df.isnull().sum().sum())
        actions_taken = []

        for col in self.df.columns:
            missing_ratio = self.df[col].isnull().mean()
            if missing_ratio > missing_threshold:
                self.df.drop(columns=[col], inplace=True)
                actions_taken.append(
                    f"Dropped column '{col}' due to high missing ratio ({missing_ratio:.1%})"
                )

        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                actions_taken.append(
                    f"Imputed numeric column '{col}' with median: {median_val}"
                )

        datetime_cols = self.df.select_dtypes(
            include=["datetime", "datetime64"]
        ).columns
        for col in datetime_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].ffill().bfill()
                actions_taken.append(
                    f"Imputed datetime column '{col}' using forward/backward fill"
                )

        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                self.df[col].fillna(fill_val, inplace=True)
                actions_taken.append(
                    f"Imputed categorical column '{col}' with: '{fill_val}'"
                )

        final_missing = int(self.df.isnull().sum().sum())

        summary = {
            "initial_missing_count": initial_missing,
            "final_missing_count": final_missing,
            "actions_log": actions_taken,
        }

        return self.df, summary