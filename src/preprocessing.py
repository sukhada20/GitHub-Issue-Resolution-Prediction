"""
src/preprocessing.py
KDD Stage 2: Preprocessing and Data Validation

Responsible for loading raw GitHub issue datasets, parsing timestamps,
computing the continuous resolution target and 7-day binary target,
cleaning binary indicator flags, handling missing columns, and deriving temporal features.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Binary columns in original dataset that are coded as 1/2 (1 = No/False, 2 = Yes/True)
BINARY_12_COLS = [
    'repo_has_org',
    'issue_has_task_type',
    'issue_has_pr',
    'issue_has_conflict',
    'issue_on_rest_day',
    'issue_has_at_tag',
    'author_org',
    'author_core_member',
    'author_newcomer',
    'author_first_issue',
    'participant_first_core_member',
    'issue_has_core_comment',
    'issue_has_other_comment'
]

# Feature sets strictly available at creation vs at observation window T=3h
CREATION_TIME_NUMERIC_FEATURES = [
    'repo_age',
    'repo_contributors',
    'repo_subscribers',
    'repo_team_size',
    'repo_open_issue_num',
    'repo_open_delta',
    'repo_closed_issue_time',
    'repo_num_commits_inproject',
    'repo_num_issues_created_inproject',
    'repo_num_issues_created_inproject_closed',
    'issue_cleaned_body_len',
    'issue_num_codes',
    'issue_num_urls',
    'issue_num_pics',
    'issue_flesch_reading_ease',
    'issue_desc_positivity',
    'issue_desc_negativity',
    'issue_desc_pos_polarity',
    'issue_desc_subjectivity',
    'issue_text_similarity',
    'author_age',
    'author_followers',
    'author_comments_negative',
    'author_comments_positive',
    'author_num_commits_inproject',
    'author_num_issues_inproject',
    'author_num_issues_closed_inproject',
    # engineered temporal
    'created_hour',
    'created_dayofweek',
    'created_month',
    'created_year'
]

CREATION_TIME_BINARY_FEATURES = [
    'repo_has_org',
    'issue_has_task_type',
    'issue_has_pr',
    'issue_has_conflict',
    'issue_on_rest_day',
    'issue_has_at_tag',
    'author_org',
    'author_core_member',
    'author_newcomer',
    'author_first_issue',
    'is_weekend'
]

CREATION_TIME_CATEGORICAL_FEATURES = [
    'repo_name'
]

LIFECYCLE_T_FEATURES = [
    'issue_num_labels_T',
    'issue_hotness_T',
    'issue_per_event_days_T',
    'issue_num_assignments_T',
    'issue_num_comments_T',
    'issue_mean_comment_size_T',
    'issue_num_referenced_by_T',
    'issue_num_subscribed_by_T',
    'issue_num_actors_T',
    'issue_has_core_comment',
    'issue_has_other_comment',
    'author_num_comments_T',
    'author_issue_closed_days_T',
    'author_num_issue_closed_T',
    'author_num_commits_inproject_T',
    'participant_first_core_member',
    'participant_first_occur',
    'participants_num_T',
    'participants_association_score_T',
    'participant_num_commits_by_actors_T',
    'participant_num_commits_by_unique_actors_T',
    'participant_num_comments_T',
    'repo_num_commits_project_T',
    'repo_num_issues_created_projected_T',
    'repo_num_issues_created_project_closed_T'
]


def load_raw_dataset(raw_csv_path: str = 'data/raw/IssueResolutionTimePrediction/dataset/issue_data_3h_normal.csv') -> pd.DataFrame:
    """Load raw CSV dataset and log metadata."""
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")
    print(f"[Preprocessing] Loading raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    print(f"[Preprocessing] Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Drop unpopulated vector columns (title_vec, body_vec)
    - Parse timestamps into UTC datetimes
    - Calculate continuous resolution_days target
    - Filter invalid negative/zero resolution records
    - Calculate binary 7-day target (1 if <= 7 days, else 0)
    - Normalize 1/2 binary encoded columns to 0/1
    - Engineer temporal features from created_at
    """
    df = df.copy()

    # Drop unpopulated text vectors if present
    for col in ['title_vec', 'body_vec']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Parse timestamps
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['closed_at'] = pd.to_datetime(df['closed_at'], utc=True)

    # Compute continuous resolution time in days: (closed_at - created_at) / 86400
    df['resolution_days'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 86400.0

    # Validation: Filter any impossible non-positive durations
    valid_mask = df['resolution_days'] > 0
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"[Preprocessing] Removed {invalid_count} records with non-positive resolution time.")
        df = df[valid_mask].copy()

    # Secondary classification target: Resolved within 7 days (calendar days)
    df['resolved_within_7d'] = (df['resolution_days'] <= 7.0).astype(int)

    # Log target: log1p of resolution days for robust regression
    df['log_resolution_days'] = np.log1p(df['resolution_days'])

    # Recode 1/2 binary columns to standard 0/1 (1 -> 0, 2 -> 1)
    for col in BINARY_12_COLS:
        if col in df.columns:
            # If values are {1, 2}, convert 1 -> 0, 2 -> 1
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({1, 2}):
                df[col] = (df[col] == 2).astype(int)

    # Engineer temporal features from created_at
    df['created_hour'] = df['created_at'].dt.hour
    df['created_dayofweek'] = df['created_at'].dt.dayofweek
    df['created_month'] = df['created_at'].dt.month
    df['created_year'] = df['created_at'].dt.year
    df['is_weekend'] = df['created_dayofweek'].isin([5, 6]).astype(int)

    # Sort strictly chronologically by created_at for time-aware splitting
    df = df.sort_values('created_at').reset_index(drop=True)

    print(f"[Preprocessing] Data cleaning complete. Final records: {len(df)}")
    print(f"  Target resolution_days: min={df['resolution_days'].min():.2f}, "
          f"median={df['resolution_days'].median():.2f}, mean={df['resolution_days'].mean():.2f}, "
          f"max={df['resolution_days'].max():.2f}")
    print(f"  7-day resolution rate: {df['resolved_within_7d'].mean():.2%}")

    return df


def save_processed_data(df: pd.DataFrame, output_path: str = 'data/processed/issues_clean.csv') -> None:
    """Save cleaned and validated dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Preprocessing] Cleaned dataset saved to: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == '__main__':
    raw_df = load_raw_dataset()
    clean_df = clean_and_validate_data(raw_df)
    save_processed_data(clean_df)
