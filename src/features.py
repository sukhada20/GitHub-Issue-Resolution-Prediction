"""
src/features.py
KDD Stage 3: Transformation, Feature Engineering, and Preprocessing Pipeline

Pure NumPy/Pandas transformer pipeline ensuring complete reproducibility,
zero data leakage, and full compatibility across restricted Windows Enterprise
code integrity environments.
"""

from typing import Tuple, List, Dict, Any, Optional
import json
import numpy as np
import pandas as pd

from src.preprocessing import (
    CREATION_TIME_NUMERIC_FEATURES,
    CREATION_TIME_BINARY_FEATURES,
    CREATION_TIME_CATEGORICAL_FEATURES,
    LIFECYCLE_T_FEATURES
)

# Creation-only feature set (primary deployment model, strictly prevents post-creation leakage)
CREATION_FEATURES = (
    CREATION_TIME_NUMERIC_FEATURES +
    CREATION_TIME_BINARY_FEATURES +
    CREATION_TIME_CATEGORICAL_FEATURES
)

# Lifecycle-augmented feature set (for comparing value of early 3h observation signals)
FULL_FEATURES = (
    CREATION_FEATURES +
    LIFECYCLE_T_FEATURES
)


class TabularFeatureTransformer:
    """
    Modular transformer for tabular GitHub issue data.
    - Fits numeric medians and standard scalers strictly on training records
    - Imputes binary missing values using mode
    - One-hot encodes repository categorical names with known categories
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ):
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else list(CREATION_TIME_NUMERIC_FEATURES)
        self.binary_cols = list(binary_cols) if binary_cols is not None else list(CREATION_TIME_BINARY_FEATURES)
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else list(CREATION_TIME_CATEGORICAL_FEATURES)

        # Learned parameters
        self.numeric_medians_: Dict[str, float] = {}
        self.numeric_means_: Dict[str, float] = {}
        self.numeric_stds_: Dict[str, float] = {}
        self.binary_modes_: Dict[str, int] = {}
        self.cat_categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> 'TabularFeatureTransformer':
        """Learn parameters from training dataframe only."""
        # 1. Numeric statistics
        for col in self.numeric_cols:
            vals = df[col].dropna()
            med = float(vals.median()) if len(vals) > 0 else 0.0
            mean = float(vals.mean()) if len(vals) > 0 else 0.0
            std = float(vals.std()) if len(vals) > 0 and vals.std() > 0 else 1.0
            self.numeric_medians_[col] = med
            self.numeric_means_[col] = mean
            self.numeric_stds_[col] = std

        # 2. Binary statistics
        for col in self.binary_cols:
            vals = df[col].dropna()
            mode_val = int(vals.mode().iloc[0]) if len(vals) > 0 else 0
            self.binary_modes_[col] = mode_val

        # 3. Categorical categories
        for col in self.categorical_cols:
            cats = sorted(df[col].dropna().unique().tolist())
            self.cat_categories_[col] = cats

        # 4. Assembled output feature names
        feature_names = []
        for col in self.numeric_cols:
            feature_names.append(col)
        for col in self.binary_cols:
            feature_names.append(col)
        for col in self.categorical_cols:
            for cat in self.cat_categories_[col]:
                feature_names.append(f"{col}_{cat}")
        self.feature_names_ = feature_names
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to produce a numeric feature matrix."""
        if not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet.")

        num_rows = len(df)
        total_cols = len(self.feature_names_)
        output_matrix = np.zeros((num_rows, total_cols), dtype=np.float32)

        col_idx = 0

        # Transform numeric columns
        for col in self.numeric_cols:
            if col in df.columns:
                series = df[col].fillna(self.numeric_medians_[col]).astype(float)
            else:
                series = pd.Series(self.numeric_medians_[col], index=df.index)
            scaled = (series.values - self.numeric_means_[col]) / self.numeric_stds_[col]
            output_matrix[:, col_idx] = scaled
            col_idx += 1

        # Transform binary columns
        for col in self.binary_cols:
            if col in df.columns:
                series = df[col].fillna(self.binary_modes_[col]).astype(float)
            else:
                series = pd.Series(self.binary_modes_[col], index=df.index)
            output_matrix[:, col_idx] = series.values
            col_idx += 1

        # Transform categorical columns (One-Hot)
        for col in self.categorical_cols:
            cat_list = self.cat_categories_[col]
            if col in df.columns:
                col_vals = df[col].astype(str)
            else:
                col_vals = pd.Series('unknown', index=df.index)

            for cat in cat_list:
                output_matrix[:, col_idx] = (col_vals == str(cat)).astype(float).values
                col_idx += 1

        return output_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.feature_names_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and learned parameters."""
        return {
            'numeric_cols': self.numeric_cols,
            'binary_cols': self.binary_cols,
            'categorical_cols': self.categorical_cols,
            'numeric_medians': self.numeric_medians_,
            'numeric_means': self.numeric_means_,
            'numeric_stds': self.numeric_stds_,
            'binary_modes': self.binary_modes_,
            'cat_categories': self.cat_categories_,
            'feature_names': self.feature_names_
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TabularFeatureTransformer':
        """Deserialize transformer from dictionary."""
        instance = cls(
            numeric_cols=data['numeric_cols'],
            binary_cols=data['binary_cols'],
            categorical_cols=data['categorical_cols']
        )
        instance.numeric_medians_ = data['numeric_medians']
        instance.numeric_means_ = data['numeric_means']
        instance.numeric_stds_ = data['numeric_stds']
        instance.binary_modes_ = data['binary_modes']
        instance.cat_categories_ = data['cat_categories']
        instance.feature_names_ = data['feature_names']
        instance.is_fitted_ = True
        return instance


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset strictly chronologically by creation timestamp:
    - Train: earliest 70% of issues
    - Validation: middle 15% of issues
    - Test: latest 15% of issues (unseen future)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    df_sorted = df.sort_values('created_at').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"[Features] Chronological Split:")
    print(f"  Train: {len(train_df)} rows ({train_df['created_at'].min().date()} to {train_df['created_at'].max().date()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['created_at'].min().date()} to {val_df['created_at'].max().date()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['created_at'].min().date()} to {test_df['created_at'].max().date()})")

    return train_df, val_df, test_df


def repository_heldout_split(
    df: pd.DataFrame,
    held_out_repos: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-repository generalization split where selected repositories
    are entirely held out from training.
    """
    if held_out_repos is None:
        held_out_repos = ['12_spring-cloud-gateway', '2_lottie-android', '8_thingsboard']

    train_df = df[~df['repo_name'].isin(held_out_repos)].copy()
    test_df = df[df['repo_name'].isin(held_out_repos)].copy()

    print(f"[Features] Repo Held-out Split:")
    print(f"  Held-out Repos: {held_out_repos}")
    print(f"  Train ({train_df['repo_name'].nunique()} repos): {len(train_df)} rows")
    print(f"  Test ({test_df['repo_name'].nunique()} repos):  {len(test_df)} rows")

    return train_df, test_df
