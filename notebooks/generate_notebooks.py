"""
notebooks/generate_notebooks.py
Generates the 5 project Jupyter notebooks following the KDD process.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_all_notebooks():
    os.makedirs('notebooks', exist_ok=True)

    # -------------------------------------------------------------
    # 01_data_understanding.ipynb
    # -------------------------------------------------------------
    nb1 = make_notebook([
        md_cell("""# 01. Data Understanding & Acquisition (KDD Stage 1)
### Predicting GitHub Issue Resolution Time Using the KDD Process

This notebook addresses **Phase 1: Selection & Data Understanding**.
- Dataset source: Research dataset from *Predicting Issue Resolution Time of OSS Using Multiple Features* (Wiley)
- Scale: 34,665 unique resolved issues across 18 major GitHub projects
- Problem definition:
  - **Primary Task**: Regression to estimate issue resolution time in calendar days ($closed\_at - created\_at$)
  - **Secondary Task**: Binary classification predicting whether an issue resolves within 7 days ($\le 7$ days)
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

raw_path = '../data/raw/IssueResolutionTimePrediction/dataset/issue_data_3h_normal.csv'
df = pd.read_csv(raw_path)
print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(3)"""),
        md_cell("""### 1.1 Dataset Schema & Available Fields
Let's inspect data types, null values, and repository distribution."""),
        code_cell("""# Inspect null values
null_counts = df.isnull().sum()
print("Columns with missing values:")
print(null_counts[null_counts > 0])

# Inspect repository counts
print("\\nIssues per repository:")
print(df['repo_name'].value_counts())"""),
        md_cell("""### 1.2 Data Dictionary Overview
The project data dictionary documents 76 columns categorized into:
1. Repository Context (`repo_age`, `repo_contributors`, `repo_team_size`, `repo_open_issue_num`)
2. Issue Characteristics (`issue_cleaned_body_len`, `issue_num_codes`, `issue_num_urls`, readability, sentiment)
3. Contributor Attributes (`author_age`, `author_core_member`, `author_newcomer`)
4. Observation-Window Dynamics ($T=3h$) (`issue_num_comments_T`, `issue_num_actors_T`)
"""),
        code_cell("""dict_path = '../data/data_dictionary.csv'
if os.path.exists(dict_path):
    dict_df = pd.read_csv(dict_path)
    print(f"Loaded Data Dictionary with {len(dict_df)} fields:")
    display(dict_df.head(10))""")
    ])

    with open('notebooks/01_data_understanding.ipynb', 'w') as f:
        json.dump(nb1, f, indent=2)

    # -------------------------------------------------------------
    # 02_preprocessing.ipynb
    # -------------------------------------------------------------
    nb2 = make_notebook([
        md_cell("""# 02. Preprocessing & Data Cleaning (KDD Stage 2)
### Cleaning, Target Definition, and Timezone Validation

This notebook implements:
1. Parsing timestamps to UTC datetimes.
2. Computing exact resolution duration:
   $$\\text{resolution\\_days} = \\frac{\\text{closed\\_at} - \\text{created\\_at}}{86,400\\text{ seconds}}$$
3. Filtering invalid negative or zero durations.
4. Defining binary 7-day target ($resolution\\_days \\le 7.0$).
5. Normalizing binary flags from 1/2 to 0/1.
6. Temporal feature extraction (`created_hour`, `created_dayofweek`, `created_month`, `created_year`, `is_weekend`).
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.preprocessing import clean_and_validate_data, load_raw_dataset

raw_df = load_raw_dataset('../data/raw/IssueResolutionTimePrediction/dataset/issue_data_3h_normal.csv')
clean_df = clean_and_validate_data(raw_df)
clean_df[['repo_name', 'created_at', 'closed_at', 'resolution_days', 'resolved_within_7d']].head()"""),
        md_cell("""### 2.1 Target Variable Inspection
Check duration percentiles, spread, and 7-day resolution rate."""),
        code_cell("""print("Resolution Days Percentiles:")
print(clean_df['resolution_days'].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.99]))

print(f"\\nBinary Target (Resolved within 7 days): {clean_df['resolved_within_7d'].mean():.2%}")""")
    ])

    with open('notebooks/02_preprocessing.ipynb', 'w') as f:
        json.dump(nb2, f, indent=2)

    # -------------------------------------------------------------
    # 03_eda.ipynb
    # -------------------------------------------------------------
    nb3 = make_notebook([
        md_cell("""# 03. Exploratory Data Analysis (KDD Stage 3)
### Understanding Resolution Time Distributions & Influential Patterns

Exploratory analysis investigating:
1. Target skewness and log transformation.
2. Repository-level variation in resolution velocity.
3. Impact of author newcomer status and core team membership.
4. Pull request attachment vs resolution duration.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))

df = pd.read_csv('../data/processed/issues_clean.csv')
df['created_at'] = pd.to_datetime(df['created_at'])

# Resolution time distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['resolution_days'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Raw Resolution Days (Heavily Skewed)')
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Count')

axes[1].hist(df['log_resolution_days'], bins=50, color='salmon', edgecolor='black')
axes[1].set_title('Log-Transformed Resolution Days: log(1 + days)')
axes[1].set_xlabel('log(1 + days)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()"""),
        md_cell("""### 3.2 Median Resolution Days by Repository
Analyze differences in cadence across open-source communities."""),
        code_cell("""repo_stats = df.groupby('repo_name')['resolution_days'].agg(['count', 'median', 'mean']).sort_values('median')
print("Resolution days by project:")
display(repo_stats)""")
    ])

    with open('notebooks/03_eda.ipynb', 'w') as f:
        json.dump(nb3, f, indent=2)

    # -------------------------------------------------------------
    # 04_feature_engineering.ipynb
    # -------------------------------------------------------------
    nb4 = make_notebook([
        md_cell("""# 04. Feature Engineering & Leakage Prevention (KDD Stage 3)
### Creation-Time Signals vs Observation-Window T Features

**Core Distinction & Leakage Prevention**:
- **Creation-Time Features**: Strictly available at the moment of issue creation.
  - Zero post-creation comments, zero future events, zero lifecycle leakage.
- **Lifecycle Features ($T=3h$)**: Measured at 3 hours post-creation.
  - Useful for triage ablation study to assess value of early maintainer engagement.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.features import TabularFeatureTransformer, chronological_split

df = pd.read_csv('../data/processed/issues_clean.csv')
train_df, val_df, test_df = chronological_split(df, 0.70, 0.15, 0.15)

transformer = TabularFeatureTransformer()
transformer.fit(train_df)
X_train = transformer.transform(train_df)
print(f"Transformed Training Matrix Shape: {X_train.shape}")
print(f"Total features: {len(transformer.get_feature_names())}")""")
    ])

    with open('notebooks/04_feature_engineering.ipynb', 'w') as f:
        json.dump(nb4, f, indent=2)

    # -------------------------------------------------------------
    # 05_modeling_evaluation.ipynb
    # -------------------------------------------------------------
    nb5 = make_notebook([
        md_cell("""# 05. Modeling, Evaluation & Explainability (KDD Stage 4 & 5)
### Model Comparison, Lifecycle Ablation, Generalization, and Feature Importances

Evaluates:
1. **Regression Task**: Baseline Median, Ridge, Decision Tree, Random Forest, XGBoost.
2. **Classification Task**: Majority Baseline, Logistic Regression, Decision Tree, Random Forest, XGBoost Classifier.
3. **Lifecycle Ablation**: Evaluating incremental boost from 3-hour triage signals.
4. **Held-out Repositories**: Cross-project generalization error.
5. **Feature Importances**: Which signals most strongly drive resolution velocity.
"""),
        code_cell("""import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

with open('../models/evaluation_summary.json', 'r') as f:
    summary = json.load(f)

reg_df = pd.DataFrame(summary['regression_comparison'])
print("=== REGRESSION MODEL COMPARISON ===")
display(reg_df[['Model', 'Test_MAE', 'Test_RMSE', 'Test_MedAE', 'Test_R2', 'Train_Time_s']])

clf_df = pd.DataFrame(summary['classification_comparison'])
print("\\n=== 7-DAY CLASSIFICATION MODEL COMPARISON ===")
display(clf_df[['Model', 'Test_Accuracy', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Test_ROC_AUC', 'Train_Time_s']])"""),
        md_cell("""### 5.2 Top Feature Importances (Gain)"""),
        code_cell("""top_feats = pd.DataFrame(summary['top_features'])
print("Top 15 Most Influential Features:")
display(top_feats.head(15))""")
    ])

    with open('notebooks/05_modeling_evaluation.ipynb', 'w') as f:
        json.dump(nb5, f, indent=2)

    print("Generated all 5 notebooks successfully in 'notebooks/'.")


if __name__ == '__main__':
    generate_all_notebooks()
