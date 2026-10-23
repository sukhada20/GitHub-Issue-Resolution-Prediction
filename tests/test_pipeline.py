"""
tests/test_pipeline.py
Automated validation tests for KDD pipeline:
- Typical and unusual input combinations
- Missing / invalid inputs
- Categories unseen during training
- Repositories outside training scope
- Repeated predictions for determinism
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from src.predict import ResolutionPredictor


def test_predictor_initialization():
    predictor = ResolutionPredictor()
    assert predictor is not None
    assert predictor.reg_bst is not None
    assert predictor.clf_bst is not None
    assert predictor.transformer is not None


def test_typical_input():
    predictor = ResolutionPredictor()
    sample = {
        'repo_name': '17_youtube-dl',
        'repo_age': 1000.0,
        'repo_contributors': 200,
        'repo_subscribers': 1000,
        'repo_team_size': 10,
        'repo_open_issue_num': 300,
        'repo_open_delta': 1.0,
        'repo_closed_issue_time': 7.0,
        'repo_num_commits_inproject': 4000,
        'repo_num_issues_created_inproject': 6000,
        'repo_num_issues_created_inproject_closed': 5500,
        'issue_cleaned_body_len': 200.0,
        'issue_num_codes': 1,
        'issue_num_urls': 1,
        'issue_num_pics': 0,
        'issue_flesch_reading_ease': 60.0,
        'issue_desc_positivity': 0.1,
        'issue_desc_negativity': 0.05,
        'issue_desc_pos_polarity': 0.1,
        'issue_desc_subjectivity': 0.3,
        'issue_text_similarity': 0.5,
        'author_age': 300.0,
        'author_followers': 10,
        'author_comments_negative': 0.0,
        'author_comments_positive': 0.1,
        'author_num_commits_inproject': 0,
        'author_num_issues_inproject': 1,
        'author_num_issues_closed_inproject': 1,
        'created_hour': 10,
        'created_dayofweek': 1,
        'created_month': 4,
        'created_year': 2021,
        'repo_has_org': 1,
        'issue_has_task_type': 1,
        'issue_has_pr': 0,
        'issue_has_conflict': 0,
        'issue_on_rest_day': 0,
        'issue_has_at_tag': 0,
        'author_org': 0,
        'author_core_member': 0,
        'author_newcomer': 1,
        'author_first_issue': 1,
        'is_weekend': 0
    }
    res = predictor.predict_single(sample)
    assert 'predicted_days' in res
    assert res['predicted_days'] > 0
    assert 'prediction_interval_days' in res
    assert res['prediction_interval_days'][0] <= res['prediction_interval_days'][1]
    assert 0.0 <= res['prob_resolved_within_7d'] <= 100.0
    assert len(res['actionable_recommendations']) > 0


def test_unseen_repository():
    """Test model behavior on an unseen repository outside the training projects."""
    predictor = ResolutionPredictor()
    sample = {
        'repo_name': 'unseen_custom_kubernetes_repo',
        'repo_open_issue_num': 150,
        'repo_closed_issue_time': 5.0,
        'issue_cleaned_body_len': 120.0,
        'issue_num_codes': 0,
        'author_core_member': 0,
        'author_newcomer': 1
    }
    res = predictor.predict_single(sample)
    assert res['predicted_days'] > 0
    assert res['prob_resolved_within_7d'] >= 0


def test_empty_or_sparse_input():
    """Test robustness against minimal dictionary with default imputations."""
    predictor = ResolutionPredictor()
    res = predictor.predict_single({})
    assert res['predicted_days'] > 0
    assert 'triage_tier' in res


def test_reproducibility():
    """Confirm repeated predictions with the same input return identical output."""
    predictor = ResolutionPredictor()
    sample = {'repo_name': '15_Python', 'author_core_member': 1, 'issue_has_pr': 1}
    res1 = predictor.predict_single(sample)
    res2 = predictor.predict_single(sample)
    assert res1['predicted_days'] == res2['predicted_days']
    assert res1['prob_resolved_within_7d'] == res2['prob_resolved_within_7d']


if __name__ == '__main__':
    test_predictor_initialization()
    test_typical_input()
    test_unseen_repository()
    test_empty_or_sparse_input()
    test_reproducibility()
    print("All unit and validation tests passed successfully!")
