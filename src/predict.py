"""
src/predict.py
KDD Stage 6: Inference and Operationalization Engine

Loads the trained feature transformer and native XGBoost models,
computes point predictions, calibrated prediction intervals,
7-day resolution probabilities, and generates actionable advice
for open-source maintainers.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.abspath('.'))

from src.features import TabularFeatureTransformer


class ResolutionPredictor:
    """Production predictor wrapper for GitHub issue resolution time."""

    def __init__(
        self,
        transformer_path: str = 'models/feature_transformer.json',
        reg_model_path: str = 'models/best_regressor.json',
        clf_model_path: str = 'models/best_classifier.json',
        summary_path: str = 'models/evaluation_summary.json'
    ):
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer not found at: {transformer_path}")
        if not os.path.exists(reg_model_path):
            raise FileNotFoundError(f"Regression model not found at: {reg_model_path}")
        if not os.path.exists(clf_model_path):
            raise FileNotFoundError(f"Classifier model not found at: {clf_model_path}")

        # Load transformer
        with open(transformer_path, 'r') as fp:
            trans_dict = json.load(fp)
        self.transformer = TabularFeatureTransformer.from_dict(trans_dict)

        # Load native XGBoost models
        self.reg_bst = xgb.Booster()
        self.reg_bst.load_model(reg_model_path)

        self.clf_bst = xgb.Booster()
        self.clf_bst.load_model(clf_model_path)

        # Load uncertainty calibration intervals if available
        self.residual_p10 = -15.0
        self.residual_p90 = 25.0
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as fp:
                summary = json.load(fp)
                calib = summary.get('uncertainty_calibration', {})
                self.residual_p10 = calib.get('p10_residual', -15.0)
                self.residual_p90 = calib.get('p90_residual', 25.0)

    def predict_single(self, issue_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a dictionary of creation-time features for an issue,
        and returns prediction metrics and maintainer guidance.
        """
        df = pd.DataFrame([issue_features])

        # Feature matrix
        X = self.transformer.transform(df)
        dmat = xgb.DMatrix(X, feature_names=self.transformer.get_feature_names())

        # Point regression prediction (log space -> expm1)
        pred_log = float(self.reg_bst.predict(dmat)[0])
        pred_days = float(np.expm1(pred_log))
        pred_days = max(0.1, pred_days)

        # Calibrated prediction interval: [pred + p10, pred + p90]
        lower_bound = max(0.1, pred_days + self.residual_p10)
        upper_bound = max(lower_bound + 0.5, pred_days + self.residual_p90)

        # Classification probability of resolution <= 7 days
        prob_7d = float(self.clf_bst.predict(dmat)[0])

        # Qualitative triage tier
        if pred_days <= 3.0:
            tier = "Fast Track (< 3 Days)"
            tier_badge = "success"
        elif pred_days <= 7.0:
            tier = "Normal Sprint (3 - 7 Days)"
            tier_badge = "info"
        elif pred_days <= 30.0:
            tier = "Extended Backlog (7 - 30 Days)"
            tier_badge = "warning"
        else:
            tier = "Stale Risk / High Latency (> 30 Days)"
            tier_badge = "danger"

        # Actionable maintainer recommendations
        recommendations = []
        if issue_features.get('issue_has_task_type', 0) == 0:
            recommendations.append("Apply a descriptive type label (bug, feature, docs). Unlabeled issues take significantly longer to triage.")
        if issue_features.get('author_newcomer', 0) == 1:
            recommendations.append("Author is a newcomer with no prior project contributions. Prompt for environment and reproducibility details early.")
        if issue_features.get('issue_num_codes', 0) == 0 and issue_features.get('issue_cleaned_body_len', 0) > 300:
            recommendations.append("Long issue description without code blocks. Request a minimal reproducible code snippet.")
        if issue_features.get('issue_has_pr', 0) == 1:
            recommendations.append("Pull request attached. Expediting CI testing and review will accelerate issue closure.")
        if issue_features.get('repo_open_issue_num', 0) > 500:
            recommendations.append("High backlog load (>500 open issues in repository). Active maintainer triage prioritization recommended.")

        if not recommendations:
            recommendations.append("Issue format, description detail, and contributor profile match historical rapid-closure patterns.")

        return {
            'predicted_days': round(pred_days, 1),
            'prediction_interval_days': (round(lower_bound, 1), round(upper_bound, 1)),
            'prob_resolved_within_7d': round(prob_7d * 100, 1),
            'triage_tier': tier,
            'tier_badge': tier_badge,
            'actionable_recommendations': recommendations
        }
