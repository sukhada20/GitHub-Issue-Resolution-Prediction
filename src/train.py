"""
src/train.py
KDD Stage 4 & 5: Data Mining, Model Training, Evaluation, and Comparison

Implements:
1. Baseline Median Regressor & Majority Classifier
2. Ridge Regression (L2 regularized linear benchmark on log target)
3. Decision Tree (Single tree benchmark)
4. Random Forest (Parallel bagged forest)
5. Gradient Boosted Trees (XGBoost Regressor & Classifier)
Evaluates chronological hold-out test sets, cross-repository generalization,
feature importances, and residual calibration intervals.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Tuple, Optional

from src.preprocessing import (
    load_raw_dataset,
    clean_and_validate_data,
    save_processed_data,
    CREATION_TIME_NUMERIC_FEATURES,
    CREATION_TIME_BINARY_FEATURES,
    CREATION_TIME_CATEGORICAL_FEATURES,
    LIFECYCLE_T_FEATURES
)
from src.features import (
    CREATION_FEATURES,
    FULL_FEATURES,
    TabularFeatureTransformer,
    chronological_split,
    repository_heldout_split
)


# =====================================================================
# EVALUATION METRIC FUNCTIONS (Pure NumPy)
# =====================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, R2, and Median Absolute Error in calendar days."""
    y_pred = np.maximum(0.0, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))
    medae = float(np.median(np.abs(y_true - y_pred)))
    return {
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 4),
        'MedAE': round(medae, 3)
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict[str, Any]:
    """Calculate Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix."""
    y_pred = (y_pred >= 0.5).astype(int)
    y_true = y_true.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float((tp + tn) / max(1, len(y_true)))
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float(2 * prec * rec / max(1e-10, prec + rec))

    # ROC AUC via Wilcoxon-Mann-Whitney rank sum
    if y_probs is not None and len(np.unique(y_true)) > 1:
        idx = np.argsort(y_probs)
        y_true_sorted = y_true[idx]
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        ranks = np.arange(1, len(y_true) + 1)
        sum_ranks_pos = np.sum(ranks[y_true_sorted == 1])
        roc_auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))
    else:
        roc_auc = 0.5

    return {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'ROC_AUC': round(roc_auc, 4),
        'Confusion_Matrix': [[tn, fp], [fn, tp]]
    }


# =====================================================================
# MODEL IMPLEMENTATIONS
# =====================================================================

class BaselineMedianRegressor:
    """Predicts training set median resolution days."""
    def __init__(self):
        self.median_val_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.median_val_ = float(np.median(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X),), fill_value=self.median_val_)


class BaselineMajorityClassifier:
    """Predicts training set majority class and probability."""
    def __init__(self):
        self.majority_class_ = 0
        self.majority_prob_ = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        p1 = float(np.mean(y))
        self.majority_class_ = 1 if p1 >= 0.5 else 0
        self.majority_prob_ = p1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X),), fill_value=self.majority_class_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X),), fill_value=self.majority_prob_)


class RidgeRegressorLog:
    """Ridge regression with L2 regularization fitted on log1p(y)."""
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Add bias column
        X_b = np.hstack([np.ones((len(X), 1), dtype=np.float32), X])
        y_log = np.log1p(y)
        reg_eye = self.alpha * np.eye(X_b.shape[1], dtype=np.float32)
        reg_eye[0, 0] = 0.0  # Do not regularize intercept
        self.beta_ = np.linalg.solve(X_b.T @ X_b + reg_eye, X_b.T @ y_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.hstack([np.ones((len(X), 1), dtype=np.float32), X])
        log_preds = X_b @ self.beta_
        return np.maximum(0.0, np.expm1(log_preds))


class LogisticRegressionClassifier:
    """Logistic regression classifier trained via L2 regularized gradient descent."""
    def __init__(self, lr: float = 0.05, max_iter: int = 500, l2_reg: float = 0.01):
        self.lr = lr
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_b = np.hstack([np.ones((len(X), 1), dtype=np.float32), X])
        w = np.zeros(X_b.shape[1], dtype=np.float32)
        for _ in range(self.max_iter):
            # Sigmoid
            z = np.clip(X_b @ w, -15.0, 15.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (X_b.T @ (p - y)) / len(y) + self.l2_reg * w
            grad[0] -= self.l2_reg * w[0]
            w -= self.lr * grad
        self.weights_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_b = np.hstack([np.ones((len(X), 1), dtype=np.float32), X])
        z = np.clip(X_b @ self.weights_, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# =====================================================================
# MAIN PIPELINE RUNNER
# =====================================================================

def run_experiment_pipeline():
    print("="*70)
    print("KDD PROCESS: GITHUB ISSUE RESOLUTION TIME PREDICTION")
    print("="*70)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    clean_path = 'data/processed/issues_clean.csv'
    if not os.path.exists(clean_path):
        raw_df = load_raw_dataset()
        df = clean_and_validate_data(raw_df)
        save_processed_data(df, clean_path)
    else:
        df = pd.read_csv(clean_path)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])

    print(f"Loaded dataset: {len(df):,} issues across {df['repo_name'].nunique()} repositories.")

    # 1. Chronological Split (70% train, 15% validation, 15% test)
    train_df, val_df, test_df = chronological_split(df, 0.70, 0.15, 0.15)

    # 2. Fit Feature Transformer on Train split only
    transformer = TabularFeatureTransformer()
    transformer.fit(train_df)

    X_train = transformer.transform(train_df)
    X_val = transformer.transform(val_df)
    X_test = transformer.transform(test_df)
    feature_names = transformer.get_feature_names()

    print(f"\nTransformed feature matrix: {X_train.shape[1]} features, {X_train.shape[0]} training samples.")

    # Targets
    y_train_reg = train_df['resolution_days'].values
    y_val_reg = val_df['resolution_days'].values
    y_test_reg = test_df['resolution_days'].values

    y_train_log = np.log1p(y_train_reg)
    y_val_log = np.log1p(y_val_reg)
    y_test_log = np.log1p(y_test_reg)

    y_train_clf = train_df['resolved_within_7d'].values
    y_val_clf = val_df['resolved_within_7d'].values
    y_test_clf = test_df['resolved_within_7d'].values

    # DMatrix objects for XGBoost
    dtrain_reg = xgb.DMatrix(X_train, label=y_train_log, feature_names=feature_names)
    dval_reg = xgb.DMatrix(X_val, label=y_val_log, feature_names=feature_names)
    dtest_reg = xgb.DMatrix(X_test, label=y_test_log, feature_names=feature_names)

    dtrain_clf = xgb.DMatrix(X_train, label=y_train_clf, feature_names=feature_names)
    dval_clf = xgb.DMatrix(X_val, label=y_val_clf, feature_names=feature_names)
    dtest_clf = xgb.DMatrix(X_test, label=y_test_clf, feature_names=feature_names)

    # -----------------------------------------------------------------
    # REGRESSION BENCHMARKS
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 4.1: REGRESSION BENCHMARKING (Creation-Time Features)")
    print("="*60)

    reg_results = []

    # 1. Baseline Median
    t0 = time.time()
    med_model = BaselineMedianRegressor().fit(X_train, y_train_reg)
    val_pred = med_model.predict(X_val)
    test_pred = med_model.predict(X_test)
    val_m = compute_regression_metrics(y_val_reg, val_pred)
    test_m = compute_regression_metrics(y_test_reg, test_pred)
    reg_results.append({'Model': 'Baseline Median', **{f'Val_{k}': v for k, v in val_m.items()}, **{f'Test_{k}': v for k, v in test_m.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"1. Baseline Median     -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")

    # 2. Ridge Regression (Log-target)
    t0 = time.time()
    ridge = RidgeRegressorLog(alpha=10.0).fit(X_train, y_train_reg)
    val_pred = ridge.predict(X_val)
    test_pred = ridge.predict(X_test)
    val_m = compute_regression_metrics(y_val_reg, val_pred)
    test_m = compute_regression_metrics(y_test_reg, test_pred)
    reg_results.append({'Model': 'Ridge Regression', **{f'Val_{k}': v for k, v in val_m.items()}, **{f'Test_{k}': v for k, v in test_m.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"2. Ridge Regression    -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")

    # 3. Decision Tree (Single Tree via XGBoost engine)
    t0 = time.time()
    dt_params = {'max_depth': 6, 'eta': 1.0, 'objective': 'reg:squarederror', 'tree_method': 'hist'}
    dt_bst = xgb.train(dt_params, dtrain_reg, num_boost_round=1)
    val_pred = np.expm1(dt_bst.predict(dval_reg))
    test_pred = np.expm1(dt_bst.predict(dtest_reg))
    val_m = compute_regression_metrics(y_val_reg, val_pred)
    test_m = compute_regression_metrics(y_test_reg, test_pred)
    reg_results.append({'Model': 'Decision Tree', **{f'Val_{k}': v for k, v in val_m.items()}, **{f'Test_{k}': v for k, v in test_m.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"3. Decision Tree       -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")

    # 4. Random Forest (Bagged ensemble)
    t0 = time.time()
    rf_params = {
        'max_depth': 10,
        'eta': 1.0,
        'num_parallel_tree': 50,
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }
    rf_bst = xgb.train(rf_params, dtrain_reg, num_boost_round=1)
    val_pred = np.expm1(rf_bst.predict(dval_reg))
    test_pred = np.expm1(rf_bst.predict(dtest_reg))
    val_m = compute_regression_metrics(y_val_reg, val_pred)
    test_m = compute_regression_metrics(y_test_reg, test_pred)
    reg_results.append({'Model': 'Random Forest', **{f'Val_{k}': v for k, v in val_m.items()}, **{f'Test_{k}': v for k, v in test_m.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"4. Random Forest       -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")

    # 5. Gradient Boosted Trees (XGBoost Regressor)
    t0 = time.time()
    xgb_reg_params = {
        'max_depth': 6,
        'eta': 0.08,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }
    best_xgb_reg = xgb.train(xgb_reg_params, dtrain_reg, num_boost_round=150)
    val_pred = np.expm1(best_xgb_reg.predict(dval_reg))
    test_pred = np.expm1(best_xgb_reg.predict(dtest_reg))
    val_m = compute_regression_metrics(y_val_reg, val_pred)
    test_m = compute_regression_metrics(y_test_reg, test_pred)
    reg_results.append({'Model': 'XGBoost', **{f'Val_{k}': v for k, v in val_m.items()}, **{f'Test_{k}': v for k, v in test_m.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"5. XGBoost (Boosted)   -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")

    # Uncertainty Calibration (Empirical residuals on Validation split)
    val_residuals = y_val_reg - val_pred
    p10_res = float(np.percentile(val_residuals, 10))
    p90_res = float(np.percentile(val_residuals, 90))
    print(f"\n[Uncertainty Calibration] 80% Residual Interval: [{p10_res:.2f}d, {p90_res:.2f}d]")

    # -----------------------------------------------------------------
    # CLASSIFICATION BENCHMARKS (7-day resolution target)
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 4.2: 7-DAY CLASSIFICATION BENCHMARKING")
    print("="*60)

    clf_results = []

    # 1. Majority Baseline
    t0 = time.time()
    maj_model = BaselineMajorityClassifier().fit(X_train, y_train_clf)
    val_p = maj_model.predict(X_val)
    val_prob = maj_model.predict_proba(X_val)
    test_p = maj_model.predict(X_test)
    test_prob = maj_model.predict_proba(X_test)
    val_cm = compute_classification_metrics(y_val_clf, val_p, val_prob)
    test_cm = compute_classification_metrics(y_test_clf, test_p, test_prob)
    clf_results.append({'Model': 'Majority Baseline', **{f'Val_{k}': v for k, v in val_cm.items() if k != 'Confusion_Matrix'}, **{f'Test_{k}': v for k, v in test_cm.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"1. Majority Baseline   -> Test Acc: {test_cm['Accuracy']} | F1: {test_cm['F1']} | ROC-AUC: {test_cm['ROC_AUC']}")

    # 2. Logistic Regression
    t0 = time.time()
    logreg = LogisticRegressionClassifier(lr=0.08, max_iter=400, l2_reg=0.01).fit(X_train, y_train_clf)
    val_prob = logreg.predict_proba(X_val)
    test_prob = logreg.predict_proba(X_test)
    val_cm = compute_classification_metrics(y_val_clf, (val_prob >= 0.5).astype(int), val_prob)
    test_cm = compute_classification_metrics(y_test_clf, (test_prob >= 0.5).astype(int), test_prob)
    clf_results.append({'Model': 'Logistic Regression', **{f'Val_{k}': v for k, v in val_cm.items() if k != 'Confusion_Matrix'}, **{f'Test_{k}': v for k, v in test_cm.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"2. Logistic Regression -> Test Acc: {test_cm['Accuracy']} | F1: {test_cm['F1']} | Precision: {test_cm['Precision']} | Recall: {test_cm['Recall']} | ROC-AUC: {test_cm['ROC_AUC']}")

    # 3. Decision Tree Classifier
    t0 = time.time()
    dt_clf_params = {'max_depth': 6, 'eta': 1.0, 'objective': 'binary:logistic', 'tree_method': 'hist'}
    dt_clf_bst = xgb.train(dt_clf_params, dtrain_clf, num_boost_round=1)
    val_prob = dt_clf_bst.predict(dval_clf)
    test_prob = dt_clf_bst.predict(dtest_clf)
    val_cm = compute_classification_metrics(y_val_clf, (val_prob >= 0.5).astype(int), val_prob)
    test_cm = compute_classification_metrics(y_test_clf, (test_prob >= 0.5).astype(int), test_prob)
    clf_results.append({'Model': 'Decision Tree', **{f'Val_{k}': v for k, v in val_cm.items() if k != 'Confusion_Matrix'}, **{f'Test_{k}': v for k, v in test_cm.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"3. Decision Tree Clf   -> Test Acc: {test_cm['Accuracy']} | F1: {test_cm['F1']} | Precision: {test_cm['Precision']} | Recall: {test_cm['Recall']} | ROC-AUC: {test_cm['ROC_AUC']}")

    # 4. Random Forest Classifier
    t0 = time.time()
    rf_clf_params = {
        'max_depth': 10,
        'eta': 1.0,
        'num_parallel_tree': 50,
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'objective': 'binary:logistic',
        'tree_method': 'hist'
    }
    rf_clf_bst = xgb.train(rf_clf_params, dtrain_clf, num_boost_round=1)
    val_prob = rf_clf_bst.predict(dval_clf)
    test_prob = rf_clf_bst.predict(dtest_clf)
    val_cm = compute_classification_metrics(y_val_clf, (val_prob >= 0.5).astype(int), val_prob)
    test_cm = compute_classification_metrics(y_test_clf, (test_prob >= 0.5).astype(int), test_prob)
    clf_results.append({'Model': 'Random Forest', **{f'Val_{k}': v for k, v in val_cm.items() if k != 'Confusion_Matrix'}, **{f'Test_{k}': v for k, v in test_cm.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"4. Random Forest Clf   -> Test Acc: {test_cm['Accuracy']} | F1: {test_cm['F1']} | Precision: {test_cm['Precision']} | Recall: {test_cm['Recall']} | ROC-AUC: {test_cm['ROC_AUC']}")

    # 5. XGBoost Classifier
    t0 = time.time()
    xgb_clf_params = {
        'max_depth': 6,
        'eta': 0.08,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }
    best_xgb_clf = xgb.train(xgb_clf_params, dtrain_clf, num_boost_round=150)
    val_prob = best_xgb_clf.predict(dval_clf)
    test_prob = best_xgb_clf.predict(dtest_clf)
    val_cm = compute_classification_metrics(y_val_clf, (val_prob >= 0.5).astype(int), val_prob)
    test_cm = compute_classification_metrics(y_test_clf, (test_prob >= 0.5).astype(int), test_prob)
    clf_results.append({'Model': 'XGBoost Classifier', **{f'Val_{k}': v for k, v in val_cm.items() if k != 'Confusion_Matrix'}, **{f'Test_{k}': v for k, v in test_cm.items()}, 'Train_Time_s': round(time.time() - t0, 3)})
    print(f"5. XGBoost Classifier  -> Test Acc: {test_cm['Accuracy']} | F1: {test_cm['F1']} | Precision: {test_cm['Precision']} | Recall: {test_cm['Recall']} | ROC-AUC: {test_cm['ROC_AUC']}")

    # -----------------------------------------------------------------
    # PHASE 4.3: LIFECYCLE ABLATION (Incremental Value of T=3h Signals)
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 4.3: LIFECYCLE ABLATION (Creation vs Early T=3h Signals)")
    print("="*60)
    transformer_full = TabularFeatureTransformer(
        numeric_cols=CREATION_TIME_NUMERIC_FEATURES + [c for c in LIFECYCLE_T_FEATURES if c not in ['participant_first_core_member', 'issue_has_core_comment', 'issue_has_other_comment']],
        binary_cols=CREATION_TIME_BINARY_FEATURES + ['participant_first_core_member', 'issue_has_core_comment', 'issue_has_other_comment'],
        categorical_cols=CREATION_TIME_CATEGORICAL_FEATURES
    )
    transformer_full.fit(train_df)
    X_train_f = transformer_full.transform(train_df)
    X_test_f = transformer_full.transform(test_df)
    dtrain_f = xgb.DMatrix(X_train_f, label=y_train_log, feature_names=transformer_full.get_feature_names())
    dtest_f = xgb.DMatrix(X_test_f, label=y_test_log, feature_names=transformer_full.get_feature_names())

    bst_f = xgb.train(xgb_reg_params, dtrain_f, num_boost_round=150)
    t3h_preds = np.expm1(bst_f.predict(dtest_f))
    t3h_metrics = compute_regression_metrics(y_test_reg, t3h_preds)
    print(f"Creation-Time Only XGBoost -> Test MAE: {test_m['MAE']}d | RMSE: {test_m['RMSE']}d | MedAE: {test_m['MedAE']}d | R2: {test_m['R2']}")
    print(f"T=3h Lifecycle-Augmented   -> Test MAE: {t3h_metrics['MAE']}d | RMSE: {t3h_metrics['RMSE']}d | MedAE: {t3h_metrics['MedAE']}d | R2: {t3h_metrics['R2']}")

    # -----------------------------------------------------------------
    # PHASE 5.4: REPOSITORY HELD-OUT GENERALIZATION
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 5.4: REPOSITORY HELD-OUT GENERALIZATION")
    print("="*60)
    held_repos = ['12_spring-cloud-gateway', '2_lottie-android', '8_thingsboard']
    repo_train_df, repo_test_df = repository_heldout_split(df, held_repos)
    repo_trans = TabularFeatureTransformer().fit(repo_train_df)
    X_repo_tr = repo_trans.transform(repo_train_df)
    X_repo_te = repo_trans.transform(repo_test_df)
    drepo_tr = xgb.DMatrix(X_repo_tr, label=np.log1p(repo_train_df['resolution_days'].values), feature_names=repo_trans.get_feature_names())
    drepo_te = xgb.DMatrix(X_repo_te, label=np.log1p(repo_test_df['resolution_days'].values), feature_names=repo_trans.get_feature_names())

    repo_bst = xgb.train(xgb_reg_params, drepo_tr, num_boost_round=100)
    repo_preds = np.expm1(repo_bst.predict(drepo_te))
    repo_metrics = compute_regression_metrics(repo_test_df['resolution_days'].values, repo_preds)
    print(f"Held-out Repos Performance -> Test MAE: {repo_metrics['MAE']}d | RMSE: {repo_metrics['RMSE']}d | MedAE: {repo_metrics['MedAE']}d | R2: {repo_metrics['R2']}")

    # -----------------------------------------------------------------
    # FEATURE IMPORTANCE & EXPLAINABILITY
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE EXTRACTION (Gain and Weight)")
    print("="*60)
    score_dict = best_xgb_reg.get_score(importance_type='gain')
    total_gain = sum(score_dict.values()) if score_dict else 1.0

    fi_list = []
    for f in feature_names:
        gain = score_dict.get(f, 0.0)
        fi_list.append({
            'Feature': f,
            'Gain': round(gain, 4),
            'Relative_%': round((gain / total_gain) * 100, 2)
        })

    fi_df = pd.DataFrame(fi_list).sort_values('Gain', ascending=False).reset_index(drop=True)
    print("\nTop 15 Most Influential Features:")
    print(fi_df.head(15).to_string(index=False))

    # -----------------------------------------------------------------
    # SAVE PIPELINES AND EXPERIMENT ARTIFACTS
    # -----------------------------------------------------------------
    # 1. Feature Transformer state
    with open('models/feature_transformer.json', 'w') as fp:
        json.dump(transformer.to_dict(), fp, indent=2)

    # 2. Native XGBoost models
    best_xgb_reg.save_model('models/best_regressor.json')
    best_xgb_clf.save_model('models/best_classifier.json')

    # 3. Evaluation Summary & Tables
    summary = {
        'dataset_size': len(df),
        'num_repositories': int(df['repo_name'].nunique()),
        'target_summary': {
            'min_days': round(float(df['resolution_days'].min()), 2),
            'median_days': round(float(df['resolution_days'].median()), 2),
            'mean_days': round(float(df['resolution_days'].mean()), 2),
            'max_days': round(float(df['resolution_days'].max()), 2),
            'resolved_within_7d_rate': round(float(df['resolved_within_7d'].mean()), 4)
        },
        'regression_comparison': reg_results,
        'classification_comparison': clf_results,
        'early_triage_ablation': {
            'creation_only_test_mae': [r['Test_MAE'] for r in reg_results if r['Model'] == 'XGBoost'][0],
            't3h_augmented_test_mae': t3h_metrics['MAE'],
            't3h_augmented_test_rmse': t3h_metrics['RMSE'],
            't3h_augmented_test_r2': t3h_metrics['R2']
        },
        'repo_heldout_generalization': {
            'heldout_repos': held_repos,
            'test_mae': repo_metrics['MAE'],
            'test_rmse': repo_metrics['RMSE'],
            'test_r2': repo_metrics['R2'],
            'test_medae': repo_metrics['MedAE']
        },
        'uncertainty_calibration': {
            'p10_residual': round(p10_res, 3),
            'p90_residual': round(p90_res, 3)
        },
        'top_features': fi_df.head(20).to_dict(orient='records')
    }

    with open('models/evaluation_summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2)

    pd.DataFrame(reg_results).to_csv('models/regression_results.csv', index=False)
    pd.DataFrame(clf_results).to_csv('models/classification_results.csv', index=False)
    fi_df.to_csv('models/feature_importances.csv', index=False)

    print("\nSaved all models, transformers, and evaluation reports to 'models/'. Pipeline complete!")


if __name__ == '__main__':
    run_experiment_pipeline()
