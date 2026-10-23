"""
app.py
GitHub Issue Resolution Time Predictor
Interactive KDD Dashboard & Machine Learning Prediction System

Demonstrates how software engineering data becomes actionable knowledge for open-source maintainers.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from src.predict import ResolutionPredictor

# Page Configuration
st.set_page_config(
    page_title="GitHub Issue Resolution Predictor | KDD Process",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top banner card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction Outcome Card */
    .outcome-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin-top: 16px;
        box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.6);
    }
    
    .tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-info { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    /* Recommendation Item */
    .rec-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #60A5FA;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Load predictor pipeline once."""
    return ResolutionPredictor()


@st.cache_data
def get_summary_and_results():
    """Load evaluation summary and benchmark tables."""
    summary = {}
    if os.path.exists('models/evaluation_summary.json'):
        with open('models/evaluation_summary.json', 'r') as fp:
            summary = json.load(fp)

    reg_df = pd.DataFrame()
    if os.path.exists('models/regression_results.csv'):
        reg_df = pd.read_csv('models/regression_results.csv')

    clf_df = pd.DataFrame()
    if os.path.exists('models/classification_results.csv'):
        clf_df = pd.read_csv('models/classification_results.csv')

    feat_df = pd.DataFrame()
    if os.path.exists('models/feature_importances.csv'):
        feat_df = pd.read_csv('models/feature_importances.csv')

    return summary, reg_df, clf_df, feat_df


@st.cache_data
def get_dataset_sample():
    """Load sampled processed dataset for visualizations."""
    clean_path = 'data/processed/issues_clean.csv'
    if os.path.exists(clean_path):
        # sample 5000 rows for instant responsive charts
        df = pd.read_csv(clean_path)
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)
        return df
    return pd.DataFrame()


predictor = get_predictor()
summary, reg_df, clf_df, feat_df = get_summary_and_results()
sample_df = get_dataset_sample()

# Sidebar Navigation
st.sidebar.markdown("### ⚡ Navigation")
page = st.sidebar.radio(
    "Select Stage:",
    [
        "📊 Overview & Discovery",
        "🔮 Issue Prediction Engine",
        "📈 Model Benchmarks",
        "🧠 Explainability & Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ KDD Methodology")
st.sidebar.markdown("""
1. **Selection**: 34,665 Issues across 18 OSS Projects
2. **Preprocessing**: Strict timestamp validation & zero leakage
3. **Transformation**: Creation-time vs T=3h signals
4. **Data Mining**: Baseline, Ridge, Trees, XGBoost
5. **Evaluation**: Chronological future holdouts
6. **Deployment**: Interactive Maintainer Predictor
""")

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity AI | Software Analytics Research")


# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if page == "📊 Overview & Discovery":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Predicting GitHub Issue Resolution Time</div>
        <div class="hero-subtitle">
            An end-to-end Knowledge Discovery in Databases (KDD) framework that turns raw software engineering telemetry into actionable resolution intelligence for open-source maintainers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">34,665</div>
            <div class="metric-lbl">Total GitHub Issues</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">18</div>
            <div class="metric-lbl">OSS Repositories</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">6.35 d</div>
            <div class="metric-lbl">Median Resolution Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">51.3%</div>
            <div class="metric-lbl">Resolved Within 7 Days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Visualizations
    tab1, tab2, tab3 = st.tabs(["📉 Resolution Time Distribution", "🏢 Repository Cadence Comparison", "👥 Contributor Velocity Impact"])

    with tab1:
        st.subheader("Resolution Time Distribution (Heavy Skewness)")
        st.write("""
        Historical issue resolution times exhibit extreme right-skewness: while 50% of issues close within **6.35 days**, a long tail extends past **1,000 days** (mean = **100.4 days**). 
        Applying a logarithmic target transformation $\\log(1 + \\text{days})$ stabilizes model training and prevents extreme outliers from dominating gradients.
        """)
        if not sample_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Distribution of Raw Resolution Days (Capped at 90 days for visualization)")
                capped_days = sample_df['resolution_days'].clip(upper=90)
                st.bar_chart(pd.Series(np.histogram(capped_days, bins=30)[0]))
            with c2:
                st.caption("Distribution of Log-Transformed Target: log(1 + days)")
                st.bar_chart(pd.Series(np.histogram(sample_df['log_resolution_days'], bins=30)[0]))

    with tab2:
        st.subheader("Median Resolution Duration Across 18 Open-Source Projects")
        st.write("""
        Project workflows and triage capacity vary drastically across software domains. Repositories like **views-widgets-samples**, **youtube-dl**, and **lottie-android** close issues in under 3.5 days, while curation repositories like **awesome-python** experience median latencies of **228 days**.
        """)
        if not sample_df.empty:
            repo_medians = sample_df.groupby('repo_name')['resolution_days'].median().sort_values()
            st.bar_chart(repo_medians)

    with tab3:
        st.subheader("Contributor Status: Core Maintainers vs Community Newcomers")
        st.write("""
        Empirical discovery: Issues submitted by **core project maintainers** have a median resolution time of **3.1 days**, compared to **8.4 days** for first-time newcomers.
        """)
        if not sample_df.empty:
            status_df = sample_df.groupby('author_core_member')['resolution_days'].median().reset_index()
            status_df['Status'] = status_df['author_core_member'].map({0: 'Community / External', 1: 'Core Team Member'})
            st.dataframe(status_df[['Status', 'resolution_days']].rename(columns={'resolution_days': 'Median Resolution Days'}))


# ==============================================================================
# PAGE 2: PREDICTION ENGINE
# ==============================================================================
elif page == "🔮 Issue Prediction Engine":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔮 Issue Resolution Predictor</div>
        <div class="hero-subtitle">
            Estimate the expected resolution duration and 7-day closure likelihood for a newly submitted GitHub issue using only <b>creation-time features</b> (strictly zero post-creation leakage).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset selection
    preset = st.selectbox(
        "⚡ Choose a Scenario Preset or Configure Below:",
        [
            "Custom Configuration",
            "Scenario A: Newcomer Bug Report (Missing Snippet & Labels)",
            "Scenario B: Core Maintainer Hotfix with PR Attached",
            "Scenario C: Community Feature Request (High Detail & Code Included)"
        ]
    )

    # Default values based on preset
    if preset == "Scenario A: Newcomer Bug Report (Missing Snippet & Labels)":
        d_repo = "17_youtube-dl"
        d_author_newcomer = 1
        d_author_core = 0
        d_has_pr = 0
        d_has_task_type = 0
        d_body_len = 380
        d_num_codes = 0
        d_num_urls = 1
        d_repo_open_issues = 650
        d_repo_closed_time = 14.0
    elif preset == "Scenario B: Core Maintainer Hotfix with PR Attached":
        d_repo = "2_lottie-android"
        d_author_newcomer = 0
        d_author_core = 1
        d_has_pr = 1
        d_has_task_type = 1
        d_body_len = 160
        d_num_codes = 2
        d_num_urls = 0
        d_repo_open_issues = 120
        d_repo_closed_time = 3.5
    elif preset == "Scenario C: Community Feature Request (High Detail & Code Included)":
        d_repo = "7_openapi-generator"
        d_author_newcomer = 0
        d_author_core = 0
        d_has_pr = 0
        d_has_task_type = 1
        d_body_len = 650
        d_num_codes = 4
        d_num_urls = 2
        d_repo_open_issues = 340
        d_repo_closed_time = 6.0
    else:
        d_repo = "17_youtube-dl"
        d_author_newcomer = 1
        d_author_core = 0
        d_has_pr = 0
        d_has_task_type = 1
        d_body_len = 250
        d_num_codes = 1
        d_num_urls = 1
        d_repo_open_issues = 350
        d_repo_closed_time = 8.0

    with st.form("prediction_form"):
        st.subheader("1. Repository & Project Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            repo_list = [
                '17_youtube-dl', '7_openapi-generator', '18_models', '9_dolphinscheduler',
                '15_Python', '4_rocketmq', '3_ghidra', '13_public-apis', '8_thingsboard',
                '5_material-components-android', '16_awesome-python', '12_spring-cloud-gateway',
                '2_lottie-android', '10_rocketmq-externals', '14_system-design-primer',
                '1_interviews', '6_Algorithms', '11_views-widgets-samples'
            ]
            repo_name = st.selectbox("Repository Identifier", repo_list, index=repo_list.index(d_repo) if d_repo in repo_list else 0)
        with c2:
            repo_open_issue_num = st.number_input("Current Open Issues Backlog", min_value=0, max_value=5000, value=d_repo_open_issues, step=50)
        with c3:
            repo_closed_issue_time = st.number_input("Historical Repo Median Closed Time (Days)", min_value=0.5, max_value=300.0, value=float(d_repo_closed_time), step=1.0)

        st.subheader("2. Author & Contributor Profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            author_core_member = st.selectbox("Core Maintainer?", [0, 1], index=d_author_core, format_func=lambda x: "Yes (Maintainer)" if x == 1 else "No (External)")
        with c2:
            author_newcomer = st.selectbox("Newcomer Contributor?", [0, 1], index=d_author_newcomer, format_func=lambda x: "Yes (First Issue)" if x == 1 else "No (Returning)")
        with c3:
            author_followers = st.number_input("Author Follower Count", min_value=0, max_value=5000, value=15, step=5)
        with c4:
            author_num_commits_inproject = st.number_input("Author Prior Project Commits", min_value=0, max_value=1000, value=5 if d_author_core else 0, step=1)

        st.subheader("3. Issue Characteristics & Body Quality")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            issue_has_pr = st.selectbox("Pull Request Attached?", [0, 1], index=d_has_pr, format_func=lambda x: "Yes (PR Attached)" if x == 1 else "No PR")
        with c2:
            issue_has_task_type = st.selectbox("Task Type Label Applied?", [0, 1], index=d_has_task_type, format_func=lambda x: "Yes (Labeled)" if x == 1 else "No (Unlabeled)")
        with c3:
            issue_cleaned_body_len = st.number_input("Description Character Length", min_value=10, max_value=5000, value=d_body_len, step=50)
        with c4:
            issue_num_codes = st.number_input("Number of Code Blocks", min_value=0, max_value=50, value=d_num_codes, step=1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            issue_num_urls = st.number_input("External URLs Included", min_value=0, max_value=20, value=d_num_urls, step=1)
        with c2:
            issue_flesch_reading_ease = st.slider("Flesch Reading Ease", min_value=10.0, max_value=100.0, value=65.0)
        with c3:
            issue_text_similarity = st.slider("Similarity to Past Issues", min_value=0.0, max_value=1.0, value=0.5)
        with c4:
            is_weekend = st.selectbox("Opened on Weekend?", [0, 1], index=0, format_func=lambda x: "Yes (Saturday/Sunday)" if x == 1 else "No (Weekday)")

        submit = st.form_submit_button("⚡ Predict Resolution Time", use_container_width=True)

    if submit:
        # Build feature dictionary
        features = {
            'repo_name': repo_name,
            'repo_age': 1500.0,
            'repo_contributors': 250,
            'repo_subscribers': 1800,
            'repo_team_size': 15,
            'repo_open_issue_num': repo_open_issue_num,
            'repo_open_delta': 1.0,
            'repo_closed_issue_time': repo_closed_issue_time,
            'repo_num_commits_inproject': 5000,
            'repo_num_issues_created_inproject': 7000,
            'repo_num_issues_created_inproject_closed': 6500,
            'issue_cleaned_body_len': float(issue_cleaned_body_len),
            'issue_num_codes': int(issue_num_codes),
            'issue_num_urls': int(issue_num_urls),
            'issue_num_pics': 0,
            'issue_flesch_reading_ease': float(issue_flesch_reading_ease),
            'issue_desc_positivity': 0.1,
            'issue_desc_negativity': 0.05,
            'issue_desc_pos_polarity': 0.1,
            'issue_desc_subjectivity': 0.4,
            'issue_text_similarity': float(issue_text_similarity),
            'author_age': 500.0,
            'author_followers': int(author_followers),
            'author_comments_negative': 0.0,
            'author_comments_positive': 0.2,
            'author_num_commits_inproject': int(author_num_commits_inproject),
            'author_num_issues_inproject': 3 if author_newcomer == 0 else 1,
            'author_num_issues_closed_inproject': 2 if author_newcomer == 0 else 0,
            'created_hour': 14,
            'created_dayofweek': 5 if is_weekend else 2,
            'created_month': 5,
            'created_year': 2021,
            'repo_has_org': 1,
            'issue_has_task_type': int(issue_has_task_type),
            'issue_has_pr': int(issue_has_pr),
            'issue_has_conflict': 0,
            'issue_on_rest_day': int(is_weekend),
            'issue_has_at_tag': 0,
            'author_org': 0,
            'author_core_member': int(author_core_member),
            'author_newcomer': int(author_newcomer),
            'author_first_issue': int(author_newcomer),
            'is_weekend': int(is_weekend)
        }

        with st.spinner("Generating prediction and maintainer recommendations..."):
            pred_res = predictor.predict_single(features)

        # Display Outcome Card
        st.markdown(f"""
        <div class="outcome-box">
            <span class="tier-badge badge-{pred_res['tier_badge']}">{pred_res['triage_tier']}</span>
            <div style="display: flex; gap: 40px; align-items: baseline; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase;">Estimated Resolution Time</span>
                    <div style="font-size: 3.2rem; font-weight: 800; color: #60A5FA;">
                        {pred_res['predicted_days']} <span style="font-size: 1.5rem; color: #94A3B8;">Days</span>
                    </div>
                </div>
                <div>
                    <span style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase;">80% Calibrated Residual Interval</span>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #E2E8F0;">
                        [{pred_res['prediction_interval_days'][0]}d — {pred_res['prediction_interval_days'][1]}d]
                    </div>
                </div>
                <div>
                    <span style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase;">Likelihood to Resolve ≤ 7 Days</span>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {'#4ADE80' if pred_res['prob_resolved_within_7d'] >= 50 else '#F87171'};">
                        {pred_res['prob_resolved_within_7d']}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Actionable Recommendations for Maintainers")
        for rec in pred_res['actionable_recommendations']:
            st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: MODEL BENCHMARKS
# ==============================================================================
elif page == "📈 Model Benchmarks":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📈 Model Evaluation & Benchmarks</div>
        <div class="hero-subtitle">
            Rigorous evaluation across increasing model complexity following KDD Stage 4 & 5. All models evaluated on strictly chronologically held-out future issues (November 2020 — September 2021).
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Regression Benchmarks", "🏷️ 7-Day Classification Benchmarks", "🔬 Lifecycle Ablation & Generalization"])

    with tab1:
        st.subheader("Primary Task: Predicting Continuous Elapsed Days")
        st.write("""
        Models evaluated on held-out future issues using **MAE** (Mean Absolute Error, in calendar days), **RMSE**, **MedAE** (Median Absolute Error), and **R²**.
        Target is trained with logarithmic transformation $\\log(1 + y)$ and back-transformed to days for reporting.
        """)
        if not reg_df.empty:
            st.dataframe(
                reg_df[['Model', 'Test_MAE', 'Test_RMSE', 'Test_MedAE', 'Test_R2', 'Train_Time_s']].rename(
                    columns={
                        'Test_MAE': 'Test MAE (Days)',
                        'Test_RMSE': 'Test RMSE (Days)',
                        'Test_MedAE': 'Test MedAE (Days)',
                        'Test_R2': 'Test R²',
                        'Train_Time_s': 'Training Time (s)'
                    }
                ),
                use_container_width=True
            )
            st.bar_chart(reg_df.set_index('Model')['Test_MAE'])

    with tab2:
        st.subheader("Secondary Task: Predicting Resolution Within 7 Days")
        st.write("""
        Binary classification benchmarks predicting whether an issue resolves within $\\le 7$ days.
        Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
        """)
        if not clf_df.empty:
            st.dataframe(
                clf_df[['Model', 'Test_Accuracy', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Test_ROC_AUC', 'Train_Time_s']].rename(
                    columns={
                        'Test_Accuracy': 'Accuracy',
                        'Test_F1': 'F1 Score',
                        'Test_Precision': 'Precision',
                        'Test_Recall': 'Recall',
                        'Test_ROC_AUC': 'ROC-AUC',
                        'Train_Time_s': 'Training Time (s)'
                    }
                ),
                use_container_width=True
            )
            st.bar_chart(clf_df.set_index('Model')['Test_ROC_AUC'])

    with tab3:
        st.subheader("Lifecycle Ablation: Value of Early 3-Hour Triage Signals")
        st.write("""
        Does observing the first 3 hours of issue lifecycle activity (comments, assignees, participants) improve prediction over creation-time signals alone?
        """)
        ablation = summary.get('early_triage_ablation', {})
        if ablation:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Creation-Time Only Test MAE", f"{ablation.get('creation_only_test_mae', 20.67):.2f} Days")
            with c2:
                st.metric("3-Hour Augmented Test MAE", f"{ablation.get('t3h_augmented_test_mae', 19.80):.2f} Days", delta="-0.87 Days (Improvement)")

        st.markdown("---")
        st.subheader("Cross-Repository Generalization (Held-out Repos)")
        st.write("""
        Evaluating the model on completely unseen projects (`spring-cloud-gateway`, `lottie-android`, `thingsboard`) reveals an out-of-distribution Test MAE of **73.29 days**.
        **Conclusion**: Issue resolution cadence is deeply repository-specific. Accurate cross-project generalization requires repository context and calibration.
        """)


# ==============================================================================
# PAGE 4: EXPLAINABILITY & KNOWLEDGE DISCOVERY
# ==============================================================================
elif page == "🧠 Explainability & Insights":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧠 Explainability & Actionable Knowledge</div>
        <div class="hero-subtitle">
            KDD Stage 5: Discovering which attributes drive software issue resolution velocity and translating findings into maintainer best practices.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Top Influential Features (XGBoost Relative Gain %)")
    st.write("Feature importance derived from gradient boosted tree split gains across the dataset:")

    if not feat_df.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.bar_chart(feat_df.head(12).set_index('Feature')['Relative_%'])
        with c2:
            st.dataframe(feat_df.head(15)[['Feature', 'Relative_%']], use_container_width=True)

    st.markdown("---")
    st.subheader("Key Findings & Actionable Recommendations for Maintainers")

    st.markdown("""
    1. **Core Team Ownership Accelerates Resolution**:
       - `author_core_member` is the #1 single predictive driver (12.17% gain).
       - Maintainer-authored issues resolve **2.7x faster** than newcomer issues.
       - *Action*: Establish automated newcomer triage bots that assist first-time authors in formatting bug reports according to maintainer templates.

    2. **Pull Requests Drastically Shorten Lifecycle**:
       - `issue_has_pr` ranks among the top features (5.89% gain).
       - Linking a pull request at issue creation provides ready-made code changes, removing the investigation bottleneck.
       - *Action*: Encourage community contributors to open draft pull requests alongside issue reports.

    3. **Code Snippets Prevent Back-and-Forth Latency**:
       - Descriptions exceeding 300 characters without code snippets suffer significant triage delays as maintainers request reproducibility info.
       - *Action*: Implement required GitHub issue forms with dedicated code snippet blocks.

    4. **Project Visibility vs Triage Bottlenecks**:
       - High subscriber count and backlog volume (`repo_subscribers`, `repo_open_issue_num`) are strong drivers of latency.
       - *Action*: Maintainers of high-visibility repositories should use automated priority triage to prevent issues from slipping into the multi-month tail.
    """)
