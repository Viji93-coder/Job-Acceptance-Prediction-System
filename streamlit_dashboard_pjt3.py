import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide"
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
          r"C:\Users\User\Downloads\HR_Job_Placement_Dataset (1).csv"
    )

df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

try:
    model = joblib.load(
        "placement_prediction_model.pkl"
    )
except:
    model = None

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Introduction",
        "Executive Summary",
        "Dataset Explorer",
        "Candidate Performance",
        "Skills Analysis",
        "Experience Analysis",
        "Recruitment Analytics",
        "Correlation Analysis",
        "Prediction System"
    ]
)

st.title(
    "🎯 Job Acceptance Prediction System"
)
# ====================================================
# Introduction
# ====================================================
if page == "Introduction":


    job_url = r"C:\Users\User\OneDrive\Desktop\job-offer-acceptance.jpg"

    st.image(job_url, use_container_width=True)

    st.markdown("""
    ### Project Overview

    This dashboard provides insights:

    - Predict whether a candidate will accept or reject a job offer.
    - Analyze factors such as education, skills, experience, and interview performance that influence job acceptance.
    - Perform data cleaning, preprocessing, and exploratory data analysis (EDA) on recruitment data.
    - Build and evaluate a machine learning model for accurate prediction.
    - Identify the key factors affecting placement success.
    - Develop an interactive Streamlit dashboard for visualization and real-time prediction.
    - Help HR teams make better recruitment decisions using data-driven insights.

    """)
# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

if page == "Executive Summary":

    total_candidates = len(df)

    placed = (
        df["status"] == "Placed"
    ).sum()

    not_placed = (
        df["status"] == "Not Placed"
    ).sum()

    placement_rate = round(
        placed / total_candidates * 100,
        2
    )

    avg_tech = round(
        df["technical_score"].mean(),
        2
    )

    avg_skill = round(
        df["skills_match_percentage"].mean(),
        2
    )

    avg_exp = round(
        df["years_of_experience"].mean(),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Candidates",
        total_candidates
    )

    c2.metric(
        "Placed",
        placed
    )

    c3.metric(
        "Not Placed",
        not_placed
    )

    c4.metric(
        "Placement Rate %",
        placement_rate
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Avg Technical Score",
        avg_tech
    )

    c6.metric(
        "Avg Skills Match %",
        avg_skill
    )

    c7.metric(
        "Avg Experience",
        avg_exp
    )

    fig, ax = plt.subplots(figsize=(3,3))

    df["status"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")
    st.pyplot(fig, use_container_width=False)

# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe())

elif page == "Candidate Performance":

    st.subheader("Technical Score vs Placement")

    fig, ax = plt.subplots(figsize=(4,3))

    sns.boxplot(
        data=df,
        x="status",
        y="technical_score",
        ax=ax
    )
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
   

    st.subheader("Aptitude Score vs Placement")

    fig, ax = plt.subplots(figsize=(4,3))

    sns.boxplot(
        data=df,
        x="status",
        y="aptitude_score",
        ax=ax
    )
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
   

    st.subheader("Communication Score vs Placement")

    fig, ax = plt.subplots(figsize=(4,3))

    sns.boxplot(
        data=df,
        x="status",
        y="communication_score",
        ax=ax
    )
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
    
# =====================================================
# SKILLS ANALYSIS
# =====================================================

elif page == "Skills Analysis":

    st.subheader("Skills Match Distribution")

    fig, ax = plt.subplots(figsize=(4,3))

    sns.histplot(
        data=df,
        x="skills_match_percentage",
        hue="status",
        kde=True,
        ax=ax
    )
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
    

# =====================================================
# EXPERIENCE ANALYSIS
# =====================================================

elif page == "Experience Analysis":

    temp_df = df.copy()

    temp_df["Experience_Category"] = pd.cut(
        temp_df["years_of_experience"],
        bins=[-1, 0, 2, 5, 20],
        labels=[
            "Fresher",
            "Junior",
            "Mid-Level",
            "Senior"
        ]
    )

    fig, ax = plt.subplots(figsize=(4,3))

    sns.countplot(
        data=temp_df,
        x="Experience_Category",
        hue="status",
        ax=ax
    )
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
    

# =====================================================
# RECRUITMENT ANALYTICS
# =====================================================

elif page == "Recruitment Analytics":

    fig, ax = plt.subplots(figsize=(4,3))

    sns.countplot(
        data=df,
        x="company_tier",
        hue="status",
        order=["Tier 1", "Tier 2", "Tier 3"],
        ax=ax
    )

    plt.xticks(rotation=20)
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=False)
    

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

elif page == "Correlation Analysis":

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=True,
        ax=ax
    )

    st.pyplot(fig)


# =====================================================
# PREDICTION SYSTEM
# =====================================================

elif page == "Prediction System":

    st.header("🎯 Job Acceptance Prediction")

    if model is None:

        st.error("Model file not found!")

    else:

        col1, col2 = st.columns(2)

        with col1:

            age_years = st.number_input(
                "Age",
                18,
                60,
                25
            )

            gender = st.selectbox(
                "Gender (Encoded)",
                [0, 1]
            )

            ssc_percentage = st.slider(
                "SSC Percentage",
                0.0, 100.0, 70.0
            )

            hsc_percentage = st.slider(
                "HSC Percentage",
                0.0, 100.0, 70.0
            )

            degree_percentage = st.slider(
                "Degree Percentage",
                0.0, 100.0, 70.0
            )

            degree_specialization = st.selectbox(
                "Degree Specialization (Encoded)",
                [0, 1, 2, 3, 4]
            )

            technical_score = st.slider(
                "Technical Score",
                0.0, 100.0, 75.0
            )

            aptitude_score = st.slider(
                "Aptitude Score",
                0.0, 100.0, 75.0
            )

            communication_score = st.slider(
                "Communication Score",
                0.0, 100.0, 75.0
            )

            skills_match_percentage = st.slider(
                "Skills Match %",
                0.0, 100.0, 80.0
            )

            certifications_count = st.slider(
                "Certifications Count",
                0, 20, 2
            )

            internship_experience = st.selectbox(
                "Internship Experience",
                [0, 1]
            )

            years_of_experience = st.slider(
                "Years Of Experience",
                0.0, 20.0, 2.0
            )

        with col2:

            career_switch_willingness = st.selectbox(
                "Career Switch Willingness",
                [0, 1]
            )

            relevant_experience = st.selectbox(
                "Relevant Experience",
                [0, 1]
            )

            previous_ctc_lpa = st.number_input(
                "Previous CTC (LPA)",
                value=3.0
            )

            expected_ctc_lpa = st.number_input(
                "Expected CTC (LPA)",
                value=5.0
            )

            company_tier = st.selectbox(
                "Company Tier",
                [1, 2, 3]
            )

            job_role_match = st.selectbox(
                "Job Role Match",
                [0, 1]
            )

            competition_level = st.selectbox(
                "Competition Level",
                [1, 2, 3]
            )

            bond_requirement = st.selectbox(
                "Bond Requirement",
                [0, 1]
            )

            notice_period_days = st.slider(
                "Notice Period Days",
                0,
                120,
                30
            )

            layoff_history = st.selectbox(
                "Layoff History",
                [0, 1]
            )

            employment_gap_months = st.slider(
                "Employment Gap Months",
                0,
                60,
                0
            )

            relocation_willingness = st.selectbox(
                "Relocation Willingness",
                [0, 1]
            )

        if st.button("Predict Job Acceptance"):

            try:

                input_df = pd.DataFrame({

                    'age_years':[age_years],
                    'gender':[gender],
                    'ssc_percentage':[ssc_percentage],
                    'hsc_percentage':[hsc_percentage],
                    'degree_percentage':[degree_percentage],
                    'degree_specialization':
                        [degree_specialization],
                    'technical_score':
                        [technical_score],
                    'aptitude_score':
                        [aptitude_score],
                    'communication_score':
                        [communication_score],
                    'skills_match_percentage':
                        [skills_match_percentage],
                    'certifications_count':
                        [certifications_count],
                    'internship_experience':
                        [internship_experience],
                    'years_of_experience':
                        [years_of_experience],
                    'career_switch_willingness':
                        [career_switch_willingness],
                    'relevant_experience':
                        [relevant_experience],
                    'previous_ctc_lpa':
                        [previous_ctc_lpa],
                    'expected_ctc_lpa':
                        [expected_ctc_lpa],
                    'company_tier':
                        [company_tier],
                    'job_role_match':
                        [job_role_match],
                    'competition_level':
                        [competition_level],
                    'bond_requirement':
                        [bond_requirement],
                    'notice_period_days':
                        [notice_period_days],
                    'layoff_history':
                        [layoff_history],
                    'employment_gap_months':
                        [employment_gap_months],
                    'relocation_willingness':
                        [relocation_willingness]

                })

                prediction = model.predict(
                    input_df
                )[0]

                st.subheader(
                    "Prediction Result"
                )

                if prediction == 1:

                    st.success(
                        "✅ Candidate Likely To Accept Job Offer"
                    )

                else:

                    st.error(
                        "❌ Candidate Likely To Reject Job Offer"
                    )

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probability = (
                        model.predict_proba(
                            input_df
                        )[0][1]
                    )

                    st.metric(
                        "Acceptance Probability",
                        f"{probability*100:.2f}%"
                    )
                st.subheader("Input Summary")

                st.dataframe(input_df)

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )
    