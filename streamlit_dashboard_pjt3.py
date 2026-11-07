import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:\Users\User\Downloads\HR_Job_Placement_Dataset (1).csv"
    )
    return df

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Summary",
        "Dataset Explorer",
        "Candidate Performance",
        "Skills Analysis",
        "Experience Analysis",
        "Recruitment Analytics",
        "Correlation Analysis"
    ]
)

st.title("🎯 Job Acceptance Prediction System")

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

if page == "Executive Summary":

    total_candidates = len(df)

    placed = len(
        df[df["status"] == "Placed"]
    )

    not_placed = len(
        df[df["status"] == "Not Placed"]
    )

    placement_rate = round(
        (placed / total_candidates) * 100,
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

    avg_ctc = round(
        df["expected_ctc_lpa"].mean(),
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

    c5, c6, c7, c8 = st.columns(4)

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

    c8.metric(
        "Avg Expected CTC",
        avg_ctc
    )

    st.subheader("Placement Distribution")

    fig, ax = plt.subplots(figsize=(2.5,2.5))

    df["status"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    st.pyplot(fig, use_container_width=False)

# =====================================================
# CANDIDATE PERFORMANCE
# =====================================================

elif page == "Candidate Performance":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Degree Percentage vs Placement")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.boxplot(
            data=df,
            x="status",
            y="degree_percentage",
            ax=ax
        )

        st.pyplot(fig)

    with col2:
        st.subheader("Technical Score vs Placement")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.boxplot(
            data=df,
            x="status",
            y="technical_score",
            ax=ax
        )

        st.pyplot(fig)

    st.subheader("Aptitude Score vs Placement")

    fig, ax = plt.subplots(figsize=(3,2))

    sns.boxplot(
        data=df,
        x="status",
        y="aptitude_score",
        ax=ax
    )

    ax.set_xlabel("Status", fontsize=8)
    ax.set_ylabel("Degree %", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    #st.pyplot(fig)
# =====================================================
# SKILLS ANALYSIS
# =====================================================

elif page == "Skills Analysis":

    st.subheader(
        "Skills Match Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.histplot(
        data=df,
        x="skills_match_percentage",
        hue="status",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "Certification Impact"
    )

    cert = (
        df.groupby(
            "certifications_count"
        )["status"]
        .value_counts()
        .unstack()
    )

    st.bar_chart(cert)

    st.subheader(
        "Internship Experience"
    )

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="internship_experience",
        hue="status",
        ax=ax
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

# =====================================================
# EXPERIENCE ANALYSIS
# =====================================================

elif page == "Experience Analysis":

    st.subheader("Experience Analysis")

    # Create experience categories
    df["Experience_Category"] = pd.cut(
        df["years_of_experience"],
        bins=[-1, 0, 2, 5],
        labels=["Fresher", "Junior", "Experienced"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("Experience vs Placement")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.boxplot(
            data=df,
            x="status",
            y="years_of_experience",
            ax=ax
        )

        ax.set_xlabel("Status", fontsize=8)
        ax.set_ylabel("Experience", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with col2:
        st.write("Experience Category Distribution")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.countplot(
            data=df,
            x="Experience_Category",
            hue="status",
            ax=ax
        )

        ax.set_xlabel("Category", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

        # optional: smaller legend
        ax.legend(fontsize=6)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

# =====================================================
# RECRUITMENT ANALYTICS
# =====================================================

elif page == "Recruitment Analytics":

    st.subheader("Placement Factors Analysis")

    # First Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Company Tier**")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.countplot(
            data=df,
            x="company_tier",
            hue="status",
            ax=ax
        )

        ax.set_xlabel("Tier", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        plt.xticks(rotation=30)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with col2:
        st.markdown("**Job Role Match**")

        fig, ax = plt.subplots(figsize=(3,2))

        sns.countplot(
            data=df,
            x="job_role_match",
            hue="status",
            ax=ax
        )

        ax.set_xlabel("Role Match", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        plt.xticks(rotation=20)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # Second Row
    st.markdown("**Competition Level**")

    fig, ax = plt.subplots(figsize=(4,2))

    sns.countplot(
        data=df,
        x="competition_level",
        hue="status",
        ax=ax
    )

    ax.set_xlabel("Competition", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
# =====================================================
# CORRELATION ANALYSIS
# =====================================================

elif page == "Correlation Analysis":

    st.subheader(
        "Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(df.head())

    st.subheader(
        "Dataset Shape"
    )

    st.write(df.shape)

    st.subheader(
        "Data Types"
    )

    st.write(df.dtypes)

    st.subheader(
        "Missing Values"
    )

    st.write(
        df.isnull().sum()
    )

    st.subheader(
        "Summary Statistics"
    )

    st.write(
        df.describe()
    )
    