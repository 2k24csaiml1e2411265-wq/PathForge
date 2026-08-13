"""
PathForge — Career Intelligence Dashboard

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.document_processing import preprocess_resume
from src.skill_extraction import extract_skills, extract_skills_from_job_row
from src.embedding_engine import create_embeddings
from src.vector_search import build_faiss_index, search_similar_jobs
from src.skill_gap import analyze_skill_gap_across_jobs
from src.market_analysis import top_demand_skills
from src.transferability import (
    average_transferability,
    compute_transferability_scores,
)
from src.ai_exposure import classify_skills, summarize_exposure
from src.resilience import compute_resilience_score, resilience_band
from src.recommendation_engine import build_recommendations
from src.gemini_service import (
    generate_career_summary,
    generate_roadmap,
    is_gemini_available,
)
from src.utils import load_jobs_dataframe, SAMPLE_DATA_DIR


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PathForge | Career Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — CLEAN LIGHT PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------- GLOBAL -------------------- */

    .stApp {
        background: #f6f8fb;
        color: #172033;
    }

    .main {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* -------------------- TOP HEADER -------------------- */

    .pf-header {
        background: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .pf-title {
        font-size: 34px;
        font-weight: 700;
        color: #172033;
        margin: 0;
        letter-spacing: -0.6px;
    }

    .pf-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-top: 6px;
    }

    /* -------------------- SIDEBAR -------------------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e9f0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.3rem 1rem;
    }

    .sidebar-title {
        font-size: 23px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 2px;
    }

    .sidebar-subtitle {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 20px;
    }

    /* -------------------- SECTION HEADINGS -------------------- */

    h1, h2, h3, h4 {
        color: #172033 !important;
    }

    h3 {
        font-size: 21px !important;
    }

    h4 {
        font-size: 17px !important;
    }

    /* -------------------- CARDS -------------------- */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 12px;
        padding: 18px;
        min-height: 110px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #172033;
        font-size: 28px;
        font-weight: 700;
    }

    .metric-small {
        color: #6b7280;
        font-size: 12px;
        margin-top: 5px;
    }

    /* -------------------- INFO BOX -------------------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 12px;
        padding: 20px 22px;
        margin: 8px 0 18px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    /* -------------------- NAV TABS -------------------- */

    button[data-baseweb="tab"] {
        color: #5b6472 !important;
        font-weight: 500;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 700;
    }

    /* -------------------- BUTTONS -------------------- */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #2563eb;
        background: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
        color: white;
    }

    /* -------------------- FILE UPLOADER -------------------- */

    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border-radius: 10px;
    }

    /* -------------------- DATAFRAME -------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #e5e9f0;
        border-radius: 10px;
    }

    /* -------------------- ALERTS -------------------- */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* -------------------- FOOTER -------------------- */

    .pf-footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #e5e9f0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def _load_jobs():
    return load_jobs_dataframe()


@st.cache_resource(show_spinner="Preparing job search...")
def _build_index(jobs_df: pd.DataFrame):
    corpus = (
        jobs_df["job_title"]
        + " "
        + jobs_df["description"]
        + " "
        + jobs_df["skills"]
    ).tolist()

    vectors, engine = create_embeddings(corpus)

    index = build_faiss_index(
        vectors,
        jobs_df["job_id"].tolist(),
    )

    return engine, index


jobs_df = _load_jobs()

if jobs_df.empty:
    st.error(
        "No job data found in data/jobs.csv. "
        "Run the provided dataset generation script and restart the app."
    )
    st.stop()

embedding_engine, job_index = _build_index(jobs_df)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">PathForge</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">'
    'Career analytics and skill-gap platform'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Resume")

uploaded_file = st.sidebar.file_uploader(
    "Upload your resume",
    type=["pdf", "txt"],
)

use_sample = st.sidebar.checkbox(
    "Use sample resume",
    value=uploaded_file is None,
)

st.sidebar.markdown("### Target Role")

role_options = [
    "(auto — best overall match)"
] + sorted(
    jobs_df["job_title"].unique().tolist()
)

target_role_choice = st.sidebar.selectbox(
    "Choose a target role",
    role_options,
)

target_role = (
    None
    if target_role_choice.startswith("(auto")
    else target_role_choice
)

top_k = st.sidebar.slider(
    "Number of job matches",
    min_value=3,
    max_value=15,
    value=6,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Job postings available: {len(jobs_df):,}"
)

st.sidebar.caption(
    f"Search model: {embedding_engine.backend_name}"
)

gemini_status = (
    "Available"
    if is_gemini_available()
    else "Optional — local fallback"
)

st.sidebar.caption(
    f"AI guidance: {gemini_status}"
)

run_analysis = st.sidebar.button(
    "Analyze Profile",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RESUME PROCESSING
# ============================================================

def _get_resume_text() -> tuple[str, str]:

    if uploaded_file is not None and not use_sample:

        text = preprocess_resume(
            uploaded_file,
            filename=uploaded_file.name,
        )

        return text, uploaded_file.name

    sample_path = SAMPLE_DATA_DIR / "sample_resume.txt"

    if sample_path.exists():

        text = preprocess_resume(
            sample_path,
            filename=str(sample_path),
        )

        return text, "sample_resume.txt"

    return "", "none"


if "analysis" not in st.session_state:
    st.session_state.analysis = None


if run_analysis or (
    st.session_state.analysis is None
    and use_sample
):

    resume_text, resume_name = _get_resume_text()

    if not resume_text.strip():

        st.sidebar.error(
            "No readable resume text was found."
        )

    else:

        with st.spinner(
            "Analyzing profile and matching relevant roles..."
        ):

            # -----------------------------
            # SKILLS
            # -----------------------------

            skills_result = extract_skills(
                resume_text
            )

            current_skills = skills_result["_all"]

            # -----------------------------
            # SEMANTIC SEARCH
            # -----------------------------

            query_vec = embedding_engine.encode(
                [resume_text]
            )[0]

            matches = search_similar_jobs(
                job_index,
                query_vec,
                top_k=top_k,
            )

            top_jobs = []
            required_skill_lists = []

            for job_id, score in matches:

                row = jobs_df[
                    jobs_df.job_id == job_id
                ].iloc[0]

                req_skills = extract_skills_from_job_row(
                    row["skills"]
                )

                required_skill_lists.append(
                    req_skills
                )

                top_jobs.append(
                    {
                        "job_id": int(job_id),
                        "job_title": row["job_title"],
                        "company": row["company"],
                        "location": row["location"],
                        "seniority": row["seniority"],
                        "similarity": float(score),
                        "required_skills": req_skills,
                        "matching_skills": [
                            s
                            for s in req_skills
                            if s in current_skills
                        ],
                    }
                )

            # -----------------------------
            # SKILL GAP
            # -----------------------------

            gap = analyze_skill_gap_across_jobs(
                current_skills,
                required_skill_lists,
            )

            # -----------------------------
            # RESILIENCE
            # -----------------------------

            resilience = compute_resilience_score(
                current_skills,
                gap["skill_coverage_percentage"],
                jobs_df,
            )

            # -----------------------------
            # RECOMMENDATIONS
            # -----------------------------

            recommendations = build_recommendations(
                gap,
                target_role=target_role,
                jobs_df=jobs_df,
            )

            # -----------------------------
            # AI EXPOSURE
            # -----------------------------

            exposure = classify_skills(
                current_skills
            )

            # -----------------------------
            # STRUCTURED CONTEXT
            # -----------------------------

            structured_for_llm = {
                "current_skills": current_skills,
                "missing_skills": gap["missing_skills"],
                "top_jobs": top_jobs,
                "skill_coverage_percentage":
                    gap["skill_coverage_percentage"],
                "resilience_score":
                    resilience["resilience_score"],
                "skills_to_learn":
                    recommendations["skills_to_learn"],
            }

            # -----------------------------
            # SUMMARY + ROADMAP
            # -----------------------------

            summary = generate_career_summary(
                structured_for_llm
            )

            roadmap = generate_roadmap(
                structured_for_llm
            )

            st.session_state.analysis = {
                "resume_name": resume_name,
                "resume_text": resume_text,
                "skills_result": skills_result,
                "current_skills": current_skills,
                "top_jobs": top_jobs,
                "gap": gap,
                "resilience": resilience,
                "recommendations": recommendations,
                "exposure": exposure,
                "summary": summary,
                "roadmap": roadmap,
            }


analysis = st.session_state.analysis


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="pf-header">
        <div class="pf-title">PathForge</div>
        <div class="pf-subtitle">
            Career analytics, skill-gap analysis and personalized career planning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if analysis is None:

    st.markdown(
        """
        <div class="info-card">
            <h3>Start your career analysis</h3>
            <p>
            Upload a resume from the sidebar, select a target role,
            and choose <b>Analyze Profile</b>.
            </p>
            <p>
            PathForge will identify your skills, compare them with
            relevant roles, highlight gaps and suggest practical next steps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Dashboard",
        "Skills",
        "Job Matches",
        "Career Resilience",
        "Recommendations",
        "Learning Roadmap",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    st.subheader("Profile Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Resume analyzed</div>
                <div class="metric-value" style="font-size:18px;">
                    {analysis["resume_name"]}
                </div>
                <div class="metric-small">Current profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Skills detected</div>
                <div class="metric-value">
                    {len(analysis["current_skills"])}
                </div>
                <div class="metric-small">Across all categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Skill coverage</div>
                <div class="metric-value">
                    {analysis["gap"]["skill_coverage_percentage"]}%
                </div>
                <div class="metric-small">Against matched roles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Career resilience</div>
                <div class="metric-value">
                    {analysis["resilience"]["resilience_score"]}/100
                </div>
                <div class="metric-small">
                    {resilience_band(analysis["resilience"]["resilience_score"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Career Summary")

    st.markdown(
        f"""
        <div class="info-card">
            {analysis["summary"]["text"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analysis["top_jobs"]:

        st.markdown("### Top Matching Roles")

        top_df = pd.DataFrame(
            analysis["top_jobs"]
        )[
            [
                "job_title",
                "company",
                "location",
                "similarity",
            ]
        ].copy()

        top_df["similarity"] = (
            top_df["similarity"] * 100
        ).round(1)

        top_df = top_df.rename(
            columns={
                "job_title": "Role",
                "company": "Company",
                "location": "Location",
                "similarity": "Match %",
            }
        )

        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# SKILLS
# ============================================================

with tabs[1]:

    left, right = st.columns(2)

    with left:

        st.markdown("### Current Skills")

        for category, skills in analysis[
            "skills_result"
        ].items():

            if category == "_all" or not skills:
                continue

            st.markdown(
                f"**{category.replace('_', ' ').title()}**"
            )

            st.write(
                ", ".join(skills)
            )

    with right:

        st.markdown("### Missing Skills")

        missing = analysis["gap"]["missing_skills"]

        if missing:
            st.write(", ".join(missing))
        else:
            st.success(
                "No major missing skills were identified."
            )

        st.markdown("### Critical Skill Gaps")

        critical = analysis["gap"]["critical_gaps"]

        if critical:
            st.write(", ".join(critical))
        else:
            st.write("No critical gaps identified.")

    st.markdown("### Market Skill Demand")

    demand = top_demand_skills(
        jobs_df,
        top_n=15,
    )

    if demand:

        demand_df = pd.DataFrame(
            demand,
            columns=[
                "skill",
                "demand_score",
            ],
        )

        fig = px.bar(
            demand_df,
            x="demand_score",
            y="skill",
            orientation="h",
        )

        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#172033"),
            xaxis_title="Demand Score",
            yaxis_title="Skill",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# JOB MATCHES
# ============================================================

with tabs[2]:

    st.markdown("### Relevant Job Matches")

    for job in analysis["top_jobs"]:

        with st.container(border=True):

            c1, c2 = st.columns(
                [4, 1]
            )

            with c1:

                st.markdown(
                    f"**{job['job_title']}**"
                )

                st.caption(
                    f"{job['company']} · "
                    f"{job['location']} · "
                    f"{job['seniority']}"
                )

            with c2:

                st.metric(
                    "Match",
                    f"{job['similarity'] * 100:.1f}%",
                )

            st.write(
                "**Required skills:** "
                + ", ".join(
                    job["required_skills"]
                )
            )

            st.write(
                "**Matching skills:** "
                + (
                    ", ".join(
                        job["matching_skills"]
                    )
                    if job["matching_skills"]
                    else "None yet"
                )
            )


# ============================================================
# CAREER RESILIENCE
# ============================================================

with tabs[3]:

    resilience = analysis["resilience"]

    st.markdown(
        f"### Career Resilience Score: "
        f"{resilience['resilience_score']} / 100"
    )

    st.caption(
        resilience["formula"]
    )

    comp_df = pd.DataFrame(
        [
            {
                "Component":
                    k.replace(
                        "_", " "
                    ).title(),
                "Score": v,
            }
            for k, v in resilience[
                "components"
            ].items()
        ]
    )

    fig = px.bar(
        comp_df,
        x="Component",
        y="Score",
    )

    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#172033"),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### AI Exposure of Current Skills")

    st.caption(
        "This is an analytical indicator, not a guaranteed prediction of job displacement."
    )

    exposure_counts = summarize_exposure(
        analysis["current_skills"]
    )

    if exposure_counts:

        exp_df = pd.DataFrame(
            list(
                exposure_counts.items()
            ),
            columns=[
                "Category",
                "Count",
            ],
        )

        fig2 = px.pie(
            exp_df,
            names="Category",
            values="Count",
            hole=0.42,
        )

        fig2.update_layout(
            paper_bgcolor="white",
            font=dict(color="#172033"),
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

    with st.expander(
        "View skill-level AI exposure"
    ):

        st.dataframe(
            pd.DataFrame(
                analysis["exposure"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Skill Transferability")

    transfer_scores = compute_transferability_scores(
        jobs_df
    )

    transfer_rows = [
        {
            "Skill": skill,
            "Transferability":
                transfer_scores.get(
                    skill,
                    0.0,
                ),
        }
        for skill in analysis[
            "current_skills"
        ]
    ]

    if transfer_rows:

        transfer_df = (
            pd.DataFrame(
                transfer_rows
            )
            .sort_values(
                "Transferability",
                ascending=False,
            )
        )

        st.dataframe(
            transfer_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

with tabs[4]:

    recs = analysis[
        "recommendations"
    ]

    st.markdown(
        "### Priority Skills to Learn"
    )

    if recs["skills_to_learn"]:

        learn_df = pd.DataFrame(
            recs["skills_to_learn"]
        )

        st.dataframe(
            learn_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "Your profile is well aligned with the selected roles."
        )

    st.markdown(
        "### Skills Worth Strengthening"
    )

    if recs["skills_to_strengthen"]:

        strengthen_df = pd.DataFrame(
            recs["skills_to_strengthen"]
        )

        st.dataframe(
            strengthen_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Suggested Career Directions"
    )

    for role in recs[
        "career_directions"
    ]:

        st.write(
            f"• {role}"
        )


# ============================================================
# LEARNING ROADMAP
# ============================================================

with tabs[5]:

    st.markdown(
        "### Personalized Learning Roadmap"
    )

    st.markdown(
        f"""
        <div class="info-card">
            {analysis["roadmap"]["text"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="pf-footer">
        PathForge · Career analytics and skill-gap analysis
    </div>
    """,
    unsafe_allow_html=True,
)