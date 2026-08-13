# PathForge

### Agentic AI-Based Career Intelligence, Skill Gap Analysis & Career Resilience Platform

PathForge is an AI-powered career intelligence platform that analyzes a candidate's resume against technology job requirements, identifies skill gaps, evaluates AI-era career resilience, and generates a personalized learning roadmap.

Unlike traditional keyword-based career tools, PathForge combines NLP, semantic search, market skill analysis, explainable scoring, and Generative AI to turn a resume into an actionable career plan.

---

## Problem Statement

The rapid adoption of Artificial Intelligence is changing the technology workforce and reshaping the skills demanded by industry. Traditional career platforms mainly rely on keyword-based job matching and often overlook critical skill gaps, market demand, skill transferability, and the potential impact of AI on different capabilities.

PathForge addresses this problem by helping users understand:

- What skills they already have
- Which skills are missing for relevant roles
- Which skills are in higher market demand
- How transferable their skills are across roles
- Which capabilities are more exposed or complementary to AI
- What they should learn and build next

---

## Key Features

- **Resume Analysis** — Upload PDF/TXT resumes with automatic text extraction and cleaning.
- **NLP Skill Extraction** — Extracts and normalizes technical skills using spaCy and rule-based matching.
- **Semantic Job Matching** — Uses Sentence Transformers and FAISS to retrieve relevant jobs beyond exact keyword matching.
- **Skill Gap Analysis** — Calculates skill coverage, missing skills, and critical gaps.
- **Market Demand Analysis** — Identifies frequently requested skills from the available job dataset.
- **Skill Transferability** — Estimates how broadly a skill applies across related roles.
- **AI Exposure Analysis** — Categorizes skills as AI-substitutable, AI-augmented, or AI-complementary using configurable heuristics.
- **Career Resilience Score** — Produces an explainable 0–100 indicator from documented weighted components.
- **Recommendations** — Ranks skills to learn, skills to strengthen, projects, and career directions.
- **Gemini Career Guidance** — Generates personalized career summaries and a five-phase roadmap when a Gemini API key is available.
- **Fallback Mode** — The application remains functional without Gemini or internet access.
- **Interactive Dashboard** — Six-section Streamlit interface for exploring results.

---

## Architecture

```text
Resume / Profile
       ↓
PDF / Text Extraction
       ↓
Text Cleaning
       ↓
NLP Skill Extraction
       ↓
Skill Normalization
       ↓
Candidate Skill Profile
       ↓
Sentence Embeddings
       ↓
FAISS Semantic Search
       ↓
Relevant Job Retrieval
       ↓
Skill Gap Analysis
       ↓
Market Demand + Transferability
       ↓
AI Exposure Analysis
       ↓
Career Resilience Score
       ↓
Recommendation Engine
       ↓
Gemini / Fallback Narrative
       ↓
Personalized Learning Roadmap
       ↓
Streamlit Dashboard