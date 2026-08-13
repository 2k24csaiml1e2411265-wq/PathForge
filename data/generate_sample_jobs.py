"""
generate_sample_jobs.py
------------------------
Builds data/jobs.csv — a SMALL, SYNTHETIC sample of technology job postings
used for demonstration purposes only.

IMPORTANT (per project spec): the real PathForge design targets a dataset of
roughly 6,520 technology job postings (e.g. a scraped/licensed LinkedIn or
Kaggle tech-jobs dataset). That real dataset is NOT bundled with this
repository because of licensing/size restrictions. This script instead
generates a small, clearly-labelled sample so the rest of the pipeline
(embeddings, FAISS search, skill-gap, market analysis, ...) can be
demonstrated end-to-end on a normal laptop.

To use a real dataset later, see data/ingest_real_dataset.py.
"""
import csv
import itertools
import random
from pathlib import Path

random.seed(42)

ROLES = {
    "Data Scientist": {
        "skills": ["python", "machine learning", "pandas", "numpy", "sql",
                   "scikit-learn", "statistics", "data analysis", "deep learning"],
        "industry": ["Technology", "Finance", "E-commerce", "Healthcare"],
    },
    "Machine Learning Engineer": {
        "skills": ["python", "tensorflow", "pytorch", "machine learning", "docker",
                   "aws", "sql", "deep learning", "rest api", "git"],
        "industry": ["Technology", "Automotive", "Healthcare"],
    },
    "NLP Engineer": {
        "skills": ["python", "natural language processing", "spacy",
                   "hugging face transformers", "sentence transformers",
                   "pytorch", "llm", "faiss", "git"],
        "industry": ["Technology", "Media", "Finance"],
    },
    "Data Analyst": {
        "skills": ["sql", "python", "data analysis", "statistics", "pandas",
                   "communication", "presentation"],
        "industry": ["E-commerce", "Finance", "Retail", "Technology"],
    },
    "Backend Developer": {
        "skills": ["python", "java", "django", "flask", "rest api", "sql",
                   "postgresql", "docker", "git", "linux"],
        "industry": ["Technology", "E-commerce", "Finance"],
    },
    "Full Stack Developer": {
        "skills": ["javascript", "react", "node.js", "express.js", "mongodb",
                   "rest api", "git", "html", "css", "typescript"],
        "industry": ["Technology", "Startups", "E-commerce"],
    },
    "Cloud/DevOps Engineer": {
        "skills": ["aws", "azure", "docker", "kubernetes", "ci/cd", "linux",
                   "bash", "git", "cloud computing"],
        "industry": ["Technology", "Finance", "Telecom"],
    },
    "AI Product Manager": {
        "skills": ["generative ai", "communication", "leadership",
                   "problem solving", "data analysis", "presentation"],
        "industry": ["Technology", "Consulting"],
    },
    "Computer Vision Engineer": {
        "skills": ["python", "computer vision", "pytorch", "tensorflow",
                   "deep learning", "opencv", "docker", "aws"],
        "industry": ["Technology", "Automotive", "Retail"],
    },
    "Database Administrator": {
        "skills": ["sql", "mysql", "postgresql", "mongodb", "linux",
                   "aws", "problem solving"],
        "industry": ["Technology", "Finance", "Healthcare"],
    },
}

COMPANIES = [
    "Nimbus Analytics", "Orbitwise Tech", "Vertex Data Labs", "Skyline Systems",
    "Bluepeak Software", "Northwind AI", "Cognivo", "Datamint",
    "Corestack Solutions", "Silverleaf Cloud", "Brightloop Inc", "Fernwood Digital",
    "Hexalytics", "Ironclad Systems", "Lumenware", "Meridian Softworks",
    "Quantalabs", "Riverstone Tech", "Solstice AI", "Trailmark Data",
]

LOCATIONS = [
    "Bengaluru, India", "Pune, India", "Hyderabad, India", "Gurugram, India",
    "Noida, India", "Chennai, India", "Remote", "Mumbai, India",
]

SENIORITY = ["Intern", "Entry-level", "Mid-level", "Senior"]

DESC_TEMPLATE = (
    "We are looking for a {seniority} {title} to join our {industry} team. "
    "The ideal candidate has hands-on experience with {skills_text}. "
    "You will work on real-world problems, collaborate with cross-functional "
    "teams, and help ship production-quality solutions."
)


def build_rows(n_per_role: int = 8):
    rows = []
    job_id = 1
    for title, meta in ROLES.items():
        for i in range(n_per_role):
            seniority = random.choice(SENIORITY)
            industry = random.choice(meta["industry"])
            company = random.choice(COMPANIES)
            location = random.choice(LOCATIONS)
            # vary the required skill subset a bit per posting
            k = random.randint(max(3, len(meta["skills"]) - 4), len(meta["skills"]))
            skills = random.sample(meta["skills"], k=k)
            skills_text = ", ".join(skills)
            description = DESC_TEMPLATE.format(
                seniority=seniority.lower(), title=title,
                industry=industry.lower(), skills_text=skills_text
            )
            rows.append({
                "job_id": job_id,
                "job_title": title,
                "company": company,
                "location": location,
                "description": description,
                "skills": "; ".join(skills),
                "seniority": seniority,
                "industry": industry,
            })
            job_id += 1
    return rows


def main():
    rows = build_rows(n_per_role=8)
    out_path = Path(__file__).parent / "jobs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "job_id", "job_title", "company", "location",
            "description", "skills", "seniority", "industry",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sample job postings to {out_path}")


if __name__ == "__main__":
    main()
