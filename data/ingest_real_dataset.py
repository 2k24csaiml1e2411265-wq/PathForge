"""
ingest_real_dataset.py
-----------------------
Helper script to plug a REAL, larger technology-jobs dataset (e.g. the
~6,520-row dataset referenced in the project spec, sourced from a licensed
provider such as Kaggle/LinkedIn exports) into PathForge, replacing the
small synthetic sample in data/jobs.csv.

PathForge does NOT ship with that real dataset — it is not included in this
repository for licensing/size reasons. Use this script to convert whatever
raw file you obtain into the schema PathForge expects.

Usage
-----
    python data/ingest_real_dataset.py --input path/to/raw_jobs.csv \
        --title-col "Job Title" --company-col "Company" \
        --location-col "Location" --description-col "Job Description" \
        --skills-col "Skills" --seniority-col "Seniority" --industry-col "Industry"

Any column you don't have can be omitted — PathForge will fill it with a
sensible default ("Unknown" / empty string) so the pipeline still runs.

Required output schema (data/jobs.csv):
    job_id, job_title, company, location, description, skills, seniority, industry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "job_id", "job_title", "company", "location",
    "description", "skills", "seniority", "industry",
]


def ingest(
    input_path: str,
    output_path: str = "data/jobs.csv",
    title_col: str | None = None,
    company_col: str | None = None,
    location_col: str | None = None,
    description_col: str | None = None,
    skills_col: str | None = None,
    seniority_col: str | None = None,
    industry_col: str | None = None,
) -> pd.DataFrame:
    src = pd.read_csv(input_path)

    mapping = {
        "job_title": title_col,
        "company": company_col,
        "location": location_col,
        "description": description_col,
        "skills": skills_col,
        "seniority": seniority_col,
        "industry": industry_col,
    }

    out = pd.DataFrame()
    out["job_id"] = range(1, len(src) + 1)
    for target_col, source_col in mapping.items():
        if source_col and source_col in src.columns:
            out[target_col] = src[source_col].fillna("").astype(str)
        else:
            out[target_col] = "" if target_col != "skills" else ""

    # Basic sanity: drop rows with no title and no description at all
    out = out[(out["job_title"].str.strip() != "") | (out["description"].str.strip() != "")]
    out = out.reset_index(drop=True)
    out["job_id"] = range(1, len(out) + 1)

    out = out[REQUIRED_COLUMNS]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest a real jobs dataset into PathForge's schema.")
    p.add_argument("--input", required=True, help="Path to the raw dataset CSV")
    p.add_argument("--output", default="data/jobs.csv", help="Where to write the normalized CSV")
    p.add_argument("--title-col")
    p.add_argument("--company-col")
    p.add_argument("--location-col")
    p.add_argument("--description-col")
    p.add_argument("--skills-col")
    p.add_argument("--seniority-col")
    p.add_argument("--industry-col")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        ingest(
            input_path=args.input,
            output_path=args.output,
            title_col=args.title_col,
            company_col=args.company_col,
            location_col=args.location_col,
            description_col=args.description_col,
            skills_col=args.skills_col,
            seniority_col=args.seniority_col,
            industry_col=args.industry_col,
        )
    except FileNotFoundError:
        print(f"ERROR: could not find input file '{args.input}'", file=sys.stderr)
        sys.exit(1)
