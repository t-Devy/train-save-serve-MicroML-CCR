from pathlib import Path

# Needs project root folder containing /src, /data, /app etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "career_research_done",
    "mentor_interview_done",
    "career_path_chosen",
    "postsecondary_plan_set",
    "resume_prepared",
    "internship_or_job_shadow",
    "applications_submitted",
    "financial_aid_plan",
    "study_habits_score",
    "future_confidence_score",
    "barriers_score",

]

TARGET_COLUMN = "ccr_ready"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_ccr.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
META_PATH = ARTIFACT_DIR / "meta.json"