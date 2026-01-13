from pydantic import BaseModel, Field

# this is the schema, or public promise of the system
# does range and field checks for incoming data

class CCRRequest(BaseModel):
    career_research_done: int = Field(ge=0, le=1)
    mentor_interview_done: int = Field(ge=0, le=1)
    career_path_chosen: int = Field(ge=0, le=1)
    postsecondary_plan_set: int = Field(ge=0, le=1)
    resume_prepared: int = Field(ge=0, le=1)
    internship_or_job_shadow: int = Field(ge=0, le=1)
    applications_submitted: int = Field(ge=0, le=1)
    financial_aid_plan: int = Field(ge=0, le=1)
    study_habits_score: int = Field(ge=1, le=5)
    future_confidence_score: int = Field(ge=1, le=5)
    barriers_score: int = Field(ge=0, le=3)


class CCRResponse(BaseModel):
    probability: float
    prediction: int
    model_version: str

