> [!NOTE]
> To be transparent, I am still a ML engineer in training. This project is my first attempt at applying what I've learned in AI engineering from DataCamp to synthetic data pertaining to the type of data I am interested in.
> This attempt was mostly a code along inside of PyCharm, using GPT to answer my questions to gain clarity on functionalities I was not familiar with.
> During these learning experiences I turn off autocompletion functions in my IDE so I can gain more intuition for what code needs to be written. 
> It is my intention to become deeply familiar with the libraries and boilerplate code needed to package and prepare models for deployment. 

# Train → Save → Serve: MicroML CCR (Binary Classification API)

A minimal end-to-end machine learning project that trains a PyTorch binary classifier on a mock “College & Career Ready (CCR)” dataset, saves model artifacts, and serves predictions through a FastAPI endpoint.

## What this demonstrates

- A clean end-to-end ML pipeline: **data → training → saved artifacts → inference → API**
- Clear separation of concerns:
  - `src/` contains machine learning logic (data loading, model definition, training, inference)
  - `app/` contains the web/API layer (request validation and routing)
  - `data/` contains the input dataset
  - `artifacts/` contains generated outputs
- Production-style consistency:
  - fixed feature order enforced as a data contract
  - preprocessing statistics saved to `meta.json` so inference exactly matches training
 
## Project structure

app/ # FastAPI app + request/response schemas
data/ # raw_ccr.csv (mock dataset)
src/ # training + inference code
artifacts/ # meta.json (tracked) + model.pt (generated locally)


## Mock dataset features

The model is trained on a small, synthetic dataset representing student-reported college and career readiness indicators.  
Each row corresponds to a single student.

Input features:

- `career_research_done` — has researched careers (0 or 1)
- `mentor_interview_done` — interviewed a mentor in a desired career field (0 or 1)
- `career_path_chosen` — has selected a career path (0 or 1)
- `postsecondary_plan_set` — has a defined postsecondary plan (0 or 1)
- `resume_prepared` — has prepared a resume or portfolio (0 or 1)
- `internship_or_job_shadow` — has completed an internship or job shadow (0 or 1)
- `applications_submitted` — has submitted at least one application (0 or 1)
- `financial_aid_plan` — has explored or started a financial aid plan (0 or 1)
- `study_habits_score` — self-reported study consistency (1–5)
- `future_confidence_score` — confidence in future plans (1–5)
- `barriers_score` — perceived barriers to progress (0–3)

Target label:

- `ccr_ready` — binary indicator of college & career readiness (0 or 1)

## Example Prediction

{
  "career_research_done": 1,
  "mentor_interview_done": 0,
  "career_path_chosen": 1,
  "postsecondary_plan_set": 1,
  "resume_prepared": 1,
  "internship_or_job_shadow": 0,
  "applications_submitted": 0,
  "financial_aid_plan": 1,
  "study_habits_score": 4,
  "future_confidence_score": 4,
  "barriers_score": 1
}
Example response:

{
  "probability": 0.62,
  "prediction": 1,
  "model_version": "0.1.0"
}



