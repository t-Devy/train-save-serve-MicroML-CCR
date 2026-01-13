from fastapi import FastAPI

from app.schemas import CCRRequest, CCRResponse
from src.inference import CCRPredictor

app = FastAPI(title="Micro ML API - CCR Ready", version="0.1.0")

# Load once at startup (prevents reloading weights per request)
predictor = CCRPredictor()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": predictor.model_version}

@app.post("/predict", response_model=CCRResponse)
def predict(req: CCRRequest):
    # Convert Pydantic model -> plain dict for our predictor
    features = req.model_dump()

    pred = predictor.predict(features)

    return CCRResponse(
        probability=pred.probability,
        prediction=pred.prediction,
        model_version=pred.model_version,
    )

@app.get("/")
def root():
    return {"message": "Micro ML API is running. Visit /docs"}