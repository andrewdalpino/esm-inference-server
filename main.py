from os import environ

import uvicorn

from fastapi import FastAPI, HTTPException

from model import ESMClassifier

from .http import HealthResponse, PredictRankRequest, PredictRankResponse

# General environment variables for model configuration.
model_name = environ.get("MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
context_length = int(environ.get("CONTEXT_LENGTH", 1024))
device = environ.get("DEVICE", "cpu")


app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence inference using ESM models.",
    version="0.0.3",
)

model = ESMClassifier(model_name, context_length, device)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""

    return {
        "status": "Ok",
    }


@app.post("/rank", response_model=PredictRankResponse)
async def predict_rank(request: PredictRankRequest):
    """Return the top k binary classifications for a protein sequence."""

    global model

    try:
        return model.predict_rank(request.sequence, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
