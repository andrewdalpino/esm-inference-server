import os
import time

import uvicorn

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from typing import List

from model import ESMModel

app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence embeddings using ESM models.",
    version="0.0.1",
)

model_name = os.environ.get("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
device = os.environ.get("ESM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

model = ESMModel(name=model_name)


class HealthResponse(BaseModel):
    status: str


class SequenceRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence.")


class ClassifyResponse(BaseModel):
    probabilities: List[float] = Field(
        description="List of probabilities for each class."
    )
    runtime: float = Field(description="Time taken to process the request in seconds.")


class Timer:
    def __enter__(self):
        self.start = time.time()
        self.duration = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""

    global model

    return {
        "status": "Ok",
    }


@app.post("/classify", response_model=ClassifyResponse)
async def predict(request: SequenceRequest):
    """Classify a protein sequence."""

    global model

    try:
        with Timer() as timer:
            result = model.classify(request.sequence)

        result.update(
            {
                "runtime": timer.duration,
            }
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
