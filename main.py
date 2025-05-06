from os import environ

import uvicorn

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, Optional

from model import ESMModel, ProbabilitiesLogitProcessor, ProteinFunctionLogitProcessor

app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence embeddings using ESM models.",
    version="0.0.1",
)

# General environment variables for model configuration.
model_name = environ.get("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
device = environ.get("ESM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Environment variables for function classification.
function_enabled = environ.get("FUNCTION_ENABLED", "true").lower() == "true"
function_terms_path = environ.get("FUNCTION_TERMS_PATH", "dataset/train_terms.tsv")
function_obo_path = environ.get("FUNCTION_OBO_PATH", "dataset/go-basic.obo")
function_min_prob = float(environ.get("FUNCTION_MIN_PROB", 0.05))

logit_processor = ProbabilitiesLogitProcessor()

if function_enabled:
    logit_processor = ProteinFunctionLogitProcessor.from_files(
        tsv_fpath=function_terms_path,
        obo_fpath=function_obo_path,
        min_probability=function_min_prob,
    )

model = ESMModel(model_name, logit_processor, device)


class HealthResponse(BaseModel):
    status: str


class ClassifyRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")


class ClassifyResponse(BaseModel):
    functions: Optional[list[str]] = Field(
        default=None, description="List of predicted protein functions (GO terms)."
    )
    probabilities: list[float] = Field(
        description="List of probabilities for each class."
    )
    runtime: float = Field(description="Time taken to process the request in seconds.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""

    return {
        "status": "Ok",
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify a protein sequence."""

    global model

    try:
        return model.classify(request.sequence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
