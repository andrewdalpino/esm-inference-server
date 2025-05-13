from argparse import ArgumentParser

import uvicorn

from fastapi import FastAPI, HTTPException

from model import ESMClassifier

from http import HealthResponse, PredictRankRequest, PredictRankResponse

parser = ArgumentParser(description="Run the ESM inference server.")

parser.add_argument("--model_name", default="facebook/esm2_t6_8M_UR50D", type=str)
parser.add_argument("--context_length", default=1024, type=int)
parser.add_argument("--device", default="cuda", type=str)

args = parser.parse_args()

app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence inference using ESM models.",
    version="0.0.3",
)

model = ESMClassifier(args.model_name, args.context_length, args.device)


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
