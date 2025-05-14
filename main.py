from os import environ

import uvicorn

from fastapi import FastAPI, HTTPException

from model import ESMClassifier

from http_types import HealthResponse, PredictRankRequest, PredictRankResponse

from fastapi.responses import JSONResponse

api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
context_length = int(environ.get("CONTEXT_LENGTH", 1024))
device = environ.get("DEVICE", "cpu")

app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence inference using ESM models.",
    version="0.0.4",
)

model = ESMClassifier(model_name, context_length, device)


@app.middleware("http")
async def exception_handler(request, call_next):
    """Handle exceptions emitted from the domain model."""

    try:
        return await call_next(request)
    except HTTPException as e:
        return JSONResponse(content={"message": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(
            content={"message": "Something went wrong."}, status_code=500
        )


@app.middleware("http")
async def authorization(request, call_next):
    """Check for the API token in the request headers."""

    if api_token:
        token = request.headers.get("Authorization")

        if not token or token != f"Bearer {api_token}":
            return JSONResponse(content={"message": "Unauthorized."}, status_code=401)

    return await call_next(request)


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
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
