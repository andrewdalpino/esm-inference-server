from os import environ

import uvicorn

from fastapi import FastAPI

from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import health, classifier

from models import ESMClassifier


api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
model_type = environ.get("MODEL_TYPE", "classifier")
context_length = int(environ.get("CONTEXT_LENGTH", 1024))
device = environ.get("DEVICE", "cpu")

app = FastAPI(
    title="ESM Inference API",
    description="API for protein sequence inference using ESM models.",
    version="0.0.5",
)

app.add_middleware(ExceptionHandler)

if api_token:
    app.add_middleware(TokenAuthentication, api_token=api_token)

app.add_middleware(ResponseTime)

match model_type:
    case "classifier":
        model = ESMClassifier(model_name, context_length, device)

        app.include_router(classifier.router)
    case _:
        raise ValueError(f"Unsupported model type: {model_type}")

app.state.model = model

app.include_router(health.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
