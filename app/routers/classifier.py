from pydantic import BaseModel, Field

from typing import Optional

from fastapi import APIRouter, Request

router = APIRouter(prefix="/model")


class PredictMulticlassRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")


class PredictBinaryRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")


class RankRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")

    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        description="Return the top k class labels with the highest probabilities.",
    )


class PredictMulticlassResponse(BaseModel):
    labels: Optional[list[str]] = Field(description="List of class labels.")

    probabilities: list[float] = Field(
        description="List of probabilities for each class."
    )


class PredictBinaryResponse(BaseModel):
    labels: Optional[list[str]] = Field(description="List of class labels.")

    probabilities: list[float] = Field(
        description="List of probabilities for each class."
    )


class RankResponse(BaseModel):
    labels: Optional[list[str]] = Field(description="List of ranked class labels.")

    probabilities: list[float] = Field(
        description="List of probabilities for each class."
    )


@router.post("/predictions/multiclass")
async def rank(request: Request, input: PredictMulticlassRequest):
    """Return the class probabilities for a protein sequence."""

    result = request.app.state.model.predict_multiclass(input.sequence)

    return PredictMulticlassResponse(
        labels=result["labels"],
        probabilities=result["probabilities"],
    )


@router.post("/predictions/binary")
async def rank(request: Request, input: PredictBinaryRequest):
    """Return the class probabilities for a protein sequence."""

    result = request.app.state.model.predict_binary(input.sequence)

    return PredictBinaryResponse(
        labels=result["labels"],
        probabilities=result["probabilities"],
    )


@router.post("/predictions/rank")
async def rank(request: Request, input: RankRequest):
    """Return the top k classifications for a protein sequence."""

    result = request.app.state.model.rank(input.sequence, input.top_k)

    return RankResponse(
        labels=result["labels"],
        probabilities=result["probabilities"],
    )
