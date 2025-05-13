from pydantic import BaseModel, Field

from typing import Optional, Any


class HealthResponse(BaseModel):
    status: str


class PredictRankRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")

    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        description="Return the top k class labels with the highest probabilities.",
    )


class PredictRankResponse(BaseModel):
    labels: Optional[list[str]] = Field(
        description="List of predicted protein functions (GO terms)."
    )
    probabilities: list[float] = Field(
        description="List of probabilities for each class."
    )
    meta: dict[str, Any] = Field(description="Metadata about the request.")
