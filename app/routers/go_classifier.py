from pydantic import BaseModel, Field
from typing import Any

from fastapi import APIRouter, Request

import networkx as nx


class PredictTermsRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")


class PredictTermsResponse(BaseModel):
    probabilities: dict[str, float] = Field(
        description="List of GO terms and their probabilities."
    )


class PredictSubgraphRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein sequence to classify.")

    top_p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum probability threshold for GO terms to be included in the subgraph.",
    )


class PredictSubgraphResponse(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    subgraph: str = Field(
        description="A subgraph of the GO DAG containing the predicted terms and their relationships."
    )

    probabilities: dict[str, float] = Field(
        description="List of predicted GO terms and their probabilities."
    )


router = APIRouter(prefix="/model")


@router.post("/gene-ontology/terms")
async def predict_terms(request: Request, input: PredictTermsRequest):
    """Return the GO term probabilities for a protein sequence."""

    probabilities = request.app.state.model.predict_terms(input.sequence)

    return PredictTermsResponse(probabilities=probabilities)


@router.post("/gene-ontology/subgraph")
async def predict_subgraph(request: Request, input: PredictSubgraphRequest):
    """Return the GO subgraph for a protein sequence."""

    subgraph, probabilities = request.app.state.model.predict_subgraph(
        input.sequence, input.top_p
    )

    subgraph_json = nx.node_link_data(subgraph, edges="edges")

    return PredictSubgraphResponse(
        subgraph=subgraph_json,
        probabilities=probabilities,
    )
