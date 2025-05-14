from pydantic import BaseModel

from fastapi import APIRouter

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""

    return {
        "status": "Ok",
    }
