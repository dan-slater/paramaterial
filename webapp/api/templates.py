from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_db
from models import User, AnalysisTemplate
from api.auth import get_current_user

router = APIRouter()

@router.get("/")
async def list_templates(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List available analysis templates"""
    # TODO: Implement template listing
    return []

@router.get("/{template_id}")
async def get_template(
    template_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get template details"""
    # TODO: Implement template retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")

# Export router
templates_router = router