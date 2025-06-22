from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_db
from models import User, Equipment
from api.auth import get_current_user

router = APIRouter()

@router.get("/")
async def list_equipment(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List available equipment"""
    # TODO: Implement equipment listing
    return []

@router.get("/{equipment_id}")
async def get_equipment(
    equipment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get equipment details"""
    # TODO: Implement equipment retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")

# Export router
equipment_router = router