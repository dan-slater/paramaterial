from sqlmodel import SQLModel, Field
from sqlalchemy import DateTime, func
from datetime import datetime
import uuid
from typing import Optional

# UUID type for use across models
def generate_uuid() -> str:
    return str(uuid.uuid4())

# Base class for all database models with common fields
class BaseModel(SQLModel):
    """Base model with ID and timestamps"""
    id: Optional[str] = Field(
        default_factory=generate_uuid,
        primary_key=True
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={
            "server_default": func.now(),
            "nullable": False
        }
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={
            "server_default": func.now(),
            "onupdate": func.now(),
            "nullable": False
        }
    )

    class Config:
        from_attributes = True