from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Create the declarative base
Base = declarative_base()

# UUID type for use across models
def generate_uuid():
    return str(uuid.uuid4())

# Mixin for common fields
class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UUIDMixin:
    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)