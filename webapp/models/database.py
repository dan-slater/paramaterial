from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

# Initialize SQLAlchemy
db = SQLAlchemy()

# UUID type for use across models
def generate_uuid():
    return str(uuid.uuid4())

# Mixin for common fields
class TimestampMixin:
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UUIDMixin:
    id = db.Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)