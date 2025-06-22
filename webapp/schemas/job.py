from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FileType(str, Enum):
    INFO_TABLE = "info_table"
    TIME_SERIES = "time_series"

class JobFileBase(BaseModel):
    file_name: str
    file_type: FileType
    file_size: int
    mime_type: Optional[str] = None

class JobFileResponse(JobFileBase):
    id: str
    job_id: str
    storage_path: str
    upload_completed: bool
    uploaded_at: datetime
    
    class Config:
        from_attributes = True

class JobBase(BaseModel):
    organization_id: Optional[str] = None
    template_id: Optional[str] = None
    equipment_id: Optional[str] = None

class JobCreate(JobBase):
    pass

class JobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    template_id: Optional[str] = None
    analysis_config: Optional[dict] = None

class JobResponse(JobBase):
    id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    analysis_config: Optional[dict] = None
    results: Optional[dict] = None
    error_message: Optional[str] = None
    files: List[JobFileResponse] = []
    
    class Config:
        from_attributes = True

class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int
    page: int
    per_page: int
    pages: int