from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timedelta
from .database import BaseModel

if TYPE_CHECKING:
    from .user import User
    from .organization import Organization
    from .template import AnalysisTemplate
    from .equipment import Equipment

class Job(BaseModel, table=True):
    __tablename__ = 'jobs'
    
    user_id: str = Field(foreign_key="users.id")
    organization_id: Optional[str] = Field(default=None, foreign_key="organizations.id")
    template_id: Optional[str] = Field(default=None, foreign_key="analysis_templates.id")
    equipment_id: Optional[str] = Field(default=None, foreign_key="equipment.id")
    
    status: str = Field(default="pending")
    completed_at: Optional[datetime] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    # results: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    # analysis_config: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    # processing_parameters: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    template_version: int = Field(default=1)
    
    # Relationships
    user: Optional["User"] = Relationship(back_populates="jobs")
    organization: Optional["Organization"] = Relationship(back_populates="jobs")
    template: Optional["AnalysisTemplate"] = Relationship(back_populates="jobs")
    equipment: Optional["Equipment"] = Relationship(back_populates="jobs")
    files: List["JobFile"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed"""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if job failed"""
        return self.status == "failed"
    
    @property
    def is_processing(self) -> bool:
        """Check if job is processing"""
        return self.status in ["uploading", "validating", "processing"]
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get job duration if completed"""
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return None
    
    def mark_completed(self):
        """Mark job as completed"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str):
        """Mark job as failed with error"""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def get_info_table_file(self):
        """Get the info table file"""
        return next((f for f in self.files if f.file_type == 'info_table'), None)
    
    def get_time_series_files(self):
        """Get time series files"""
        return [f for f in self.files if f.file_type == 'time_series']
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'organization_id': self.organization_id,
            'template_id': self.template_id,
            'equipment_id': self.equipment_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'results': self.results,
            'analysis_config': self.analysis_config,
            'processing_parameters': self.processing_parameters,
            'template_version': self.template_version,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'file_count': len(self.files)
        }
    
    def __repr__(self):
        return f'<Job {self.id} ({self.status})>'

class JobFile(BaseModel, table=True):
    __tablename__ = 'job_files'
    
    job_id: str = Field(foreign_key="jobs.id")
    file_name: str = Field(max_length=255)
    file_type: str
    file_size: Optional[int] = Field(default=None)  # in bytes
    storage_path: str = Field(max_length=500)  # local file path or cloud storage URL
    upload_completed: bool = Field(default=False)
    mime_type: Optional[str] = Field(default=None, max_length=100)
    checksum: Optional[str] = Field(default=None, max_length=64)  # SHA-256 hash for integrity
    
    # Relationships
    job: Optional["Job"] = Relationship(back_populates="files")
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0.0
    
    @property
    def is_csv(self) -> bool:
        """Check if file is CSV"""
        return self.file_name.lower().endswith('.csv')
    
    @property
    def is_excel(self) -> bool:
        """Check if file is Excel"""
        return self.file_name.lower().endswith(('.xlsx', '.xls'))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'file_size_mb': self.file_size_mb,
            'storage_path': self.storage_path,
            'upload_completed': self.upload_completed,
            'mime_type': self.mime_type,
            'checksum': self.checksum,
            'is_csv': self.is_csv,
            'is_excel': self.is_excel,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<JobFile {self.file_name} ({self.file_type})>'