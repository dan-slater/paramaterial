from sqlalchemy import Column, String, Integer, Text, DateTime, Enum, ForeignKey, BigInteger, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .database import Base, UUIDMixin, TimestampMixin
from datetime import datetime

class Job(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'jobs'
    
    user_id = Column(UUID(as_uuid=False), ForeignKey('users.id'), nullable=False)
    organization_id = Column(UUID(as_uuid=False), ForeignKey('organizations.id'))
    template_id = Column(UUID(as_uuid=False), ForeignKey('analysis_templates.id'))
    equipment_id = Column(UUID(as_uuid=False), ForeignKey('equipment.id'))
    
    status = Column(Enum('pending', 'uploading', 'validating', 'processing', 'completed', 'failed', name='job_status'), 
                   nullable=False, default='pending')
    completed_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    results = Column(JSONB, default=dict)
    analysis_config = Column(JSONB, default=dict)
    processing_parameters = Column(JSONB, default=dict)  # Actual parameters used
    template_version = Column(Integer, default=1)
    
    # Relationships
    user = relationship('User', back_populates='jobs')
    organization = relationship('Organization', back_populates='jobs')
    template = relationship('AnalysisTemplate', back_populates='jobs')
    equipment = relationship('Equipment', back_populates='jobs')
    files = relationship('JobFile', back_populates='job', cascade='all, delete-orphan')
    
    @property
    def is_completed(self):
        """Check if job is completed"""
        return self.status == 'completed'
    
    @property
    def is_failed(self):
        """Check if job failed"""
        return self.status == 'failed'
    
    @property
    def is_processing(self):
        """Check if job is processing"""
        return self.status in ['uploading', 'validating', 'processing']
    
    @property
    def duration(self):
        """Get job duration if completed"""
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return None
    
    def mark_completed(self):
        """Mark job as completed"""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message):
        """Mark job as failed with error"""
        self.status = 'failed'
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

class JobFile(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'job_files'
    
    job_id = Column(UUID(as_uuid=False), ForeignKey('jobs.id'), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(Enum('info_table', 'time_series', name='file_types'), nullable=False)
    file_size = Column(BigInteger)  # in bytes
    storage_path = Column(String(500), nullable=False)  # local file path or cloud storage URL
    upload_completed = Column(Boolean, default=False, nullable=False)
    mime_type = Column(String(100))
    checksum = Column(String(64))  # SHA-256 hash for integrity
    
    # Relationships
    job = relationship('Job', back_populates='files')
    
    @property
    def file_size_mb(self):
        """Get file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
    
    @property
    def is_csv(self):
        """Check if file is CSV"""
        return self.file_name.lower().endswith('.csv')
    
    @property
    def is_excel(self):
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