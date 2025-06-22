from sqlalchemy import Column, String, Text, Boolean, Date, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .database import Base, UUIDMixin, TimestampMixin

class Equipment(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'equipment'
    
    organization_id = Column(UUID(as_uuid=False), ForeignKey('organizations.id'), nullable=False)
    name = Column(String(200), nullable=False)
    model = Column(String(200))
    description = Column(Text)
    equipment_type = Column(String(100), nullable=False)  # 'gleeble', 'sem', 'ebsd', 'tensile_tester', etc.
    specifications = Column(JSONB, default=dict)
    location = Column(String(200))
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Additional metadata
    manufacturer = Column(String(200))
    serial_number = Column(String(100))
    installation_date = Column(Date)
    last_calibration = Column(Date)
    capabilities = Column(JSONB, default=dict)
    
    # Relationships
    organization = relationship('Organization', back_populates='equipment')
    templates = relationship('AnalysisTemplate', back_populates='equipment')
    jobs = relationship('Job', back_populates='equipment')
    
    def get_template_count(self):
        """Get number of analysis templates for this equipment"""
        return len(self.templates)
    
    def get_job_count(self):
        """Get number of jobs run on this equipment"""
        return len(self.jobs)
    
    def is_available(self):
        """Check if equipment is available for use"""
        return self.is_active
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'name': self.name,
            'model': self.model,
            'description': self.description,
            'equipment_type': self.equipment_type,
            'specifications': self.specifications,
            'location': self.location,
            'is_active': self.is_active,
            'manufacturer': self.manufacturer,
            'serial_number': self.serial_number,
            'installation_date': self.installation_date.isoformat() if self.installation_date else None,
            'last_calibration': self.last_calibration.isoformat() if self.last_calibration else None,
            'capabilities': self.capabilities,
            'template_count': self.get_template_count(),
            'job_count': self.get_job_count(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Equipment {self.name} ({self.equipment_type})>'