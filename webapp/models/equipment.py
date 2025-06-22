from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import date
from .database import BaseModel

if TYPE_CHECKING:
    from .organization import Organization
    from .template import AnalysisTemplate
    from .job import Job

class Equipment(BaseModel, table=True):
    __tablename__ = 'equipment'
    
    organization_id: str = Field(foreign_key="organizations.id")
    name: str = Field(max_length=200)
    model: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None)
    equipment_type: str = Field(max_length=100)  # 'gleeble', 'sem', 'ebsd', 'tensile_tester', etc.
    # specifications: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    location: Optional[str] = Field(default=None, max_length=200)
    is_active: bool = Field(default=True)
    
    # Additional metadata
    manufacturer: Optional[str] = Field(default=None, max_length=200)
    serial_number: Optional[str] = Field(default=None, max_length=100)
    installation_date: Optional[date] = Field(default=None)
    last_calibration: Optional[date] = Field(default=None)
    # capabilities: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    
    # Relationships
    organization: Optional["Organization"] = Relationship(back_populates="equipment")
    templates: List["AnalysisTemplate"] = Relationship(back_populates="equipment")
    jobs: List["Job"] = Relationship(back_populates="equipment")
    
    def get_template_count(self) -> int:
        """Get number of analysis templates for this equipment"""
        return len(self.templates)
    
    def get_job_count(self) -> int:
        """Get number of jobs run on this equipment"""
        return len(self.jobs)
    
    def is_available(self) -> bool:
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