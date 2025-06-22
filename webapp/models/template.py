from sqlalchemy import Column, String, Text, Boolean, Integer, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from .database import Base, UUIDMixin, TimestampMixin

class AnalysisTemplate(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'analysis_templates'
    
    organization_id = Column(UUID(as_uuid=False), ForeignKey('organizations.id'), nullable=False)
    equipment_id = Column(UUID(as_uuid=False), ForeignKey('equipment.id'))
    created_by = Column(UUID(as_uuid=False), ForeignKey('users.id'), nullable=False)
    
    name = Column(String(200), nullable=False)
    description = Column(Text)
    template_type = Column(Enum('processing', 'analysis', 'visualization', name='template_types'), 
                          nullable=False)
    template_data = Column(JSONB, nullable=False, default=dict)
    parameters = Column(JSONB, nullable=False, default=dict)
    is_public = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    tags = Column(ARRAY(String), default=list)
    
    # Template versioning
    version = Column(Integer, default=1, nullable=False)
    parent_template_id = Column(UUID(as_uuid=False), ForeignKey('analysis_templates.id'))
    
    # Relationships
    organization = relationship('Organization', back_populates='templates')
    equipment = relationship('Equipment', back_populates='templates')
    creator = relationship('User', back_populates='created_templates')
    jobs = relationship('Job', back_populates='template')
    parent_template = relationship('AnalysisTemplate', remote_side='AnalysisTemplate.id')
    
    def increment_usage(self):
        """Increment usage count when template is used"""
        self.usage_count += 1
    
    def is_accessible_by(self, user):
        """Check if template is accessible by user"""
        if self.is_public:
            return True
        if self.created_by == user.id:
            return True
        # Check if user is member of organization
        return user.is_member_of(self.organization_id)
    
    def clone(self, new_name, created_by):
        """Create a clone of this template"""
        clone = AnalysisTemplate(
            organization_id=self.organization_id,
            equipment_id=self.equipment_id,
            created_by=created_by,
            name=new_name,
            description=f"Clone of {self.name}",
            template_type=self.template_type,
            template_data=self.template_data.copy(),
            parameters=self.parameters.copy(),
            is_public=False,
            tags=self.tags.copy() if self.tags else [],
            parent_template_id=self.id
        )
        return clone
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'equipment_id': self.equipment_id,
            'created_by': self.created_by,
            'name': self.name,
            'description': self.description,
            'template_type': self.template_type,
            'template_data': self.template_data,
            'parameters': self.parameters,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'tags': self.tags,
            'version': self.version,
            'parent_template_id': self.parent_template_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<AnalysisTemplate {self.name} ({self.template_type})>'